"""Isolated PPO recovery-cost experiment; no deployment guard changes."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv
from experiments.recovery_curriculum import RecoveryNavigationCurriculum
from experiments.static_deadlock_curriculum import StaticDeadlockCurriculum
from experiments.prior_anchored_ppo import PriorAnchoredPPO
from experiments.feasibility_feedback_ppo import FeasibilityFeedbackPPO
from experiments.predictive_policy import migrate_predictive_policy
from neural_navigation.safety_context import OBSERVATION_DIM, PREDICTIVE_DIM, PREDICTIVE_SCHEMA, SCHEMA, RECOVERY_SCHEMA, contract, validate_contract
from train_safety_context import ContextProgress


class RecoveryProgress(ContextProgress):
    def __init__(self,*args,schema,**kwargs):
        super().__init__(*args,**kwargs);self.schema=schema
    def _on_step(self):
        result=super()._on_step()
        if self.num_timesteps%1000<self.training_env.num_envs:
            path=self.output/'progress.json';data=json.loads(path.read_text())
            data['context_schema']=self.schema;path.write_text(json.dumps(data,indent=2))
            if hasattr(self.model,'last_anchor_statistics'):
                data['reference_anchor']=self.model.last_anchor_statistics
                path.write_text(json.dumps(data,indent=2))
            if hasattr(self.model,'last_feedback_statistics'):
                data.pop('reference_anchor',None)
                data['feasibility_feedback']=self.model.last_feedback_statistics
                path.write_text(json.dumps(data,indent=2))
        return result


def migrate_probe_columns(model,source,target):
    if source==target:return None
    if source!=SCHEMA or target!=RECOVERY_SCHEMA:
        raise ValueError('Only explicit V2-to-V3 probe migration is supported')
    for network in (model.policy.mlp_extractor.policy_net,model.policy.mlp_extractor.value_net):
        weight=network[0].weight
        if weight.ndim!=2 or weight.shape[1]!=OBSERVATION_DIM:
            raise ValueError('Flat context actor/critic required')
        with torch.no_grad():weight[:,240:244].zero_()
        state=model.policy.optimizer.state.get(weight,{})
        for key in ('exp_avg','exp_avg_sq','max_exp_avg_sq'):
            if key in state:state[key][:,240:244].zero_()
    return dict(source=source,target=target,reset_input_columns=[240,241,242,243],
        reset_actor_and_critic=True,reset_corresponding_optimizer_moments=True,
        other_weights_and_optimizer_retained=True)


def environment(components, templates, static_deadlock=False):
    def create():
        torch.set_num_threads(1)
        curriculum=StaticDeadlockCurriculum if static_deadlock else RecoveryNavigationCurriculum
        env=curriculum(components,stage='robust',route_mode='shortcut',
            motion_profiles=True,footprint_clearance_cost=1.,failure_templates=templates)
        keywords=(
            'verified_mission_success','collision_events','safety_termination',
            'training_case_json','execution_guard_override_fraction',
            'repeated_rejection_penalty','repeated_rejection_seconds')
        if static_deadlock:keywords+=('training_static_deadlock_termination','static_deadlock_seconds')
        return Monitor(env,str(components/'worker_0.csv'),info_keywords=keywords)
    return create


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--failure-pool',type=Path,required=True)
    p.add_argument('--steps',type=int,default=16000)
    p.add_argument('--seed',type=int,default=92101)
    p.add_argument('--probe-schema',choices=(SCHEMA,RECOVERY_SCHEMA,PREDICTIVE_SCHEMA),default=SCHEMA)
    p.add_argument('--static-deadlock',action='store_true')
    p.add_argument('--anchor-strength',type=float,default=0.)
    p.add_argument('--feasibility-feedback',action='store_true')
    p.add_argument('--manifest',type=Path,action='append',required=True)
    a=p.parse_args()
    if a.steps<=0 or a.output.exists():p.error('Positive steps and fresh output required')
    if not 0 <= a.anchor_strength <= 1. or (a.static_deadlock and a.anchor_strength):
        p.error('Use finite anchor strength 0..1, isolated from static-deadlock reward')
    if a.feasibility_feedback and (a.static_deadlock or a.anchor_strength or a.probe_schema != RECOVERY_SCHEMA):
        p.error('Feasibility feedback requires V3 and an isolated auxiliary objective')
    a.checkpoint=a.checkpoint.resolve();a.output=a.output.resolve()
    components=a.checkpoint.parent
    hardware=json.loads((components/'hardware.json').read_text())
    validate_contract(hardware.get('policy_observation'))
    if hardware['policy_observation']['schema'] not in (SCHEMA,RECOVERY_SCHEMA):p.error('V2/V3 checkpoint required')
    parent_contract=hardware['policy_observation']
    if parent_contract['schema']==RECOVERY_SCHEMA and a.probe_schema not in (RECOVERY_SCHEMA,PREDICTIVE_SCHEMA):
        p.error('V3 checkpoint must retain V3 probe semantics')
    if a.probe_schema==PREDICTIVE_SCHEMA and (parent_contract['schema']!=RECOVERY_SCHEMA or a.static_deadlock):
        p.error('Predictive migration requires a V3 parent and no static-deadlock objective')
    hardware['policy_observation']=contract(a.probe_schema)
    pool=json.loads(a.failure_pool.read_text())
    if pool.get('kind')!='training' or not pool.get('templates'):p.error('Training-only failure pool required')
    a.output.mkdir(parents=True);candidate=a.output/'candidate';candidate.mkdir()
    for name in ('hardware.json','localizer.pt','router.pt'):
        shutil.copy2(components/name,candidate/name)
    (candidate/'hardware.json').write_text(json.dumps(hardware,indent=2))
    shutil.copy2(a.failure_pool,candidate/'failure_pool.json')
    project=Path(__file__).resolve().parents[2]
    config=dict(status='initializing',requested_steps=a.steps,num_envs=1,seed=a.seed,
        parent_checkpoint=str(a.checkpoint),parent_checkpoint_sha256=hashlib.sha256(a.checkpoint.read_bytes()).hexdigest(),
        policy_observation=hardware['policy_observation'],reward_unchanged_from_guarded_clearance=False,
        repeated_rejection_cost=dict(grace_s=1.,maximum_per_step=.2,ramp_s=2.,
            only_rejected_translation=True,require_accepted_probe=True,exempt_voluntary_wait=True),
        physical_constraints_unchanged=True,guard_unchanged=True,localizer_router_frozen=True,
        parent_probe_schema=parent_contract['schema'],
        failure_template_fraction=.4,optimizer_retained=True,exact_rollout_or_rng_resume=False,
        deployment_approved=False,final_acceptance_used=False,launch_source_sha256={})
    config['static_deadlock_terminal_cost']=(dict(patience_s=3., maximum_pose_change_m=.03,
        penalty=25., require_static_origin_outside_free=True, require_no_accepted_probe=True,
        require_accepted_scan_localization=True, minimum_map_agreement=.8,
        sensor_dynamic_wait_exempt=True, runtime_unchanged=True) if a.static_deadlock else None)
    config['reference_mean_anchor']=dict(enabled=bool(a.anchor_strength),strength=a.anchor_strength,scale=.3,
        frozen_warm_start_policy=bool(a.anchor_strength),require_all_probes_accepted=True,previous_stop_reason_none=True,
        minimum_corridor_m=.8,minimum_waypoint_heading_cos=.2,
        exclude_teacher_reverse=True,training_only=True,runtime_unchanged=True)
    config['feasibility_feedback']=dict(enabled=a.feasibility_feedback,strength=.1,
        minimum_rejected_stationary_seconds=1.,nearest_current_accepted_probe=True,
        clipped_mean_straight_through_gradient=True,reference_actor_used=False,
        future_safety_guaranteed=False,training_only=True,runtime_unchanged=True)
    # Preserve actual dependencies, including the previous training entry from
    # which the exploration-floor/progress callback is imported.
    for source in sorted((project/'training').rglob('*.py')):
        relative=source.relative_to(project)
        if 'runs' in relative.parts:continue
        config['launch_source_sha256'][str(relative)]=hashlib.sha256(source.read_bytes()).hexdigest()
        target=candidate/'code_snapshot'/relative;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(source,target)
    status=dict(status='running',phase='initializing',deployment_approved=False,started_unix=time.time())
    def save():
        (candidate/'config.json').write_text(json.dumps(config,indent=2))
        (a.output/'status.json').write_text(json.dumps(status,indent=2))
    save();torch.set_num_threads(1)
    env=None
    try:
        env=SubprocVecEnv([environment(candidate,pool['templates'],a.static_deadlock)],start_method='spawn')
        env.seed(a.seed)
        algorithm=FeasibilityFeedbackPPO if a.feasibility_feedback else PriorAnchoredPPO if a.anchor_strength else PPO
        extra=dict(anchor_strength=.1 if a.feasibility_feedback else a.anchor_strength,anchor_scale=.3) if (a.anchor_strength or a.feasibility_feedback) else {}
        if a.probe_schema==PREDICTIVE_SCHEMA:
            model=migrate_predictive_policy(a.checkpoint,algorithm=algorithm,seed=a.seed,**extra)
            model.set_env(env)
            config['predictive_migration']=model.predictive_migration
            config['reference_mean_anchor']['exclude_supported_stationary_forecast_risk']=True
        else:
            model=algorithm.load(a.checkpoint,env=env,device='cpu',seed=a.seed,
                n_steps=250,batch_size=250,n_epochs=4,learning_rate=3e-5,target_kl=.02,ent_coef=.003,**extra)
        expected=PREDICTIVE_DIM if a.probe_schema==PREDICTIVE_SCHEMA else OBSERVATION_DIM
        if model.observation_space.shape!=(expected,):raise ValueError('Context policy required')
        if a.probe_schema!=parent_contract['schema'] and a.probe_schema!=PREDICTIVE_SCHEMA:
            # These four columns now describe different velocities. Erase
            # their old associations and Adam moments; retain all other
            # weights, optimization state and exploration variance.
            config['probe_schema_migration']=migrate_probe_columns(model,parent_contract['schema'],a.probe_schema)
        config['prior_checkpoint_steps']=model.num_timesteps
        if a.anchor_strength:
            model.snapshot_reference()
            torch.save(model.reference_policy.state_dict(),candidate/'anchor_reference.pt')
            config['reference_mean_anchor']['reference_state_sha256']=hashlib.sha256((candidate/'anchor_reference.pt').read_bytes()).hexdigest()
        model.save(candidate/'policy_initial.zip')
        config['status']='running';status['phase']='training';save()
        model.learn(total_timesteps=a.steps,callback=RecoveryProgress(candidate,None,None,None,a.steps,2000,0,
            schema=a.probe_schema),reset_num_timesteps=True)
        model.save(candidate/'policy.zip');env.close();env=None
        config.update(status='complete',additional_ppo_steps_completed=model.num_timesteps);save()
        status['phase']='export';save()
        portable=a.output/'portable'
        with (a.output/'export.log').open('w') as log:
            subprocess.run([sys.executable,'training/export_navigation_actor.py','--policy',str(candidate/'policy.zip'),
                '--components',str(candidate),'--output',str(portable)],cwd=project,stdout=log,stderr=subprocess.STDOUT,check=True)
        summaries=[]
        for index,manifest in enumerate(a.manifest):
            status['phase']=f'runtime_{index}';save()
            with (a.output/f'runtime_{index}.log').open('w') as log:
                subprocess.run([sys.executable,'training/evaluate_sensor_navigation.py','--bundle',str(portable),
                    '--manifest',str(manifest.resolve()),'--workers','1','--output',str(a.output/f'runtime_{index}.json'),
                    '--trace-dir',str(a.output/f'traces_{index}')],cwd=project,stdout=log,stderr=subprocess.STDOUT,check=True)
            report=json.loads((a.output/f'runtime_{index}.json').read_text())
            summaries.append({k:report[k] for k in ('completed','safe_success','tasks','collisions')})
        status.update(status='complete',summaries=summaries,development_gate_passed=all(
            x['completed'] and x['safe_success']==x['tasks'] for x in summaries),finished_unix=time.time());save()
    except Exception as error:
        status.update(status='failed',error=str(error));save();raise
    finally:
        if env is not None:env.close()


if __name__=='__main__':main()
