"""Conservative masked PPO pilot, frozen learned localization/routing, no promotion."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import time
from importlib.metadata import version
import torch
from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv
from experiments.masked_navigation import MaskedNavigationCurriculum,FocusedMaskedCurriculum,warm_start
from experiments.anchored_masked_ppo import AnchoredMaskedPPO
from experiments.export_masked_navigation import export_actor
from neural_navigation.masked_actions import action_contract
from neural_navigation.safety_context import contract,PREDICTIVE_SCHEMA,PREDICTIVE_SCHEMAS,BOUNDED_PREDICTIVE_SCHEMA,RECOVERY_SCHEMA,validate_contract


def environment(components,templates,recorded_cases=(),replay_probability=.25,replay_seed=0):
    def create():
        torch.set_num_threads(1)
        kwargs=dict(stage='robust',route_mode='shortcut',motion_profiles=True,
            footprint_clearance_cost=1.,failure_templates=templates)
        env=(FocusedMaskedCurriculum(components,recorded_cases=recorded_cases,
            replay_probability=replay_probability,replay_seed=replay_seed,**kwargs)
            if recorded_cases else MaskedNavigationCurriculum(components,**kwargs))
        return Monitor(env,str(components/'worker_0.csv'),info_keywords=(
            'verified_mission_success','collision_events','safety_termination','training_case_json'))
    return create


class Progress(BaseCallback):
    def __init__(self,output,steps):
        super().__init__();self.output=output;self.steps=steps;self.started=time.monotonic()
        self.episodes=0;self.arrivals=0;self.changed=0;self.nonzero=0;self.wait_with_alternative=0
    def _on_step(self):
        for info in self.locals['infos']:
            self.changed+=int(info['nonzero_choice_changed_by_guard'])
            self.nonzero+=int(info['masked_choice']!=0)
            self.wait_with_alternative+=int(info['masked_chose_wait_with_nonzero_available'])
            if 'episode' in info:
                self.episodes+=1;self.arrivals+=int(info['episode']['verified_mission_success'])
                with (self.output/'episodes.jsonl').open('a') as f:
                    f.write(json.dumps(dict(info['episode'],step=self.num_timesteps,worker=0))+'\n')
        if self.num_timesteps%500==0 or self.num_timesteps>=self.steps:
            elapsed=time.monotonic()-self.started
            progress=dict(steps=self.num_timesteps,requested_steps=self.steps,elapsed_seconds=elapsed,
                steps_per_second=self.num_timesteps/max(elapsed,1e-6),episodes=self.episodes,
                training_missions_verified=self.arrivals,nonzero_choices=self.nonzero,
                nonzero_choices_changed_by_guard=self.changed,wait_with_nonzero_alternative=self.wait_with_alternative)
            progress['reference_statistics']=getattr(self.model,'last_reference_statistics',{})
            (self.output/'progress.json').write_text(json.dumps(progress,indent=2))
        if self.num_timesteps%1000==0:self.model.save(self.output/f'policy_{self.num_timesteps}.zip')
        return True


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--failure-pool',type=Path,required=True)
    parser.add_argument('--manifest',type=Path,action='append',required=True)
    parser.add_argument('--steps',type=int,default=4000)
    parser.add_argument('--seed',type=int,default=92201)
    parser.add_argument('--rectangular-guard',action='store_true')
    parser.add_argument('--bounded-pose-risk',action='store_true')
    parser.add_argument('--replay-training-log',type=Path)
    parser.add_argument('--replay-step',type=int,action='append')
    parser.add_argument('--replay-probability',type=float,default=.25)
    parser.add_argument('--reference-strength',type=float,default=0.)
    parser.add_argument('--reference-kl-budget',type=float,default=.02)
    args=parser.parse_args()
    if args.output.exists() or args.steps<=0 or args.steps%250:
        parser.error('Fresh output and positive multiple of250 steps required')
    if not all(math.isfinite(x) for x in (args.reference_strength,args.reference_kl_budget)) or args.reference_strength<0 or args.reference_kl_budget<=0:
        parser.error('Nonnegative reference strength and positive finite KL budget required')
    args.checkpoint=args.checkpoint.resolve();args.output=args.output.resolve()
    components=args.checkpoint.parent;hardware=json.loads((components/'hardware.json').read_text())
    validate_contract(hardware['policy_observation'])
    if hardware['policy_observation']['schema'] not in (RECOVERY_SCHEMA,)+PREDICTIVE_SCHEMAS:
        parser.error('V3/V4 parent required')
    resume_masked='policy_action' in hardware
    if args.reference_strength and not resume_masked:
        parser.error('Reference protection requires a native masked checkpoint')
    if resume_masked and hardware['policy_action']!=action_contract():
        parser.error('Supported masked velocity contract required')
    pool=json.loads(args.failure_pool.read_text())
    if pool.get('kind')!='training' or not pool.get('templates'):parser.error('Training-only templates required')
    recorded_cases=[]
    if bool(args.replay_training_log)!=bool(args.replay_step) or not 0<=args.replay_probability<=.5:
        parser.error('Recorded training log and steps must be supplied together; probability 0..0.5')
    if args.replay_training_log:
        records=[json.loads(line) for line in args.replay_training_log.read_text().splitlines()]
        for step in args.replay_step:
            matches=[row for row in records if row.get('step')==step and 'worker' in row and 'training_case_json' in row]
            if len(matches)!=1:parser.error('Unique recorded training episode required')
            case=json.loads(matches[0]['training_case_json'])
            if case.get('stage')!='robust':parser.error('Recorded robust training tasks only')
            case['focused_replay_origin']=dict(training_log=str(args.replay_training_log.resolve()),step=step)
            recorded_cases.append(case)
    for path in args.manifest:
        if json.loads(path.read_text()).get('kind') not in ('development','training_failure_replay'):
            parser.error('Acceptance data must remain sealed')
    target_schema=BOUNDED_PREDICTIVE_SCHEMA if args.bounded_pose_risk else (
        hardware['policy_observation']['schema'] if resume_masked else PREDICTIVE_SCHEMA)
    hardware['policy_observation']=contract(target_schema);hardware['policy_action']=action_contract()
    if args.rectangular_guard:
        hardware['near_field_memory']=dict(hardware.get('near_field_memory',{}),
            static_footprint_schema='oriented_rectangle_v1',static_footprint_margin_m=.04,
            static_footprint_pose_sigma_scale=0.)
    args.output.mkdir(parents=True);candidate=args.output/'candidate';candidate.mkdir()
    for name in ('localizer.pt','router.pt'):shutil.copy2(components/name,candidate/name)
    (candidate/'hardware.json').write_text(json.dumps(hardware,indent=2))
    shutil.copy2(args.failure_pool,candidate/'failure_pool.json')
    if recorded_cases:(candidate/'focused_training_cases.json').write_text(json.dumps(recorded_cases,indent=2))
    project=Path(__file__).resolve().parents[2];snapshots={}
    for path in sorted((project/'training').rglob('*.py')):
        relative=path.relative_to(project)
        if 'runs' in relative.parts:continue
        snapshots[str(relative)]=hashlib.sha256(path.read_bytes()).hexdigest()
        target=candidate/'code_snapshot'/relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target)
    config=dict(requested_steps=args.steps,seed=args.seed,num_envs=1,device='cpu',
        parent_checkpoint=str(args.checkpoint),parent_checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        policy_observation=hardware['policy_observation'],policy_action=action_contract(),
        rectangular_guard=args.rectangular_guard,nominal_pose_geometry_only=args.rectangular_guard,
        bounded_pose_risk=args.bounded_pose_risk,continued_masked_actor=resume_masked,
        focused_training_replay_probability=args.replay_probability if recorded_cases else 0.,
        recorded_training_cases=recorded_cases,
        reference_protection=dict(strength=args.reference_strength,kl_budget=args.reference_kl_budget,
            reference_checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest() if args.reference_strength else None,
            scope='sampled clear sensor states and training-only FIFO cache',
            deployment_reference=False,global_safety_guarantee=False),
        packages={'stable-baselines3':version('stable-baselines3'),
            'sb3-contrib':version('sb3-contrib'),'torch':torch.__version__},
        deployment_approved=False,final_acceptance_used=False,localizer_router_frozen=True,
        velocity_limits_unchanged=True,reward_unchanged_from_recovery=True,
        launch_source_sha256=snapshots)
    status=dict(status='running',phase='initializing',started_unix=time.time(),deployment_approved=False)
    def save():
        (candidate/'config.json').write_text(json.dumps(config,indent=2))
        (args.output/'status.json').write_text(json.dumps(status,indent=2))
    save();torch.set_num_threads(1);env=None
    try:
        env=SubprocVecEnv([environment(candidate,pool['templates'],recorded_cases,args.replay_probability,args.seed)],start_method='spawn');env.seed(args.seed)
        if resume_masked:
            model_class=AnchoredMaskedPPO if args.reference_strength else MaskablePPO
            model=model_class.load(args.checkpoint,device='cpu')
            if model.observation_space.shape!=(282,) or model.action_space.n!=12:
                raise ValueError('Native masked policy dimensions required')
            model.set_random_seed(args.seed)
            config['migration']=dict(actor_critic_and_optimizer_retained=True,
                observation_schema_changed=target_schema!=json.loads((components/'hardware.json').read_text())['policy_observation']['schema'],
                exact_rollout_or_rng_resume=False)
        else:
            model=warm_start(args.checkpoint,seed=args.seed)
            config['migration']=model.masking_migration
        model.set_env(env)
        if args.reference_strength:
            model.reference_strength=args.reference_strength
            model.reference_kl_budget=args.reference_kl_budget
            model.snapshot_reference()
        model.save(candidate/'policy_initial.zip')
        export_actor(candidate/'policy_initial.zip',candidate,args.output/'initial_portable')
        status['phase']='training';save();callback=Progress(candidate,args.steps)
        model.learn(args.steps,callback=callback,reset_num_timesteps=True)
        model.save(candidate/'policy.zip');env.close();env=None
        config['completed_steps']=model.num_timesteps;config['nonzero_choices_changed_by_guard']=callback.changed
        config['final_reference_statistics']=getattr(model,'last_reference_statistics',{})
        if callback.changed:raise ValueError('Current masks and actual nonzero execution disagree')
        status['phase']='export';save();export_actor(candidate/'policy.zip',candidate,args.output/'portable')
        # Same masked representation/guard before learning, evaluated on the
        # first diagnostic manifest after training. No evaluation transitions
        # enter the PPO buffer. This separates filtering from learned changes.
        status['phase']='initial_runtime';save()
        with (args.output/'initial_runtime.log').open('w') as log:
            subprocess.run([sys.executable,'training/evaluate_sensor_navigation.py',
                '--bundle',str(args.output/'initial_portable'),'--manifest',str(args.manifest[0].resolve()),
                '--workers','1','--output',str(args.output/'initial_runtime.json'),
                '--trace-dir',str(args.output/'initial_traces')],cwd=project,
                stdout=log,stderr=subprocess.STDOUT,check=True)
        summaries=[]
        for index,manifest in enumerate(args.manifest):
            status['phase']=f'runtime_{index}';save()
            with (args.output/f'runtime_{index}.log').open('w') as log:
                subprocess.run([sys.executable,'training/evaluate_sensor_navigation.py',
                    '--bundle',str(args.output/'portable'),'--manifest',str(manifest.resolve()),
                    '--workers','1','--output',str(args.output/f'runtime_{index}.json'),
                    '--trace-dir',str(args.output/f'traces_{index}')],cwd=project,
                    stdout=log,stderr=subprocess.STDOUT,check=True)
            report=json.loads((args.output/f'runtime_{index}.json').read_text())
            summaries.append({k:report[k] for k in ('completed','tasks','safe_success','collisions')})
        status.update(status='complete',summaries=summaries,finished_unix=time.time(),
            development_gate_passed=all(x['completed'] and x['tasks']==x['safe_success'] for x in summaries));save()
    except Exception as error:
        status.update(status='failed',error=str(error));save();raise
    finally:
        if env is not None:env.close()


if __name__=='__main__':main()
