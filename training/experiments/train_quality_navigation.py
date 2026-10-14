"""Small quality-observation PPO pilot and matched sensor-core diagnostics.

No promotion. Fresh seeded tasks are development diagnostics, not acceptance.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time
import numpy as np
import torch
from sb3_contrib import MaskablePPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv
from experiments.masked_navigation import FocusedMaskedCurriculum,PortableMaskedActor
from experiments.quality_masked_navigation import QualityMaskedWrapper,QualityAnchoredPPO,warm_start_quality,resume_quality
from experiments.fixed_training_replay import reset_training_episode
from experiments.train_masked_navigation import Progress
from experiments.recovery_sampling import ScheduledRecoveryCurriculum,RecoveryCoverageProgress
from experiments.audit_stop_feedback import audit
from neural_navigation.quality_prediction_context import contract,TOTAL_DIM
from neural_navigation.masked_actions import ACTIONS,masks_from_observation,action_contract


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def make_environment(components,templates,cases,seed,monitor=False,reward_scale=1.,scheduled_recovery=False):
    def create():
        torch.set_num_threads(1)
        curriculum=ScheduledRecoveryCurriculum if scheduled_recovery else FocusedMaskedCurriculum
        base=curriculum(components,stage='robust',route_mode='shortcut',
            motion_profiles=True,footprint_clearance_cost=1.,failure_templates=templates,
            recorded_cases=cases,replay_probability=.25,replay_seed=seed)
        wrapped=QualityMaskedWrapper(base,reward_scale=reward_scale)
        return Monitor(wrapped,str(components/'worker_0.csv'),info_keywords=(
            'verified_mission_success','collision_events','safety_termination','training_case_json')) if monitor else wrapped
    return create


def export(policy,components,output):
    output.mkdir();model=MaskablePPO.load(policy,device='cpu')
    if model.observation_space.shape!=(TOTAL_DIM,) or model.action_space.n!=12:
        raise ValueError('Quality actor dimensions required')
    actor=PortableMaskedActor(model.policy).eval()
    inputs=np.random.default_rng(29).uniform(-1,1,(256,TOTAL_DIM)).astype(np.float32)
    inputs[:,235:244]=np.random.default_rng(30).integers(0,2,(256,9))
    inputs[:,267:269]=np.random.default_rng(31).integers(0,2,(256,2));inputs[0,235:244]=0;inputs[0,267:269]=0
    expected=ACTIONS[model.predict(inputs,action_masks=masks_from_observation(inputs[:,:282]),deterministic=True)[0]]
    traced=torch.jit.trace(actor,torch.from_numpy(inputs))
    with torch.no_grad():actual=traced(torch.from_numpy(inputs)).numpy()
    np.testing.assert_allclose(actual,expected,atol=0,rtol=0)
    traced.save(str(output/'actor.pt'))
    for name in ('hardware.json','localizer.pt','router.pt','quality_observation.json'):
        shutil.copy2(components/name,output/name)
    manifest=dict(policy_sha256=digest(policy),observation_contract=contract(),action_contract=action_contract(),
        comparison_inputs=256,exact_action_parity=True,deployment_approved=False,
        requires_quality_feature_generator=True,
        files={p.name:digest(p) for p in output.iterdir() if p.is_file()})
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2))


def evaluate(policy,components,templates,task,output):
    model=MaskablePPO.load(policy,device='cpu');actor=torch.jit.load(str(output.parent/'portable/actor.pt'))
    # Use the corresponding export, never the final actor for initial results.
    if policy.name=='policy_initial.zip':actor=torch.jit.load(str(output.parent/'initial_portable/actor.pt'))
    env=make_environment(components,templates,(),0)();rows=[]
    try:
        if isinstance(task,int):obs,_=env.reset(seed=task);case=json.loads(env.env.training_case_json)
        else:
            case=task;env.prediction.reset()
            prefix,_=reset_training_episode(env.env,case);obs=env.prediction.augment(prefix,env.env.core)
        for _ in range(env.env.max_steps):
            masks=env.action_masks();choice=int(model.predict(obs,action_masks=masks,deterministic=True)[0])
            with torch.no_grad():portable=actor(torch.from_numpy(obs[None])).numpy()[0]
            np.testing.assert_allclose(portable,ACTIONS[choice],atol=0,rtol=0)
            if not masks[choice]:raise ValueError('Selected rejected action')
            row=dict(timestamp=env.env.active.elapsed,estimate=env.env.core.estimate.tolist(),
                policy_observation=obs.tolist(),choice=choice,mask=masks.tolist(),proposed_action=ACTIONS[choice].tolist())
            obs,_,done,truncated,info=env.step(choice)
            if info['nonzero_choice_changed_by_guard']:raise ValueError('Nonzero execution mismatch')
            row.update(command=info['commanded_velocity'].tolist(),stop_reason=info['execution_guard_reason']);rows.append(row)
            if done or truncated:break
        prefix_trace=dict(case=case,samples=[dict(r,policy_observation=r['policy_observation'][:282]) for r in rows])
        feedback=audit(prefix_trace)
        if feedback['counts']['missing'] or feedback['counts']['feedback_mismatch']:raise ValueError('Feedback mismatch')
        result=dict(policy_sha256=digest(policy),safe_success=bool(info['verified_mission_success']),
            collisions=int(info['collision_events']),timeout=bool(truncated),seconds=env.env.active.elapsed,
            travel_m=env.env.active.travel,frames=len(rows),
            voluntary_wait_frames=sum(r['choice']==0 and any(r['mask'][1:]) for r in rows),
            only_stop_frames=sum(not any(r['mask'][1:]) for r in rows))
        output.write_text(json.dumps(dict(case=case,samples=rows,result=result,feedback=feedback),allow_nan=False))
        return result
    finally:env.close()


def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--failure-pool',type=Path,required=True)
    p.add_argument('--training-log',type=Path,required=True);p.add_argument('--replay-step',type=int,required=True)
    p.add_argument('--steps',type=int,default=2000);p.add_argument('--seed',type=int,default=92401)
    p.add_argument('--resume-quality',action='store_true')
    p.add_argument('--scheduled-recovery',action='store_true',help='Alternate recorded training and general episodes')
    a=p.parse_args()
    if a.output.exists() or a.steps<=0 or a.steps%250:p.error('Fresh output and positive multiple of250 steps required')
    pool=json.loads(a.failure_pool.read_text())
    if pool.get('kind')!='training':p.error('Training-only templates required')
    records=[json.loads(x) for x in a.training_log.read_text().splitlines()]
    matches=[r for r in records if r.get('step')==a.replay_step and 'worker' in r and 'training_case_json' in r]
    if len(matches)!=1:p.error('Unique recorded training episode required')
    case=json.loads(matches[0]['training_case_json'])
    if case.get('stage')!='robust':p.error('Robust training episode required')
    parent=MaskablePPO.load(a.checkpoint,device='cpu')
    reward_scale=float(getattr(parent,'training_reward_scale',1.))
    if not np.isfinite(reward_scale) or not 0<reward_scale<=1:p.error('Invalid parent reward units')
    if a.resume_quality:
        if parent.observation_space.shape!=(TOTAL_DIM,):p.error('300-field resume source required')
        if json.loads((a.checkpoint.parent/'quality_observation.json').read_text())!=contract():p.error('Quality contract mismatch')
    elif reward_scale!=1.:p.error('Normalized checkpoint requires --resume-quality')
    del parent
    a.output=a.output.resolve();a.checkpoint=a.checkpoint.resolve();a.output.mkdir()
    components=a.output/'candidate';components.mkdir()
    for name in ('hardware.json','localizer.pt','router.pt'):shutil.copy2(a.checkpoint.parent/name,components/name)
    (components/'quality_observation.json').write_text(json.dumps(contract(),indent=2))
    project=Path(__file__).resolve().parents[2];source={}
    for path in (project/'training').rglob('*.py'):
        relative=path.relative_to(project)
        if 'runs' in relative.parts:continue
        source[str(relative)]=digest(path);target=components/'code_snapshot'/relative
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target)
    config=dict(parent_policy_sha256=digest(a.checkpoint),steps=a.steps,seed=a.seed,
        quality_contract=contract(),source_sha256=source,replay_training_step=a.replay_step,
        failure_pool_sha256=digest(a.failure_pool),training_log_sha256=digest(a.training_log),
        evaluation_seeds=list(range(938000,938008)),evaluation_kind='fresh development seeds plus one reused training failure',
        reward_unchanged=reward_scale==1.,native_reward_unchanged=True,training_reward_scale=reward_scale,
        resume_quality=a.resume_quality,guard_unchanged=True,num_envs=1,reference_strength=2.,reference_kl_budget=.02,
        scheduled_recovery=a.scheduled_recovery,
        localizer_router_frozen=True,deployment_approved=False)
    (components/'config.json').write_text(json.dumps(config,indent=2));status=dict(status='running',phase='initializing',started_unix=time.time())
    def save():(a.output/'status.json').write_text(json.dumps(status,indent=2))
    save();torch.set_num_threads(1);env=None
    try:
        env=SubprocVecEnv([make_environment(components,pool['templates'],[case],a.seed,True,reward_scale,a.scheduled_recovery)],start_method='spawn');env.seed(a.seed)
        model=(resume_quality(a.checkpoint,env,seed=a.seed) if a.resume_quality else
            warm_start_quality(a.checkpoint,env,seed=a.seed,model_class=QualityAnchoredPPO,reference_strength=2.,reference_kl_budget=.02))
        model.training_reward_scale=reward_scale
        model.snapshot_reference();model.save(components/'policy_initial.zip')
        export(components/'policy_initial.zip',components,a.output/'initial_portable')
        status['phase']='training';save()
        callback=(RecoveryCoverageProgress if a.scheduled_recovery else Progress)(components,a.steps)
        model.learn(a.steps,callback=callback,reset_num_timesteps=True)
        if callback.changed:raise ValueError('Masked nonzero execution disagreement')
        model.save(components/'policy.zip');env.close();env=None
        config.update(completed_steps=model.num_timesteps,reference_statistics=model.last_reference_statistics)
        (components/'config.json').write_text(json.dumps(config,indent=2));export(components/'policy.zip',components,a.output/'portable')
        results=[]
        for index,task in enumerate([case]+config['evaluation_seeds']):
            status.update(phase='matched_diagnostic',completed_cases=index);save()
            pair={}
            for label,filename in [('initial','policy_initial.zip'),('final','policy.zip')]:
                pair[label]=evaluate(components/filename,components,pool['templates'],task,a.output/f'{label}_{index}.json')
            results.append(pair)
            (a.output/'MATCHED_RESULTS.json').write_text(json.dumps(dict(results=results,completed=len(results)==9),indent=2))
        status.update(status='complete',finished_unix=time.time(),development_gate_passed=all(r['final']['safe_success'] for r in results));save()
    except Exception as error:
        status.update(status='failed',error=str(error));save();raise
    finally:
        if env is not None:env.close()


if __name__=='__main__':main()
