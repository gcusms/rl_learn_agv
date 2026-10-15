"""Matched PPO continuation: old versus refreshed reference, same starting actor."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
import time
import torch
from sb3_contrib import MaskablePPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from experiments.quality_masked_navigation import resume_quality
from experiments.recovery_sampling import RecoveryCoverageProgress
from experiments.train_quality_navigation import make_environment,export,evaluate,digest
from neural_navigation.quality_prediction_context import contract


def state_hash(model):
    value=hashlib.sha256()
    for name,tensor in sorted(model.policy.state_dict().items()):
        value.update(name.encode());value.update(tensor.detach().cpu().numpy().tobytes())
    return value.hexdigest()


def main():
    parser=argparse.ArgumentParser()
    for name in ('checkpoint','old-reference','failure-pool','training-log','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--replay-step',type=int,required=True)
    parser.add_argument('--steps',type=int,default=1000);parser.add_argument('--seed',type=int,default=92531)
    args=parser.parse_args()
    if args.output.exists() or args.steps<=0 or args.steps%250:parser.error('Fresh output, positive multiple of250 steps required')
    pool=json.loads(args.failure_pool.read_text())
    if pool['kind']!='training':parser.error('Training-only templates required')
    records=[json.loads(s) for s in args.training_log.read_text().splitlines()]
    matches=[r for r in records if r.get('step')==args.replay_step and 'worker' in r and 'training_case_json' in r]
    if len(matches)!=1:parser.error('Unique recorded training task required')
    case=json.loads(matches[0]['training_case_json'])
    if case['stage']!='robust':parser.error('Robust training task required')
    if json.loads((args.checkpoint.parent/'quality_observation.json').read_text())!=contract():parser.error('Quality contract required')
    args.output.mkdir();torch.set_num_threads(1);results={};initial_hash=None
    project=Path(__file__).resolve().parents[2];sources={}
    for path in (project/'training').rglob('*.py'):
        relative=path.relative_to(project)
        if 'runs' in relative.parts:continue
        sources[str(relative)]=digest(path);target=args.output/'source_snapshot'/relative
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target)
    config=dict(parent_policy_sha256=digest(args.checkpoint),old_reference_sha256=digest(args.old_reference),
        source_sha256=sources,steps_per_arm=args.steps,seed=args.seed,training_log_sha256=digest(args.training_log),
        failure_pool_sha256=digest(args.failure_pool),recorded_training_step=args.replay_step,
        reference_strength=2.,reference_kl_budget=.02,scheduled_recovery=True,num_envs=1,deployment_approved=False)
    (args.output/'config.json').write_text(json.dumps(config,indent=2))
    status=dict(status='running',started_unix=time.time())
    def save():(args.output/'status.json').write_text(json.dumps(status,indent=2))
    save()
    try:
        for label,teacher_path in [('old_reference',args.old_reference),('refreshed_reference',args.checkpoint)]:
            root=args.output/label;root.mkdir();components=root/'candidate';components.mkdir()
            for name in ('hardware.json','localizer.pt','router.pt','quality_observation.json'):
                shutil.copy2(args.checkpoint.parent/name,components/name)
            parent=MaskablePPO.load(args.checkpoint,device='cpu');scale=float(parent.training_reward_scale);del parent
            env=SubprocVecEnv([make_environment(components,pool['templates'],[case],args.seed,True,scale,True)],start_method='spawn')
            try:
                env.seed(args.seed);model=resume_quality(args.checkpoint,env,seed=args.seed)
                teacher=MaskablePPO.load(teacher_path,device='cpu')
                if teacher.observation_space!=model.observation_space or teacher.action_space!=model.action_space:
                    raise ValueError('Reference architecture/observation mismatch')
                model.reference_policy=copy.deepcopy(teacher.policy);del teacher
                model.reference_policy.set_training_mode(False);model.reference_policy.requires_grad_(False)
                # Loading differently seeded teachers resets global RNGs;
                # restore the identical sampling seed before either rollout.
                model.set_random_seed(args.seed)
                current_hash=state_hash(model)
                if initial_hash is not None and current_hash!=initial_hash:raise ValueError('Matched arms start from different weights')
                initial_hash=current_hash;model.save(components/'policy_initial.zip')
                status.update(phase='training',arm=label);save()
                callback=RecoveryCoverageProgress(components,args.steps)
                model.learn(args.steps,callback=callback,reset_num_timesteps=True)
                if callback.changed:raise ValueError('Nonzero execution mismatch')
                model.save(components/'policy.zip')
                metadata=dict(config,arm=label,initial_weights_sha256=current_hash,training_reward_scale=scale,
                    reference_policy_sha256=digest(teacher_path),completed_steps=model.num_timesteps,
                    reference_statistics=model.last_reference_statistics)
                (components/'config.json').write_text(json.dumps(metadata,indent=2))
            finally:env.close()
            export(components/'policy.zip',components,root/'portable')
            status.update(phase='history_diagnostic',arm=label);save()
            results[label]=evaluate(components/'policy.zip',components,pool['templates'],case,root/'history.json')
            (args.output/'COMPARISON.json').write_text(json.dumps(dict(results=results,completed=len(results)==2,
                limitation='Single-seed matched reference intervention on a reused training task; not independent safety/generalization acceptance.'),indent=2))
        status.update(status='complete',finished_unix=time.time());save()
    except Exception as error:
        status.update(status='failed',error=str(error));save();raise


if __name__=='__main__':main()
