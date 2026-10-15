"""Critic pilot on two terminal training episodes, third trajectory held out."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np
import torch
from sb3_contrib import MaskablePPO
from experiments.critic_recalibration import fit_critic
from experiments.replay_training_demonstration import discounted_returns
from neural_navigation.masked_actions import masks_from_observation


def read_episode(path,gamma):
    data=json.loads(path.read_text());rows=data['transitions']
    if (data['kind'] not in ('reproduced_safe_training_demonstration','reproduced_failed_training_episode')
        or not data['observations_and_masks_reproduced'] or data['gamma']!=gamma or data['timeout']
        or not rows[-1]['terminated'] or any(r['terminated'] for r in rows[:-1]) or any(r['truncated'] for r in rows)):
        raise ValueError('Reproduced complete terminal training episode required')
    if (data['kind']=='reproduced_safe_training_demonstration' and (not data['safe_success'] or data['collisions']!=0)) or (data['kind']=='reproduced_failed_training_episode' and (data['safe_success'] or data['collisions']<1)):
        raise ValueError('Terminal outcome and episode type disagree')
    obs=np.asarray([r['observation'] for r in rows],np.float32)
    returns=np.asarray([r['return'] for r in rows],np.float32)
    if obs.shape!=(len(rows),300) or not np.isfinite(np.r_[obs.ravel(),returns]).all():raise ValueError('Finite quality inputs required')
    np.testing.assert_array_equal(masks_from_observation(obs[:,:282]),[r['action_mask'] for r in rows])
    # Serialized returns are double precision; avoid losing precision first.
    np.testing.assert_allclose([r['return'] for r in rows],discounted_returns([r['reward'] for r in rows],gamma),atol=1e-8,rtol=0)
    return torch.from_numpy(obs),torch.from_numpy(returns)


def metrics(model,obs,targets,scale):
    with torch.no_grad():values=model.policy.predict_values(obs).flatten()*scale
    return dict(frames=len(obs),raw_reward_mse=float(torch.mean((values-targets)**2)),
        raw_value_std=float(values.std()),raw_value_min=float(values.min()),raw_value_max=float(values.max()))


def actor_hash(model):
    value=hashlib.sha256()
    for name,tensor in sorted(model.policy.state_dict().items()):
        if name.startswith(('mlp_extractor.policy_net.','action_net.')):
            value.update(name.encode());value.update(tensor.detach().cpu().numpy().tobytes())
    return value.hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--train-episode',type=Path,action='append',required=True)
    p.add_argument('--validation-episode',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--updates',type=int,default=600);a=p.parse_args()
    if a.output.exists() or not 1<=a.updates<=2000:p.error('Fresh output and 1..2000 updates required')
    if a.validation_episode.resolve() in [p.resolve() for p in a.train_episode]:p.error('Trajectory holdout must be separate')
    torch.set_num_threads(1);model=MaskablePPO.load(a.checkpoint,device='cpu')
    if getattr(model,'training_reward_scale',1.)!=1.:raise ValueError('Raw-unit parent required')
    episodes=[read_episode(p,model.gamma) for p in a.train_episode];train=[torch.cat([x[i] for x in episodes]) for i in range(2)]
    validation=read_episode(a.validation_episode,model.gamma);before_actor=actor_hash(model)
    before=dict(train=metrics(model,*train,1.),validation=metrics(model,*validation,1.))
    history=fit_critic(model.policy,*train,updates=a.updates,restore_reward_units=False)
    after=dict(train=metrics(model,*train,60.),validation=metrics(model,*validation,60.))
    if actor_hash(model)!=before_actor:raise ValueError('Actor changed during critic calibration')
    a.output.mkdir();candidate=a.output/'candidate';candidate.mkdir()
    for name in ('hardware.json','localizer.pt','router.pt','quality_observation.json'):
        shutil.copy2(a.checkpoint.parent/name,candidate/name)
    model.training_reward_scale=1/60.;model.critic_calibration_updates=a.updates
    model.save(candidate/'policy.zip')
    report=dict(before=before,after=after,actor_weights_exactly_unchanged=True,actor_hash=before_actor,
        actor_optimizer_preserved=True,critic_optimizer_cleared=True,reward_scale=1/60.,
        training_episodes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in a.train_episode},
        validation_episode=str(a.validation_episode),validation_sha256=hashlib.sha256(a.validation_episode.read_bytes()).hexdigest(),
        parent_sha256=hashlib.sha256(a.checkpoint.read_bytes()).hexdigest(),updates=history,
        deployment_approved=False,limitation='Trajectory holdout from same training task, not map generalization or safety. '
            'Actor unchanged; future PPO must scale native rewards by1/60 to match critic units.')
    (a.output/'CALIBRATION.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({k:report[k] for k in ['before','after','actor_weights_exactly_unchanged','reward_scale']}))


if __name__=='__main__':main()
