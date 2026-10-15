"""Positive-advantage self-imitation losses for recorded training experience.

Based on Oh et al., ICML 2018, equations 2/3. Current masks and safety-only
data admission are engineering additions; this is not a full SIL reproduction.
"""
import numpy as np
import torch


def validate_demonstration(data,gamma,dimensions=300):
    if (data.get('kind')!='reproduced_safe_training_demonstration'
        or not data.get('observations_and_masks_reproduced') or not data.get('safe_success')
        or data.get('collisions')!=0 or data.get('timeout') or data.get('gamma')!=gamma):
        raise ValueError('Reproduced safe training experience with matching discount required')
    rows=data['transitions']
    if not rows or not rows[-1]['terminated'] or any(r['terminated'] for r in rows[:-1]) or any(r['truncated'] for r in rows):
        raise ValueError('One complete non-truncated episode required')
    obs=np.asarray([r['observation'] for r in rows],np.float32)
    actions=np.asarray([r['action'] for r in rows]);masks=np.asarray([r['action_mask'] for r in rows],bool)
    rewards=np.asarray([r['reward'] for r in rows],float);returns=np.asarray([r['return'] for r in rows],float)
    if obs.shape!=(len(rows),dimensions) or masks.shape!=(len(rows),12) or actions.dtype.kind not in 'iu':
        raise ValueError('Native observation, mask and discrete action dimensions required')
    if not np.isfinite(np.r_[obs.ravel(),rewards,returns]).all() or np.any(actions<0) or np.any(actions>=12):
        raise ValueError('Finite experience and valid choices required')
    if not masks[np.arange(len(rows)),actions].all():raise ValueError('Demonstration contains rejected actions')
    from neural_navigation.masked_actions import masks_from_observation
    np.testing.assert_array_equal(masks,masks_from_observation(obs[:,:282]))
    from experiments.replay_training_demonstration import discounted_returns
    np.testing.assert_allclose(returns,discounted_returns(rewards,gamma),atol=1e-8,rtol=0)
    return obs,actions.astype(np.int64),masks,returns.astype(np.float32)


def losses(policy,observations,actions,masks,returns):
    if masks.shape!=(len(observations),12) or actions.shape!=(len(observations),) or returns.shape!=(len(observations),):
        raise ValueError('Aligned demonstration batch required')
    if not masks.bool().gather(1,actions[:,None]).all():raise ValueError('Rejected demonstration action')
    values,log_prob,_=policy.evaluate_actions(observations,actions,action_masks=masks)
    advantage=torch.clamp(returns-values.flatten(),min=0.)
    actor=-(log_prob*advantage.detach()).mean()
    critic=.5*advantage.square().mean()
    return actor,critic,dict(positive_advantage_states=int((advantage>0).sum()),
        mean_positive_advantage=float(advantage.detach().mean()))
