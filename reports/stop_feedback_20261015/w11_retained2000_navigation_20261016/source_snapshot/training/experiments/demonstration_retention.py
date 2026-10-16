"""Optional training-only replay of verified legal recovery demonstrations.

The caller must obtain examples with recovery_demonstrations.load_demonstrations.
Replay supplements PPO's actor loss; it never changes a sampled action, mask,
old likelihood, reward or policy input, and is absent from the exported actor.
"""
import hashlib
import math

import numpy as np
import torch

from experiments.stop_recovery_policy import masks_tensor


def configure(model, observations, masks, choices, coefficient=.05, batch_size=128, seed=0):
    raw_choices, raw_masks = np.asarray(choices), np.asarray(masks)
    if (not np.issubdtype(raw_choices.dtype, np.integer)
            or not np.isin(raw_masks, [0, 1]).all()):
        raise ValueError('Integer discrete labels and boolean masks required')
    observations = np.array(observations, dtype=np.float32, copy=True)
    masks = np.array(masks, dtype=bool, copy=True)
    choices = np.array(choices, dtype=np.int64, copy=True)
    if (observations.ndim != 2 or observations.shape[1] != 436
            or masks.shape != (len(observations), 12)
            or choices.shape != (len(observations),)
            or not np.isfinite(observations).all()
            or not math.isfinite(coefficient) or not 0 < coefficient <= .5
            or not isinstance(batch_size, int) or batch_size < 2
            or batch_size > 1024 or batch_size % 2):
        raise ValueError('Finite 436D masked demonstrations and bounded balanced replay required')
    if (not len(choices) or np.any((choices < 0) | (choices >= 12))
            or not masks[np.arange(len(choices)), choices].all()):
        raise ValueError('Replay labels must be legal in their original masks')
    x = torch.from_numpy(observations).to(model.device)
    with torch.no_grad():
        if (not np.array_equal(masks_tensor(x).cpu().numpy(), masks)
                or not model.policy.recovery_support(x).all()):
            raise ValueError('Replay must preserve original masks and trainable recovery support')
    stop = np.flatnonzero(choices == 0)
    movement = np.flatnonzero(choices != 0)
    if not len(stop) or not len(movement):
        raise ValueError('Retain both verified waiting and movement examples')
    digest = hashlib.sha256()
    for array in (observations, masks, choices):
        digest.update(array.tobytes())
    model.demonstration_retention_contract = dict(
        schema='masked_balanced_demo_retention_v1', coefficient=float(coefficient),
        batch_size=batch_size, training_examples=len(choices),
        stop_examples=len(stop), movement_examples=len(movement),
        examples_sha256=digest.hexdigest(),
        update_support='PPO minibatches with trainable recovery actor samples',
        independent_checkpointed_sampling_rng=True, environment_steps_added=0,
        runtime_teacher=False, actions_masks_likelihoods_unchanged=True)
    model._demonstration_retention = dict(observations=observations, masks=masks,
        choices=choices, stop=stop, movement=movement,
        generator=np.random.default_rng(seed))
    return model.demonstration_retention_contract


def loss(model):
    """Return a balanced supervised term without touching PPO's rollout buffer."""
    state = model._demonstration_retention
    specification = model.demonstration_retention_contract
    half = specification['batch_size'] // 2
    generator = state['generator']
    selected = np.concatenate([generator.choice(state['stop'], half),
                               generator.choice(state['movement'], half)])
    observations = torch.as_tensor(state['observations'][selected], device=model.device)
    choices = torch.as_tensor(state['choices'][selected], device=model.device)
    distribution = model.policy.get_distribution(observations, state['masks'][selected])
    value = -distribution.log_prob(choices).mean()
    if not torch.isfinite(value):
        raise ValueError('Nonfinite masked demonstration retention loss')
    return value, len(selected)
