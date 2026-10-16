"""Optional training loss for saturated stops in observed, supported waits.

The uniform prior covers original legal choices, including stop. This changes
gradients only; it never samples, replaces an action, changes a mask, or runs in
the exported actor. A legal motion is not a guarantee of useful future motion.
"""
import math

import torch

from experiments.stop_recovery_policy import masks_tensor
from neural_navigation.decision_feedback import STAT_SLICE


def configure(model, coefficient=.001, minimum_wait_seconds=2.):
    if (not math.isfinite(coefficient) or not 0 < coefficient <= .01
            or not math.isfinite(minimum_wait_seconds)
            or not 1. <= minimum_wait_seconds <= 5.):
        raise ValueError('Coefficient0..0.01 and observable wait threshold1..5s required')
    specification = dict(schema='supported_wait_legal_prior_v1',
        coefficient=float(coefficient), minimum_wait_seconds=float(minimum_wait_seconds),
        minimum_stop_probability=.95, loss='KL(uniform_original_legal_actions || policy)',
        support='trainable_recovery_and_previous_executed_idle_with_choices',
        includes_stop=True, original_masks_retained=True,
        sampled_actions_and_likelihoods_unchanged=True,
        environment_steps_added=0, runtime_teacher=False, runtime_rule_added=False)
    model.long_wait_exploration_contract = specification
    return specification


def loss(model, observations, action_masks):
    specification = model.long_wait_exploration_contract
    if (observations.ndim != 2 or observations.shape[1] != 436
            or not torch.isfinite(observations).all()):
        raise ValueError('Finite original436D rollout observations required')
    raw_masks = torch.as_tensor(action_masks, device=observations.device)
    if (raw_masks.shape != (len(observations), 12)
            or not ((raw_masks == 0) | (raw_masks == 1)).all()):
        raise ValueError('Original twelve-choice boolean masks required')
    masks = raw_masks.bool()
    if (not masks[:, 0].all()
            or not torch.equal(masks_tensor(observations), masks)):
        raise ValueError('Stop and original observation masks must remain unchanged')
    with torch.no_grad():
        supported = model.policy.recovery_support(observations)
        timer = observations[:, 300+STAT_SLICE.start+3]*5.
        selected = (supported & masks[:, 1:].any(-1)
            & (timer >= specification['minimum_wait_seconds']-1e-6))
    if not selected.any():
        return torch.zeros((), device=observations.device), 0
    masks = masks[selected]
    distribution = model.policy.get_distribution(observations[selected], masks)
    confident = distribution.distribution.probs[:, 0].detach() >= specification['minimum_stop_probability']
    count = int(confident.sum())
    if not count:
        return torch.zeros((), device=observations.device), 0
    # Work with normalized logits instead of log(1-p_stop): the latter loses
    # precision when a float32 softmax already rounds p_stop to exactly one.
    log_probabilities = distribution.distribution.logits
    legal_count = masks.sum(-1).to(log_probabilities.dtype)
    values = (-log_probabilities.masked_fill(~masks, 0.).sum(-1)/legal_count
              -legal_count.log())
    value = values[confident].mean().clamp_min(0.)
    if not torch.isfinite(value):
        raise ValueError('Nonfinite legal-prior exploration loss')
    return value, count
