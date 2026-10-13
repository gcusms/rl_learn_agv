"""Keep recovery logits available during sensor-observed reorientation.

This changes learning support, not the command: the masked learned actor still
chooses each action. No duration, direction, target frame or maneuver is fixed.
Untrained prototype; not imported by the currently running matched trials.
"""
import copy
import numpy as np
import torch
from experiments.focused_recovery_policy import FocusedRecoveryFeatures, recovery_support
from experiments.soft_prior_recovery_policy import SoftPriorRecoveryPolicy, warm_start_soft_prior
from experiments.focused_recovery_policy import FocusedRecoveryPPO
from sb3_contrib import MaskablePPO


def continued_support(observations, alignment_cos=.7):
    alternatives = ((observations[:,235:244] > .5).any(-1)
                    | (observations[:,267:269] > .5).any(-1))
    trusted = ((observations[:,294] > .5) & (observations[:,229] >= .5)
               & (torch.linalg.vector_norm(observations[:,216:218], dim=-1) > .025))
    # Actual previous command, not a proposed maneuver or simulator motion.
    pure_turn = ((observations[:,247].abs() < .025)
                 & (observations[:,248].abs() > .5))
    continuing = trusted & pure_turn & (observations[:,232] < alignment_cos)
    return (observations[:,264] < .5) & (recovery_support(observations)
                                       | (alternatives & continuing))


class ContinuedRecoveryFeatures(FocusedRecoveryFeatures):
    def __init__(self, source, alignment_cos):
        super().__init__(source)
        self.alignment_cos = float(alignment_cos)

    def forward(self, observations):
        return torch.cat([self.base(observations), self.residual(observations),
                          continued_support(observations, self.alignment_cos)[:,None]
                          .to(observations.dtype)], dim=-1)


class ContinuedRecoveryPolicy(SoftPriorRecoveryPolicy):
    def __init__(self, *args, continuation_alignment_cos=.7, **kwargs):
        self.continuation_alignment_cos = float(continuation_alignment_cos)
        if not np.isfinite(self.continuation_alignment_cos) or not 0 < self.continuation_alignment_cos < 1:
            raise ValueError('Continuation alignment cosine must be finite and in (0,1)')
        super().__init__(*args, **kwargs)

    def _build(self, lr_schedule):
        super()._build(lr_schedule)
        # Reuse parameter objects registered in the existing Adam groups.
        self.mlp_extractor.policy_net = ContinuedRecoveryFeatures(
            self.mlp_extractor.policy_net, self.continuation_alignment_cos)

    def _get_constructor_parameters(self):
        return dict(super()._get_constructor_parameters(),
                    continuation_alignment_cos=self.continuation_alignment_cos)


def warm_start_continued(checkpoint, env, seed=92611, prior_scale=None, alignment_cos=.7):
    source = MaskablePPO.load(checkpoint, device='cpu')
    retained_recovery = isinstance(source.policy, SoftPriorRecoveryPolicy)
    if retained_recovery:
        model = source
        model.set_env(env)
        model.set_random_seed(seed)
        model.seed = seed
        model.__class__ = FocusedRecoveryPPO
        if prior_scale is None:
            prior_scale = model.policy.recovery_prior_scale
        scales = env.get_attr('reward_scale') if hasattr(env, 'get_attr') else [env.reward_scale]
        if any(scale != model.training_reward_scale for scale in scales):
            raise ValueError('Reward units differ')
    else:
        if hasattr(source.policy.mlp_extractor.policy_net, 'residual'):
            raise ValueError('Unsupported trained recovery policy; do not discard its learned branch')
        del source
        if prior_scale is None:
            prior_scale = 1/6
        model = warm_start_soft_prior(checkpoint, env, seed, prior_scale)
    previous = model.policy
    options = copy.deepcopy(model.policy_kwargs)
    options['continuation_alignment_cos'] = alignment_cos
    options['recovery_prior_scale'] = prior_scale
    rng = torch.get_rng_state()
    try:
        policy = ContinuedRecoveryPolicy(model.observation_space, model.action_space,
                                         model.lr_schedule, **options).to(model.device)
        policy.load_state_dict(previous.state_dict())
        policy.optimizer.load_state_dict(copy.deepcopy(previous.optimizer.state_dict()))
    finally:
        torch.set_rng_state(rng)
    model.policy = policy
    model.policy_class = ContinuedRecoveryPolicy
    model.policy_kwargs = options
    model.gated_migration.update(continuation_alignment_cos=alignment_cos,
        continuation_uses_previous_executed_pure_turn=True, fixed_maneuver=False,
        existing_recovery_weights_and_adam_retained=retained_recovery,
        exact_old_training_rng_resume=False)
    return model
