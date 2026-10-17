"""Causal, bounded exploration for an overconfident masked navigation actor.

This changes the stochastic policy used by PPO, not the admissible commands.
Sampling and likelihood evaluation use the same temperature. Positive scaling
preserves the deterministic action order, including the selectable stop action.
No simulator labels or elapsed-episode intervention times enter the policy.
"""
import torch

from experiments.stop_feedback_ppo import StopFeedbackPolicy, decision_support
from neural_navigation.decision_feedback import RISK_SLICE, STAT_SLICE


SCHEMA = 'sensor_temperature_v1'


def contract():
    return dict(schema=SCHEMA, maximum_temperature=3., minimum_temperature=1.,
        idle_ramp_seconds=[1., 3.], goal_exclusion_radius_m=.3,
        sources=['current_guard_forecast', 'supported_moving_tracks',
                 'causal_executed_idle_with_choices_seconds'],
        require_scan_and_prediction_valid=True, require_motion_choice=True,
        sampling_and_ppo_likelihood_identical=True,
        deterministic_argmax_preserved=True, stop_remains_allowed=True,
        nominal_reference_anchor_preserved=True, simulator_truth_used=False)


def temperature(observations):
    stats = observations[:, 300+STAT_SLICE.start:300+STAT_SLICE.stop]
    risk = observations[:, 300+RISK_SLICE.start:300+RISK_SLICE.stop]
    waited = ((stats[:, 3]*5.-1.)/2.).clamp(0., 1.)
    moving = (stats[:, 5]*4.).clamp(0., 1.)
    forecast = (risk.max(-1).values > 0.).to(observations.dtype)
    heat = torch.maximum(waited, torch.maximum(moving, forecast))
    far = torch.linalg.vector_norm(observations[:, 218:220]*12., dim=-1) > .3
    usable = decision_support(observations) & (stats[:, 4] > .5) & (stats[:, 7] > .5) & far
    return 1.+2.*heat*usable.to(observations.dtype)


class SensorExplorationPolicy(StopFeedbackPolicy):
    """Maskable SB3 policy with one common distribution for every PPO path."""

    def _latents(self, observations):
        features = self.extract_features(observations)
        if self.share_features_extractor:
            return self.mlp_extractor(features)
        actor, critic = features
        return (self.mlp_extractor.forward_actor(actor),
                self.mlp_extractor.forward_critic(critic))

    def _distribution(self, observations, actor, action_masks, tempered=True):
        logits = self.action_net(actor)
        if tempered:
            logits = logits/temperature(observations)[:, None]
        distribution = self.action_dist.proba_distribution(action_logits=logits)
        if action_masks is not None:
            distribution.apply_masking(action_masks)
        return distribution

    def forward(self, obs, deterministic=False, action_masks=None):
        actor, critic = self._latents(obs)
        distribution = self._distribution(obs, actor, action_masks)
        actions = distribution.get_actions(deterministic=deterministic)
        return actions, self.value_net(critic), distribution.log_prob(actions)

    def evaluate_actions(self, obs, actions, action_masks=None):
        actor, critic = self._latents(obs)
        distribution = self._distribution(obs, actor, action_masks)
        return self.value_net(critic), distribution.log_prob(actions), distribution.entropy()

    def get_distribution(self, obs, action_masks=None):
        actor, _ = self._latents(obs)
        return self._distribution(obs, actor, action_masks)

    def get_nominal_distribution(self, obs, action_masks=None):
        # Retention constrains the same untempered learned command preference
        # as before, without penalizing the explicitly declared exploration.
        actor, _ = self._latents(obs)
        return self._distribution(obs, actor, action_masks, tempered=False)
