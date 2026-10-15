"""PPO with a training-only reference-mean cost in clear navigation states.

The reference never executes actions. Blockage/recovery states are exempt.
This regularizer is not a safety certificate or a hard policy trust region.
"""
import copy
import numpy as np
import torch as th
import torch.nn.functional as F
from stable_baselines3 import PPO
from stable_baselines3.common.utils import explained_variance
from neural_navigation.safety_context import BASE_DIM, OBSERVATION_DIM, PREDICTIVE_DIM, REASON_START, REASONS


class PriorAnchoredPPO(PPO):
    def __init__(self, *args, anchor_strength=0., anchor_scale=.3, **kwargs):
        if not np.isfinite([anchor_strength, anchor_scale]).all() or anchor_strength < 0 or anchor_scale <= 0:
            raise ValueError('Finite nonnegative strength and positive scale required')
        self.anchor_strength = float(anchor_strength)
        self.anchor_scale = float(anchor_scale)
        self.reference_policy = None
        self.last_anchor_statistics = {}
        super().__init__(*args, **kwargs)

    def _excluded_save_params(self):
        return super()._excluded_save_params()+['reference_policy']

    def snapshot_reference(self):
        if self.observation_space.shape not in ((OBSERVATION_DIM,), (PREDICTIVE_DIM,)) or self.action_space.shape != (2,):
            raise ValueError('Versioned continuous navigation policy required')
        self.reference_policy = copy.deepcopy(self.policy)
        self.reference_policy.set_training_mode(False)
        self.reference_policy.requires_grad_(False)

    def reference_loss(self, observations):
        if self.reference_policy is None:
            raise ValueError('Freeze a reference policy before anchored learning')
        with th.no_grad():
            reference = self.reference_policy.get_distribution(observations).distribution.mean
            # All quantities are existing sensor-derived policy inputs. Do
            # not imitate a rejected reverse proposal or immobilized state.
            mask = ((observations[:, BASE_DIM:BASE_DIM+9] > .5).all(dim=1)
                    & (observations[:, BASE_DIM+REASON_START+REASONS.index('none')] > .5)
                    & (observations[:, OBSERVATION_DIM-1] < .04)
                    & (observations[:, 234] > .4)  # forward corridor >0.8m
                    & (observations[:, 232] > .2)  # waypoint mostly ahead
                    & (reference[:, 0] >= 0.)
                    & ((reference[:, 0] > .125) | (reference[:, 1].abs() > .2)))
            if observations.shape[1] == PREDICTIVE_DIM:
                # Do not preserve the teacher's behavior when supported scan
                # tracks predict that remaining here would be dangerous.
                mask &= ~((observations[:, -1] > .5) & (observations[:, OBSERVATION_DIM+2] > 0.))
        current = self.policy.get_distribution(observations).distribution.mean
        per_state = .5*((current-reference)/self.anchor_scale).square().mean(dim=1)
        return (per_state*mask.to(per_state.dtype)).mean(), mask.float().mean()

    def train(self):
        if self.anchor_strength == 0.:
            # Use the installed SB3 path exactly for the matched control.
            return super().train()
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)
        clip_range = self.clip_range(self._current_progress_remaining)
        clip_range_vf = self.clip_range_vf(self._current_progress_remaining) if self.clip_range_vf is not None else None
        entropy_losses, pg_losses, value_losses, clip_fractions = [], [], [], []
        anchor_losses, anchor_fractions = [], []
        continue_training = True
        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            for rollout in self.rollout_buffer.get(self.batch_size):
                values, log_prob, entropy = self.policy.evaluate_actions(rollout.observations, rollout.actions)
                values = values.flatten()
                advantages = rollout.advantages
                if self.normalize_advantage and len(advantages) > 1:
                    advantages = (advantages-advantages.mean())/(advantages.std()+1e-8)
                ratio = th.exp(log_prob-rollout.old_log_prob)
                policy_loss = -th.min(advantages*ratio,
                    advantages*th.clamp(ratio, 1-clip_range, 1+clip_range)).mean()
                pg_losses.append(policy_loss.item())
                clip_fractions.append(th.mean((th.abs(ratio-1) > clip_range).float()).item())
                values_pred = values if clip_range_vf is None else rollout.old_values+th.clamp(
                    values-rollout.old_values, -clip_range_vf, clip_range_vf)
                value_loss = F.mse_loss(rollout.returns, values_pred)
                value_losses.append(value_loss.item())
                entropy_loss = -th.mean(-log_prob) if entropy is None else -th.mean(entropy)
                entropy_losses.append(entropy_loss.item())
                anchor_loss, anchor_fraction = self.reference_loss(rollout.observations)
                anchor_losses.append(anchor_loss.item())
                anchor_fractions.append(anchor_fraction.item())
                loss = policy_loss+self.ent_coef*entropy_loss+self.vf_coef*value_loss+self.anchor_strength*anchor_loss
                with th.no_grad():
                    log_ratio = log_prob-rollout.old_log_prob
                    approx_kl = th.mean((th.exp(log_ratio)-1)-log_ratio).cpu().numpy()
                    approx_kl_divs.append(approx_kl)
                if self.target_kl is not None and approx_kl > 1.5*self.target_kl:
                    continue_training = False
                    break
                self.policy.optimizer.zero_grad()
                loss.backward()
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()
            self._n_updates += 1
            if not continue_training:
                break
        explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())
        for name, data in [('entropy_loss', entropy_losses), ('policy_gradient_loss', pg_losses),
                           ('value_loss', value_losses), ('approx_kl', approx_kl_divs),
                           ('clip_fraction', clip_fractions), ('reference_mean_loss', anchor_losses),
                           ('reference_state_fraction', anchor_fractions)]:
            self.logger.record('train/'+name, np.mean(data))
        self.logger.record('train/loss', loss.item())
        self.logger.record('train/explained_variance', explained_var)
        self.logger.record('train/std', th.exp(self.policy.log_std).mean().item())
        self.logger.record('train/n_updates', self._n_updates, exclude='tensorboard')
        self.logger.record('train/clip_range', clip_range)
        if clip_range_vf is not None:
            self.logger.record('train/clip_range_vf', clip_range_vf)
        self.last_anchor_statistics = dict(reference_mean_loss=float(np.mean(anchor_losses)),
            eligible_state_fraction=float(np.mean(anchor_fractions)), updates=self._n_updates)
