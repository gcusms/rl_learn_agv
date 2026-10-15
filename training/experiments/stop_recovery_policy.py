"""Learned recovery at an unchanged learned base's voluntary stops.

The residual can prefer any of the same twelve masked choices, including stop.
It is identically zero whenever the frozen base prefers a legal moving action,
at close approach, or when sensing/forecast inputs are unavailable. There is no
runtime teacher, timed maneuver, minimum velocity or removal of the stop action.
"""
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from stable_baselines3.common.utils import explained_variance

from experiments.conditional_recovery_ppo import actor_terms
from experiments.stop_feedback_ppo import PrefixLinear, StopFeedbackPPO
from neural_navigation.decision_feedback import TRAIN_DIM, STAT_SLICE, WAIT_SCHEMA


SCHEMA = 'stop_feedback_recovery_residual_v1'


def contract():
    return dict(schema=SCHEMA, feedback_schema=WAIT_SCHEMA, training_dimensions=TRAIN_DIM,
        action_count=12, residual_width=64, base_actor_frozen=True,
        support='masked_base_prefers_stop_and_motion_available_and_valid_inputs_and_goal_far',
        minimum_goal_distance_m=.3, stop_remains_selectable=True,
        actions_masks_guard_unchanged=True, no_runtime_teacher=True,
        no_elapsed_episode_time_input=True, simulator_truth_used=False)


def masks_tensor(observations):
    return torch.cat([torch.ones_like(observations[:, :1], dtype=torch.bool),
        observations[:, 235:244] > .5, observations[:, 267:269] > .5], dim=-1)


def usable(observations):
    stats = observations[:, 300+STAT_SLICE.start:300+STAT_SLICE.stop]
    return (masks_tensor(observations)[:, 1:].any(-1)
        & (stats[:, 4] > .5) & (stats[:, 7] > .5)
        & (torch.linalg.vector_norm(observations[:, 218:220]*12., dim=-1) > .3))


class RecoveryFeatures(torch.nn.Module):
    def __init__(self, base, width):
        super().__init__()
        self.base = base.requires_grad_(False)
        self.residual = torch.nn.Sequential(torch.nn.Linear(TRAIN_DIM, width), torch.nn.Tanh(),
            torch.nn.Linear(width, width), torch.nn.Tanh())

    def forward(self, observations):
        return torch.cat([self.base(observations[:, :300]), self.residual(observations),
            masks_tensor(observations).to(observations.dtype),
            usable(observations)[:, None].to(observations.dtype)], dim=-1)


class RecoveryHead(torch.nn.Module):
    def __init__(self, base, base_width, residual_width):
        super().__init__()
        self.base = base.requires_grad_(False)
        self.base_width = base_width
        self.residual = torch.nn.Linear(residual_width, 12)
        torch.nn.init.zeros_(self.residual.weight)
        torch.nn.init.zeros_(self.residual.bias)

    def parts(self, features):
        baseline = self.base(features[:, :self.base_width])
        masks = features[:, -13:-1] > .5
        base_choice = baseline.masked_fill(~masks, -1e8).argmax(-1)
        support = (base_choice == 0) & (features[:, -1] > .5)
        return baseline, support

    def forward(self, features):
        baseline, support = self.parts(features)
        residual = self.residual(features[:, self.base_width:-13])
        return baseline+support[:, None].to(features.dtype)*residual


class StopRecoveryPolicy(MaskableActorCriticPolicy):
    recovery_contract = staticmethod(contract)

    def _build(self, lr_schedule):
        super()._build(lr_schedule)
        if self.observation_space.shape != (TRAIN_DIM,) or self.action_space.n != 12:
            raise ValueError('Explicit 436-D, twelve-choice feedback contract required')
        if any(True for _ in self.features_extractor.parameters()):
            raise ValueError('A trainable shared extractor would change the frozen base')
        width = self.mlp_extractor.latent_dim_pi
        base = self.mlp_extractor.policy_net
        first = base[0]
        base[0] = torch.nn.Linear(300, first.out_features)
        self.mlp_extractor.policy_net = RecoveryFeatures(base, 64)
        self.action_net = RecoveryHead(self.action_net, width, 64)
        self.mlp_extractor.latent_dim_pi = width+64+13
        self.mlp_extractor.value_net[0] = PrefixLinear(self.mlp_extractor.value_net[0])
        self.optimizer = self.optimizer_class(
            [p for p in self.parameters() if p.requires_grad],
            lr=lr_schedule(1), **self.optimizer_kwargs)

    def recovery_support(self, observations):
        with torch.no_grad():
            features = self.mlp_extractor.policy_net(observations)
            return self.action_net.parts(features)[1]


class StopRecoveryPPO(StopFeedbackPPO):
    def train(self):
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)
        clip = self.clip_range(self._current_progress_remaining)
        logs = {k: [] for k in ['entropy_loss', 'policy_gradient_loss', 'value_loss',
            'approx_kl', 'clip_fraction', 'actor_recovery_fraction']}
        retention = getattr(self, 'demonstration_retention_contract', None)
        if retention is not None:
            logs.update(demonstration_retention_loss=[], demonstration_retention_samples=[])
        running = True
        loss = None
        for _ in range(self.n_epochs):
            for batch in self.rollout_buffer.get(self.batch_size):
                values, log_prob, entropy = self.policy.evaluate_actions(batch.observations,
                    batch.actions.long().flatten(), action_masks=batch.action_masks)
                pg, ent, kl, fraction, count = actor_terms(batch.observations, log_prob,
                    batch.old_log_prob, batch.advantages, entropy, clip, self.policy.recovery_support)
                vf = F.mse_loss(batch.returns, values.flatten())
                loss = pg+self.ent_coef*ent+self.vf_coef*vf
                for k, value in [('entropy_loss', ent), ('policy_gradient_loss', pg),
                        ('value_loss', vf), ('approx_kl', kl), ('clip_fraction', fraction)]:
                    logs[k].append(float(value.detach()))
                logs['actor_recovery_fraction'].append(count/len(batch.advantages))
                if count and self.target_kl is not None and kl > 1.5*self.target_kl:
                    running = False
                    break
                if retention is not None:
                    retained, replay_count = torch.zeros((), device=self.device), 0
                    if count:
                        from experiments.demonstration_retention import loss as retention_loss
                        retained, replay_count = retention_loss(self)
                        loss = loss+retention['coefficient']*retained
                    logs['demonstration_retention_loss'].append(float(retained.detach()))
                    logs['demonstration_retention_samples'].append(replay_count)
                self.policy.optimizer.zero_grad()
                loss.backward()
                if not count:
                    for name, parameter in self.policy.named_parameters():
                        if name.startswith(('mlp_extractor.policy_net.residual.', 'action_net.residual.')):
                            parameter.grad = None
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()
            self._n_updates += 1
            if not running:
                break
        for k, values in logs.items():
            self.logger.record('train/'+k, float(np.mean(values)) if values else 0.)
        self.logger.record('train/explained_variance', explained_variance(
            self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten()))
        self.logger.record('train/n_updates', self._n_updates, exclude='tensorboard')
        if loss is not None:
            self.logger.record('train/loss', float(loss.detach()))


def warm_start_recovery(checkpoint, env, seed=231015, policy_class=StopRecoveryPolicy):
    if not issubclass(policy_class, StopRecoveryPolicy):
        raise ValueError('An explicitly versioned recovery policy is required')
    old = MaskablePPO.load(checkpoint, device='cpu')
    if old.observation_space.shape != (300,) or old.action_space.n != 12:
        raise ValueError('Original native 300-D quality PPO required')
    if env.observation_space.shape != (TRAIN_DIM,) or env.feedback_schema != WAIT_SCHEMA:
        raise ValueError('Explicit V2 feedback training environment required')
    if env.reward_scale != float(old.training_reward_scale):
        raise ValueError('Inherited critic reward units must be preserved')
    model = StopRecoveryPPO(policy_class, env, seed=seed, device='cpu',
        n_steps=250, batch_size=125, n_epochs=4, learning_rate=3e-4,
        target_kl=.02, ent_coef=.01, gamma=old.gamma, gae_lambda=old.gae_lambda,
        vf_coef=old.vf_coef, max_grad_norm=old.max_grad_norm,
        policy_kwargs=copy.deepcopy(old.policy_kwargs))
    policy = model.policy
    policy.mlp_extractor.policy_net.base.load_state_dict(old.policy.mlp_extractor.policy_net.state_dict())
    policy.action_net.base.load_state_dict(old.policy.action_net.state_dict())
    source = old.policy.state_dict()
    state = policy.state_dict()
    for name in state:
        if not name.startswith(('mlp_extractor.value_net.', 'value_net.')):
            continue
        if name == 'mlp_extractor.value_net.0.weight':
            state[name].zero_()
            state[name][:, :300] = source[name]
        else:
            state[name].copy_(source[name])
    policy.load_state_dict(state)
    previous = dict(old.policy.named_parameters())
    for name, parameter in policy.named_parameters():
        if name not in previous or not name.startswith(('mlp_extractor.value_net.', 'value_net.')):
            continue
        history = old.policy.optimizer.state.get(previous[name])
        if not history:
            continue
        migrated = {}
        for key, value in history.items():
            if torch.is_tensor(value) and name == 'mlp_extractor.value_net.0.weight' and value.shape == previous[name].shape:
                expanded = torch.zeros_like(parameter)
                expanded[:, :300] = value
                migrated[key] = expanded
            else:
                migrated[key] = value.clone() if torch.is_tensor(value) else copy.deepcopy(value)
        policy.optimizer.state[parameter] = migrated
    model.feedback_schema = WAIT_SCHEMA
    model.exploration_schema = 'none'
    model.exploration_contract = None
    model.training_reward_scale = env.reward_scale
    model.recovery_contract = policy.recovery_contract()
    model.feedback_migration = dict(parent_checkpoint_timesteps=old.num_timesteps,
        dimensions=TRAIN_DIM, legacy_prefix_dimensions=300, feedback_schema=WAIT_SCHEMA,
        base_actor_frozen=True, all_actor_parameters_trainable=False,
        residual_head_initially_zero=True, recovery_contract=model.recovery_contract,
        critic_moments_retained=True, exact_rollout_resume=False,
        reward_unchanged=True, actions_masks_guard_unchanged=True)
    return model


def export_recovery(model, bundle, output):
    specification = model.policy.recovery_contract()
    if model.recovery_contract != specification:
        raise ValueError('Recovery export differs from the saved policy contract')
    from experiments.train_stop_feedback import export_candidate
    export_candidate(model, bundle, output)
    output = Path(output)
    p = output/'navigation_preservation.json'
    preservation = json.loads(p.read_text())
    preservation.update(navigation_actor_all_parameters_trainable=False,
        base_navigation_actor_frozen=True, recovery_contract=specification)
    p.write_text(json.dumps(preservation, indent=2)+'\n')
    p = output/'manifest.json'
    manifest = json.loads(p.read_text())
    manifest['navigation_preservation_contract'] = preservation
    manifest['files']['navigation_preservation.json'] = hashlib.sha256((output/'navigation_preservation.json').read_bytes()).hexdigest()
    p.write_text(json.dumps(manifest, indent=2)+'\n')
    p = output/'MODEL_CARD.json'
    card = json.loads(p.read_text())
    card.update(method='demonstration-initialized masked PPO recovery residual',
        frozen_components=['base_navigation_actor', 'localizer', 'router', 'directional_terminal_actor'],
        recovery_contract=specification, manifest_sha256=hashlib.sha256((output/'manifest.json').read_bytes()).hexdigest())
    if getattr(model, 'demonstration_retention_contract', None) is not None:
        card['training_demonstration_retention'] = model.demonstration_retention_contract
    p.write_text(json.dumps(card, indent=2)+'\n')
