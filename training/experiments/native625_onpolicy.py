"""On-policy likelihoods for the existing 625D guard-reason navigation actor.

The actor's head, support and legal actions are unchanged. Frozen parent states
have a point-mass distribution; the critic can still learn from them. Sampling
uses a private generator so restoring physics cannot repeat the policy RNG.
"""
from dataclasses import dataclass
import copy

import numpy as np
import torch
import torch.nn.functional as F

from neural_navigation.masked_actions import ACTIONS, COMMANDS


def generalized_advantage(rewards, values, next_values, terminated, episode_ends,
        gamma=.99, gae_lambda=.95):
    """Use final-state values for truncation, without crossing a reset.

    A real finite task deadline is terminal. An arbitrary rollout cut may
    bootstrap, but its next_values must come from the final, pre-reset state.
    """
    tensors = [torch.as_tensor(x) for x in
        (rewards, values, next_values, terminated, episode_ends)]
    reward, value, following = [x.to(torch.float32) for x in tensors[:3]]
    terminal, ends = [x.to(torch.bool) for x in tensors[3:]]
    if not reward.ndim == 1 or any(x.shape != reward.shape for x in tensors):
        raise ValueError('One scalar per actual transition is required')
    if (terminal & ~ends).any():
        raise ValueError('A terminal transition must also end its episode')
    advantage = torch.zeros_like(reward)
    carry = torch.zeros((), dtype=value.dtype)
    for index in range(len(reward) - 1, -1, -1):
        delta = reward[index] + gamma * following[index] * (~terminal[index]) - value[index]
        carry = delta + gamma * gae_lambda * (~ends[index]) * carry
        advantage[index] = carry
    return advantage, advantage + value


@dataclass
class PolicySample:
    observation: np.ndarray
    mask: np.ndarray
    action: int
    old_log_probability: float
    old_value: float
    temperature: float
    executed_command: np.ndarray | None = None
    execution_changed: bool | None = None

    def record_execution(self, result):
        if self.executed_command is not None:
            raise ValueError('A sampled decision can only execute once')
        np.testing.assert_allclose(result['proposed_action'], ACTIONS[self.action], rtol=0, atol=2e-8)
        command = np.asarray(result['command'], np.float64)
        if command.shape != (2,) or not np.isfinite(command).all():
            raise ValueError('Finite actual execution is required')
        self.executed_command = command.copy()
        self.execution_changed = bool(np.max(abs(command - COMMANDS[self.action])) > 2e-8)
        # The sampled action and its old likelihood are deliberately untouched.


class Native625OnPolicy(torch.nn.Module):
    def __init__(self, actor, learning_rate=1e-4, temperature=2.):
        super().__init__()
        if not np.isfinite(temperature) or temperature <= 0:
            raise ValueError('A fixed positive sampling temperature is required')
        self.actor = actor.eval()
        self.temperature = float(temperature)
        self.reference_head = copy.deepcopy(actor.head).eval().requires_grad_(False)
        self.critic = torch.nn.Sequential(torch.nn.Linear(408, 32), torch.nn.Tanh(),
            torch.nn.Linear(32, 1))
        torch.nn.init.zeros_(self.critic[-1].weight)
        torch.nn.init.zeros_(self.critic[-1].bias)
        self.actor_optimizer = torch.optim.Adam(self.actor.head.parameters(), lr=learning_rate)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)

    def context(self, observations, masks=None):
        observations = torch.as_tensor(observations, dtype=torch.float32)
        if observations.ndim != 2 or observations.shape[1] != 625 or not torch.isfinite(observations).all():
            raise ValueError('The complete finite native625 observation is required')
        with torch.no_grad():
            features, baseline = self.actor.features(observations)
            legal = self.actor.masks(observations)
            supported = self.actor.support(observations)
            baseline_choice = (baseline[:, None] - self.actor.actions[None]).abs().amax(2).argmin(1)
        if masks is not None and not torch.equal(legal, torch.as_tensor(masks, dtype=torch.bool)):
            raise ValueError('Use the mask captured with this observation, not a later mask')
        if not legal[:, 0].all():
            raise ValueError('Stopping must remain selectable')
        if not legal[torch.arange(len(legal)), baseline_choice].all():
            raise ValueError('Frozen parent returned an inadmissible action')
        return features.detach(), legal, supported, baseline_choice

    def evaluate(self, observations, masks=None, actions=None):
        features, legal, support, baseline = self.context(observations, masks)
        logits = (self.actor.head(features) + 2. * features[:, 76:88]) / self.temperature
        frozen = torch.full_like(logits, -1e8)
        frozen.scatter_(1, baseline[:, None], 0.)
        effective = torch.where(support[:, None], logits.masked_fill(~legal, -1e8), frozen)
        distribution = torch.distributions.Categorical(logits=effective)
        values = self.critic(features).flatten()
        eligible = support & legal[:, 1:].any(1)
        result = dict(distribution=distribution, values=values, eligible=eligible,
            entropy=distribution.entropy(), features=features, masks=legal, support=support)
        if actions is not None:
            actions = torch.as_tensor(actions, dtype=torch.long)
            if actions.shape != (len(legal),) or not ((actions >= 0) & (actions < 12)).all():
                raise ValueError('Original sampled action indices are required')
            if not legal[torch.arange(len(legal)), actions].all():
                raise ValueError('Sampled action is invalid in its original mask')
            if ((~support) & (actions != baseline)).any():
                raise ValueError('Frozen-parent samples must retain the parent action')
            result['log_probability'] = distribution.log_prob(actions)
        return result

    @torch.no_grad()
    def sample(self, observation, generator):
        x = np.asarray(observation, np.float32)
        terms = self.evaluate(x[None])
        action = int(torch.multinomial(terms['distribution'].probs, 1, generator=generator)[0, 0])
        return PolicySample(observation=x.copy(), mask=terms['masks'][0].numpy().copy(),
            action=action, old_log_probability=float(terms['distribution'].log_prob(torch.tensor([action]))[0]),
            old_value=float(terms['values'][0]), temperature=self.temperature)

    def anchor_loss(self, observations):
        if observations is None:
            return next(self.actor.head.parameters()).sum() * 0.
        features, legal, support, _ = self.context(observations)
        if not support.any():
            return next(self.actor.head.parameters()).sum() * 0.
        features, legal = features[support], legal[support]
        with torch.no_grad():
            reference = (self.reference_head(features) + 2. * features[:, 76:88]) / self.temperature
            reference = F.log_softmax(reference.masked_fill(~legal, -1e8), dim=1)
        current = (self.actor.head(features) + 2. * features[:, 76:88]) / self.temperature
        current = F.log_softmax(current.masked_fill(~legal, -1e8), dim=1)
        return (reference.exp() * (reference - current)).sum(1).mean()

    def update(self, observations, masks, actions, old_log_probability, advantages, returns,
            *, sampled_temperature, anchors=None, clip=.2, entropy_coefficient=.01,
            anchor_strength=.05, target_kl=.02):
        if sampled_temperature != self.temperature:
            raise ValueError('Sampling and PPO likelihoods must use the same temperature')
        terms = self.evaluate(observations, masks, actions)
        old = torch.as_tensor(old_log_probability, dtype=torch.float32).detach()
        advantage = torch.as_tensor(advantages, dtype=torch.float32).detach()
        returns = torch.as_tensor(returns, dtype=torch.float32).detach()
        if any(x.shape != terms['values'].shape for x in (old, advantage, returns)):
            raise ValueError('One likelihood, advantage and return per sampled action required')
        keep = terms['eligible']
        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_updated = False
        kl = entropy = actor_loss = anchor = fraction = 0.
        if keep.any():
            selected = advantage[keep]
            if len(selected) > 1:
                selected = (selected - selected.mean()) / (selected.std(unbiased=False) + 1e-8)
            log_ratio = terms['log_probability'][keep] - old[keep]
            ratio = log_ratio.exp()
            estimate = ((ratio - 1.) - log_ratio).mean()
            kl = float(estimate.detach())
            fraction = float((abs(ratio.detach() - 1.) > clip).float().mean())
            entropy_term = terms['entropy'][keep].mean()
            pg = -torch.minimum(selected * ratio, selected * ratio.clamp(1.-clip, 1.+clip)).mean()
            anchoring = self.anchor_loss(anchors)
            loss = pg - entropy_coefficient * entropy_term + anchor_strength * anchoring
            entropy, actor_loss, anchor = [float(x.detach()) for x in (entropy_term, pg, anchoring)]
            if kl <= 1.5 * target_kl:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.head.parameters(), 1.)
                self.actor_optimizer.step()
                actor_updated = True
        value_loss = F.mse_loss(terms['values'], returns)
        self.critic_optimizer.zero_grad(set_to_none=True)
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.)
        self.critic_optimizer.step()
        return dict(actor_updated=actor_updated, eligible_samples=int(keep.sum()),
            total_samples=len(keep), approx_kl=kl, clip_fraction=fraction,
            entropy=entropy, actor_loss=actor_loss, anchor_kl=anchor,
            value_loss=float(value_loss.detach()),
            actor_optimizer_untouched_on_no_choice=not bool(keep.any()))
