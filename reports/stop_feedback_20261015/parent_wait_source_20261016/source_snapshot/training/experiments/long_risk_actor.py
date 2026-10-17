"""Learned twelve-choice risk head around a frozen portable RL navigation actor."""
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import torch
import torch.nn.functional as F

from neural_navigation.long_risk_observation import contract, RUNTIME_DIM
from neural_navigation.masked_actions import ACTIONS


RUNTIME_NAME = 'neural_navigation.long_risk_runtime.LongRiskRuntime'


class LongRiskActor(torch.nn.Module):
    def __init__(self, parent):
        super().__init__()
        self.parent = parent
        for parameter in self.parent.parameters():
            parameter.requires_grad_(False)
        self.register_buffer('actions', torch.from_numpy(ACTIONS.copy()))
        self.head = torch.nn.Sequential(torch.nn.Linear(88, 32), torch.nn.Tanh(),
            torch.nn.Linear(32, 32), torch.nn.Tanh(), torch.nn.Linear(32, 12))
        torch.nn.init.zeros_(self.head[-1].weight)
        torch.nn.init.zeros_(self.head[-1].bias)

    def masks(self, observations):
        return torch.cat([torch.ones_like(observations[:, :1], dtype=torch.bool),
            observations[:, 235:244] > .5, observations[:, 267:269] > .5], dim=1)

    def support(self, observations):
        history = observations[:, 439:].reshape(-1, 3, 14)
        recent_or_waiting = ((history[:, :, 12] > .5).any(dim=1)
            | (observations[:, 433:435] > 0.).any(dim=1))
        return (recent_or_waiting
            & (observations[:, 435] > .5) & (observations[:, 438] > .5)
            & self.masks(observations)[:, 1:].any(dim=1)
            & (torch.linalg.vector_norm(observations[:, 218:220] * 12., dim=1) > .3))

    def features(self, observations):
        baseline = self.parent(observations[:, :439])
        distance = (baseline[:, None, :] - self.actions[None, :, :]).abs().amax(dim=2)
        choice = distance.argmin(dim=1)
        one_hot = F.one_hot(choice, num_classes=12).to(observations.dtype)
        features = torch.cat([observations[:, 439:], self.masks(observations).to(observations.dtype),
            observations[:, 399:419], observations[:, 433:435], one_hot], dim=1)
        return features, baseline

    def logits(self, observations):
        features, _ = self.features(observations)
        return self.head(features) + 2. * features[:, -12:]

    def forward(self, observations):
        features, baseline = self.features(observations)
        logits = self.head(features) + 2. * features[:, -12:]
        selected = logits.masked_fill(~self.masks(observations), -1e8).argmax(dim=1)
        return torch.where(self.support(observations)[:, None], self.actions[selected], baseline)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def export(actor, parent_bundle, output, checks):
    parent_bundle, output = Path(parent_bundle), Path(output)
    if output.exists():
        raise FileExistsError(output)
    from neural_navigation.goal_pose_runtime import check_files
    parent_manifest = json.loads((parent_bundle / 'manifest.json').read_text())
    check_files(parent_bundle, parent_manifest)
    shutil.copytree(parent_bundle, output, ignore=shutil.ignore_patterns('MODEL_CARD.json'))
    shutil.copy2(parent_bundle / 'actor.pt', output / 'parent_actor.pt')
    actor.eval()
    tensor = torch.from_numpy(np.asarray(checks, np.float32))
    if tensor.ndim != 2 or tensor.shape[1] != RUNTIME_DIM:
        raise ValueError('Export checks require the complete runtime observation contract')
    with torch.no_grad():
        expected = actor(tensor)
        saved = torch.jit.trace_module(actor, {'forward': (tensor,), 'logits': (tensor,),
            'support': (tensor,)})
        np.testing.assert_array_equal(saved(tensor).numpy(), expected.numpy())
    saved.save(str(output / 'actor.pt'))
    preservation = dict(schema='frozen_rl_actor_with_learned_long_risk_head_v1',
        parent_actor_sha256=digest(output / 'parent_actor.pt'),
        unchanged_old_observation_prefix=439, new_observation_dimensions=RUNTIME_DIM,
        head_features=['three_frame_action_risk_history', 'action_mask',
            'previous_executed_stop_reason', 'previous_executed_idle_and_idle_with_choices',
            'frozen_parent_choice_one_hot'],
        trainable_head_dimensions=[88, 32, 32, 12], outside_support_parent_exact=True,
        near_goal_parent_directional_actor_exact=True, full_twelve_action_set=True,
        stop_always_selectable=True, no_runtime_teacher=True, no_forced_motion=True,
        actions_masks_body_guard_and_blind_memory_unchanged=True, new_ppo_steps=0)
    for name, value in (('long_risk_observation.json', contract()),
            ('navigation_preservation.json', preservation)):
        (output / name).write_text(json.dumps(value, indent=2) + '\n')
    manifest = dict(parent_manifest, required_runtime=RUNTIME_NAME,
        observation_contract=contract(), navigation_preservation_contract=preservation,
        parent_manifest_sha256=digest(parent_bundle / 'manifest.json'),
        additional_ppo_steps=0, deployment_approved=False)
    manifest['files'] = dict(parent_manifest['files'])
    for name in ('actor.pt', 'parent_actor.pt', 'long_risk_observation.json', 'navigation_preservation.json'):
        manifest['files'][name] = digest(output / name)
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    check_files(output, manifest)
    (output / 'MODEL_CARD.json').write_text(json.dumps(dict(deployment_approved=False,
        method='supervised risk head on a frozen RL navigation actor', new_ppo_steps=0,
        frozen_components=['parent_navigation_actor', 'localizer', 'router', 'directional_terminal_actor'],
        manifest_sha256=digest(output / 'manifest.json'),
        limitations=['Development-source fitting and export parity do not establish navigation safety.',
            'Constant track velocity is an approximation; independent closed-loop gates are required.']), indent=2) + '\n')
