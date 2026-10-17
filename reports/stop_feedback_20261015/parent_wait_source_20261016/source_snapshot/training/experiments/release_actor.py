"""Learn all twelve actions using measured geometry and frozen RL navigation."""
import json
from pathlib import Path
import shutil

import numpy as np
import torch

from experiments.long_risk_actor import LongRiskActor, digest
from neural_navigation.release_observation import contract, RUNTIME_DIM


RUNTIME_NAME = 'neural_navigation.release_runtime.ReleaseRuntime'
FEATURE_DIM = 312


class ReleaseActor(LongRiskActor):
    def __init__(self, parent):
        super().__init__(parent)
        self.head = torch.nn.Sequential(torch.nn.Linear(FEATURE_DIM, 32), torch.nn.Tanh(),
            torch.nn.Linear(32, 32), torch.nn.Tanh(), torch.nn.Linear(32, 12))

    def initialize_risk_head(self, previous):
        # The new geometry channels initially have zero weight: exact old decisions.
        with torch.no_grad():
            self.head[0].weight.zero_()
            self.head[0].weight[:, :88].copy_(previous.head[0].weight)
            self.head[0].bias.copy_(previous.head[0].bias)
            self.head[2].load_state_dict(previous.head[2].state_dict())
            self.head[4].load_state_dict(previous.head[4].state_dict())

    def features(self, observations):
        old, baseline = super().features(observations[:, :481])
        return torch.cat([old, observations[:, 144:216], observations[:, 481:625],
            observations[:, 216:224]], dim=1), baseline

    def support(self, observations):
        return super().support(observations[:, :481])

    def logits(self, observations):
        features, _ = self.features(observations)
        return self.head(features) + 2. * features[:, 76:88]

    def forward(self, observations):
        features, baseline = self.features(observations)
        logits = self.head(features) + 2. * features[:, 76:88]
        choice = logits.masked_fill(~self.masks(observations), -1e8).argmax(1)
        return torch.where(self.support(observations)[:, None], self.actions[choice], baseline)


def export(actor, risk_bundle, output, checks):
    risk_bundle, output = Path(risk_bundle), Path(output)
    if output.exists():
        raise FileExistsError(output)
    from neural_navigation.goal_pose_runtime import check_files
    manifest = json.loads((risk_bundle / 'manifest.json').read_text())
    check_files(risk_bundle, manifest)
    shutil.copytree(risk_bundle, output, ignore=shutil.ignore_patterns('MODEL_CARD.json'))
    shutil.copy2(risk_bundle / 'actor.pt', output / 'previous_risk_actor.pt')
    tensor = torch.from_numpy(np.asarray(checks, np.float32))
    assert tensor.ndim == 2 and tensor.shape[1] == RUNTIME_DIM
    actor.eval()
    with torch.no_grad():
        saved = torch.jit.trace_module(actor, {'forward': (tensor,), 'logits': (tensor,),
            'support': (tensor,)})
        np.testing.assert_array_equal(saved(tensor).numpy(), actor(tensor).numpy())
    saved.save(str(output / 'actor.pt'))
    preservation = dict(schema='measured_release_head_on_frozen_rl_actor_v1',
        unchanged_runtime_prefix_dimensions=481, dimensions=RUNTIME_DIM,
        head_dimensions=[FEATURE_DIM, 32, 32, 12],
        head_features=['old_88_risk_mask_stop_feedback_parent_choice',
            'current_72_minimum_ranges', 'current_72_hit_and_72_no_return_fractions',
            'robot_relative_waypoint_and_goal', 'measured_velocity_and_previous_execution'],
        initialization='old risk weights retained; additional context weights initially zero',
        parent_actor_sha256=digest(output / 'parent_actor.pt'),
        previous_risk_actor_sha256=digest(output / 'previous_risk_actor.pt'),
        outside_support_parent_exact=True, near_goal_directional_parent_exact=True,
        stop_always_selectable=True, no_forced_motion=True, no_runtime_teacher=True,
        actions_masks_body_margin_blind_memory_and_deadlines_unchanged=True, new_ppo_steps=0)
    for name, value in [('release_observation.json', contract()), ('navigation_preservation.json', preservation)]:
        (output / name).write_text(json.dumps(value, indent=2) + '\n')
    manifest = dict(manifest, required_runtime=RUNTIME_NAME, observation_contract=contract(),
        navigation_preservation_contract=preservation, additional_ppo_steps=0, deployment_approved=False)
    manifest['files'] = dict(manifest['files'])
    for name in ['actor.pt', 'previous_risk_actor.pt', 'release_observation.json', 'navigation_preservation.json']:
        manifest['files'][name] = digest(output / name)
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    check_files(output, manifest)
    (output / 'MODEL_CARD.json').write_text(json.dumps(dict(deployment_approved=False,
        method='supervised release-context head on frozen RL navigation', new_ppo_steps=0,
        frozen_components=['parent_actor', 'localizer', 'router', 'directional_terminal_actor'],
        limitations=['Complete-safe development demonstrations do not prove independent navigation success.',
            'Unknown/no-return flags do not authorize motion; original guard remains unchanged.']), indent=2) + '\n')
