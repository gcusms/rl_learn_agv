"""Keep the learned release head available during a sensor-observed detour.

This changes learning support, never motion permission. It uses the same near
obstacle and restricted-action contexts as the earlier proactive recovery
policy. Invalid sensing, stop-only states and terminal docking keep the frozen
parent. There is no episode clock, map coordinate or forced-motion exception.
"""
import json
from pathlib import Path

import torch

from experiments.release_actor import ReleaseActor, export as export_release
from experiments.long_risk_actor import digest


SCHEMA = 'sensor_context_release_support_v1'


def contract():
    return dict(schema=SCHEMA, observation_dimensions=625,
        nearest_scan_context_m=.8,
        support='valid_goal_far_motion_available_and_old_context_or_near_scan_or_restricted_mask_or_guard_forecast',
        no_episode_clock_or_case_or_map_coordinate_gate=True,
        unchanged_action_permissions=True, stop_always_selectable=True,
        invalid_stop_only_and_terminal_parent_exact=True)


class ContextReleaseActor(ReleaseActor):
    def support(self, observations):
        masks = self.masks(observations)
        near = observations[:, 144:216].amin(1) * 12. < .8
        restricted = ~masks[:, 1:].all(1)
        forecast = observations[:, 419:431].amax(1) > 0.
        valid = ((observations[:, 435] > .5) & (observations[:, 438] > .5)
            & masks[:, 1:].any(1)
            & (torch.linalg.vector_norm(observations[:, 218:220] * 12., dim=1) > .3))
        return super().support(observations) | (valid & (near | restricted | forecast))


def export(actor, risk_bundle, output, checks):
    export_release(actor, risk_bundle, output, checks)
    output = Path(output)
    preservation = json.loads((output / 'navigation_preservation.json').read_text())
    preservation.update(schema='measured_release_with_sensor_context_support_v1',
        initialization='old risk weights; decisions may differ in newly trainable sensor contexts',
        learning_support=contract(), outside_support_parent_exact=True)
    (output / 'navigation_preservation.json').write_text(json.dumps(preservation, indent=2) + '\n')
    manifest = json.loads((output / 'manifest.json').read_text())
    manifest['navigation_preservation_contract'] = preservation
    manifest['files']['navigation_preservation.json'] = digest(output / 'navigation_preservation.json')
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    card = json.loads((output / 'MODEL_CARD.json').read_text())
    card.update(learning_support=contract(), manifest_sha256=digest(output / 'manifest.json'))
    (output / 'MODEL_CARD.json').write_text(json.dumps(card, indent=2) + '\n')
    from neural_navigation.goal_pose_runtime import check_files
    check_files(output, manifest)
