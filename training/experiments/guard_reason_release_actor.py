"""Expose current per-action guard explanations to the learned recovery head.

The 625D sensor contract and hard permissions are unchanged. The old release
head consumed the previous execution reason but discarded current rejection
groups. Additional weights start at zero for an exact migration of that head.
"""
import json
from pathlib import Path

import torch

from experiments.context_release_actor import ContextReleaseActor, export as export_context
from experiments.long_risk_actor import digest
from neural_navigation.decision_feedback import GROUPS


FEATURE_DIM = 408


class GuardReasonProjection(torch.nn.Module):
    def __init__(self, previous):
        super().__init__()
        self.previous = previous
        self.current_reason = torch.nn.Linear(96, 32, bias=False)
        torch.nn.init.zeros_(self.current_reason.weight)

    def forward(self, features):
        # Keep the old GEMM shape: padding its matrix with zero columns can
        # change floating-point accumulation and very close action ties.
        return self.previous(features[:, :312]) + self.current_reason(features[:, 312:])


def contract():
    return dict(schema='current_guard_reason_release_head_v1',
        observation_dimensions=625, old_head_dimensions=312,
        head_dimensions=[FEATURE_DIM, 32, 32, 12],
        additional_features='current_12_by_8_guard_rejection_groups',
        source_slice=[303, 399], rejection_groups=list(GROUPS),
        first_projection='unchanged_312_projection_plus_zero_initialized_96_projection',
        same_current_scan_as_action_mask=True, simulator_truth_used=False,
        previous_executed_reason_retained=True, runtime_teacher=False,
        action_permissions_and_learning_support_unchanged=True,
        no_forced_motion=True, stop_always_selectable=True)


class GuardReasonReleaseActor(ContextReleaseActor):
    def __init__(self, parent):
        super().__init__(parent)
        self.head[0] = GuardReasonProjection(self.head[0])

    def initialize_risk_head(self, previous):
        with torch.no_grad():
            self.head[0].previous.weight.zero_()
            self.head[0].previous.weight[:, :88].copy_(previous.head[0].weight)
            self.head[0].previous.bias.copy_(previous.head[0].bias)
            self.head[0].current_reason.weight.zero_()
            self.head[2].load_state_dict(previous.head[2].state_dict())
            self.head[4].load_state_dict(previous.head[4].state_dict())

    def initialize_release_head(self, previous):
        """Migrate a 312D release head without introducing a behavior change."""
        with torch.no_grad():
            self.head[0].previous.load_state_dict(previous.head[0].state_dict())
            self.head[0].current_reason.weight.zero_()
            self.head[2].load_state_dict(previous.head[2].state_dict())
            self.head[4].load_state_dict(previous.head[4].state_dict())

    def features(self, observations):
        previous, baseline = super().features(observations)
        return torch.cat([previous, observations[:, 303:399]], dim=1), baseline


def export(actor, risk_bundle, output, checks):
    export_context(actor, risk_bundle, output, checks)
    output = Path(output)
    preservation = json.loads((output / 'navigation_preservation.json').read_text())
    preservation.update(schema='release_head_with_current_guard_reasons_v1',
        head_dimensions=[FEATURE_DIM, 32, 32, 12], guard_reason_features=contract())
    preservation['head_features'].append('current_12_by_8_guard_rejection_groups')
    (output / 'navigation_preservation.json').write_text(json.dumps(preservation, indent=2) + '\n')
    manifest = json.loads((output / 'manifest.json').read_text())
    manifest['navigation_preservation_contract'] = preservation
    manifest['files']['navigation_preservation.json'] = digest(output / 'navigation_preservation.json')
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    card = json.loads((output / 'MODEL_CARD.json').read_text())
    card.update(guard_reason_features=contract(), manifest_sha256=digest(output / 'manifest.json'))
    (output / 'MODEL_CARD.json').write_text(json.dumps(card, indent=2) + '\n')
    from neural_navigation.goal_pose_runtime import check_files
    check_files(output, manifest)
