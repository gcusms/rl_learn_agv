"""Read-only coverage audit; no policy prediction or counterfactual execution."""
import hashlib
import json
from pathlib import Path

import numpy as np

from neural_navigation.decision_feedback import FEEDBACK_REASONS
from neural_navigation.experiment_budget import ExperimentBudget


ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / 'reports/stop_feedback_20261015'
OUT = Path(__file__).parent


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def hist(values):
    keys, counts = np.unique(values, return_counts=True)
    return {str(k): int(c) for k, c in zip(keys, counts)}


def active(x, masks):
    history = x[:, 439:481].reshape(-1, 3, 14)
    previous = (history[:, :, 12] > .5).any(1) | (x[:, 433:435] > 0).any(1)
    context = previous | (x[:, 144:216].min(1) * 12 < .8)
    context |= ~masks[:, 1:].all(1) | (x[:, 419:431].max(1) > 0)
    return (context & (x[:, 435] > .5) & (x[:, 438] > .5)
        & masks[:, 1:].any(1) & (np.linalg.norm(x[:, 218:220] * 12, axis=1) > .3))


def main():
    budget = ExperimentBudget(.03, memory_mib=512, directory=OUT)
    source = BASE / 'context_feedback_sources_20261016/demonstrations.npz'
    meta = json.loads((source.parent / 'RESULTS.json').read_text())
    assert digest(source) == meta['dataset_sha256']
    with np.load(source, allow_pickle=False) as archive:
        data = {k: archive[k].copy() for k in archive.files}
    x, y, masks = [data[k] for k in ['observations', 'choices', 'masks']]
    enabled = active(x, masks)
    saturated = x[:, 434] >= 1. - 1e-6
    current = json.loads((BASE / 'context_broad_diagnostic_20261016/RESULTS.json').read_text())
    row = next(r for r in current['groups']['hard3']['rows'] if r['case_id'] == 'w11_training_stall_t106')
    path = ROOT / row['trace_path']; assert digest(path) == row['trace_sha256']
    with np.load(path, allow_pickle=False) as archive:
        trace = {k: archive[k].copy() for k in archive.files}
    selected = trace['model_controlled'].astype(bool)
    obs, legal, choices = [trace[k][selected] for k in ['observations', 'masks', 'choices']]
    signature = np.sum(legal.astype(np.int64) * (1 << np.arange(12)), axis=1)
    source_signature = np.sum(masks.astype(np.int64) * (1 << np.arange(12)), axis=1)
    sets = []
    for value in np.unique(signature):
        here = signature == value
        matched = enabled & (source_signature == value)
        sets.append(dict(legal_actions=np.flatnonzero(legal[here][0]).tolist(),
            physical_frames=int(here.sum()), source_labels=hist(y[matched]),
            source_saturated_motion_labels=hist(y[matched & saturated & (y != 0)])))
    reasons = obs[:, 399:419].argmax(1)
    result = dict(completed=True, source_frames=len(x), active_source_frames=int(enabled.sum()),
        source_sha256=digest(source), physical_trace_sha256=digest(path),
        actual106_frames=len(obs), actual106_head_active_frames=int(active(obs, legal).sum()),
        actual106_selected_actions=hist(choices), legal_action_sets=sets,
        actual106_previous_reasons=hist(np.asarray(FEEDBACK_REASONS)[reasons]),
        actual106_saturated_legal_wait_frames=int((obs[:, 434] >= 1. - 1e-6).sum()),
        actual106_nearest_scan_m=[float((obs[:, 144:216].min(1) * 12).min()),
            float((obs[:, 144:216].min(1) * 12).max())],
        active_source_motion_labels=hist(y[enabled & (y != 0)]),
        active_source_saturated_motion_labels=hist(y[enabled & saturated & (y != 0)]),
        all_inputs_are_recorded_sensor_observations=True, no_relabeling_or_policy_change=True,
        no_counterfactuals_or_training_or_new_physics=True,
        limitations=['Coverage is evidence of distribution mismatch, not proof of its only cause.',
            'A legal motion is not guaranteed to make progress or stay safe in future.'])
    budget.tick(); result['resource'] = budget.statistics()
    (OUT / 'RESULTS.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
