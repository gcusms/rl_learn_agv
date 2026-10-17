"""Diagnose a completed failed trace without executing or teaching its actions."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from experiments.long_risk_actor import LongRiskActor, digest
from neural_navigation.experiment_budget import ExperimentBudget
from neural_navigation.masked_actions import ACTIONS


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'reports/stop_feedback_20261015'


def write(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def longest_run(values):
    best = current = 0
    for value in values:
        current = current + 1 if value else 0
        best = max(best, current)
    return best


def main(output):
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    budget = ExperimentBudget(.15, memory_mib=2048, directory=output)
    model_path = ROOT / 'models/stop_feedback_20261015/long_risk_bc1000_s231016'
    report_path = BASE / 'long_risk_validation_20261016/RESULTS.json'
    report = json.loads(report_path.read_text())
    assert report['completed']
    row = next(row for row in report['rows'] if row['case_id'] == 'early_yield_sensor_holdout_05_0')
    assert row['timeout'] and row['collisions'] == 0 and not row['safe_success']
    trace_path = ROOT / row['trace_path']
    assert digest(trace_path) == row['trace_sha256']
    paths = [Path(__file__), ROOT / 'training/experiments/long_risk_actor.py',
        ROOT / 'training/neural_navigation/experiment_budget.py', report_path, trace_path,
        model_path / 'checkpoint.pt', model_path / 'parent_actor.pt',
        model_path / 'demonstrations.npz', model_path / 'portable/actor.pt']
    inputs = {str(path.relative_to(ROOT)): digest(path) for path in paths}
    write(output / 'PLAN.json', dict(inputs=inputs, completed_trace_only=True,
        new_physical_steps=0, new_optimizer_updates=0, failed_trace_not_a_positive_source=True,
        no_decision_overrides=True, no_changes_to_safety_or_models=True))
    with np.load(trace_path, allow_pickle=False) as archive:
        data = {key: archive[key].copy() for key in archive.files}
    with np.load(model_path / 'demonstrations.npz', allow_pickle=False) as archive:
        sources = {key: archive[key].copy() for key in archive.files}
    state = torch.load(model_path / 'checkpoint.pt', map_location='cpu', weights_only=False)
    actor = LongRiskActor(torch.jit.load(str(model_path / 'parent_actor.pt'), map_location='cpu')).eval()
    actor.load_state_dict(state['actor'])
    portable = torch.jit.load(str(model_path / 'portable/actor.pt'), map_location='cpu').eval()
    x = torch.from_numpy(data['observations'])
    source_x = torch.from_numpy(sources['observations'])
    with torch.no_grad():
        features, parent = actor.features(x)
        source_features, _ = actor.features(source_x)
        masks = actor.masks(x)
        active = actor.support(x).numpy()
        logits = actor.logits(x).masked_fill(~masks, -1e8)
        stop_probability = logits.softmax(1)[:, 0].numpy()
        commands = actor(x).numpy()
        np.testing.assert_array_equal(commands, portable(x).numpy())
    np.testing.assert_array_equal(commands, ACTIONS[data['choices']])
    np.testing.assert_array_equal(masks.numpy(), data['masks'])
    parent_choice = np.abs(parent.numpy()[:, None, :] - ACTIONS[None, :, :]).max(2).argmin(1)
    observations = data['observations']
    history = observations[:, 439:].reshape(-1, 3, 14)
    history_valid = (history[:, :, 12] > .5).any(1)
    legal_count = data['masks'][:, 1:].sum(1)
    voluntary = (data['choices'] == 0) & (legal_count > 0)
    unsupported_wait = voluntary & ~history_valid
    exact_inputs = features.numpy()
    source_inputs = source_features.numpy()
    unique_inputs, counts = np.unique(exact_inputs[voluntary], axis=0, return_counts=True)
    most_repeated = unique_inputs[counts.argmax()]
    repeated_indices = np.flatnonzero(np.all(exact_inputs == most_repeated, axis=1))
    source_distances = np.abs(source_inputs - most_repeated).max(1)
    nearest = np.flatnonzero(source_distances == source_distances.min())
    records = []
    selected = sorted(set([0, *range(0, len(x), 50), len(x) - 1,
        int(np.flatnonzero(voluntary)[0]), int(repeated_indices[0]), int(repeated_indices[-1])]))
    for i in selected:
        scan = data['scans'][i]
        valid = np.isfinite(scan) & (scan >= .1) & (scan < 12.)
        records.append(dict(frame=i, sensor_time_s=float(data['labels'][i, 0]),
            recorded_choice=int(data['choices'][i]), parent_choice=int(parent_choice[i]),
            head_active=bool(active[i]), legal_motion_actions=int(legal_count[i]),
            risk_history_valid=history_valid[i].item(),
            previous_idle_features=observations[i, 433:435].tolist(),
            previous_stop_reason_index=int(observations[i, 399:419].argmax()),
            conditional_head_stop_probability=float(stop_probability[i]),
            waypoint_robot_m=(observations[i, 216:218] * 2).tolist(),
            goal_robot_m=(observations[i, 218:220] * 12).tolist(),
            measured_velocity=data['measured_velocity'][i].tolist(),
            finite_measured_returns=int(valid.sum()),
            minimum_measured_range_m=float(scan[valid].min()) if valid.any() else None))
    # These components already exist in the runtime prefix but the new head omits them.
    ignored = observations[repeated_indices, 144:224]
    span = np.ptp(ignored, axis=0)
    budget.tick()
    for name, expected in inputs.items():
        assert digest(ROOT / name) == expected, name
    result = dict(completed=True, reproduced_eager_portable_and_recorded_choices=len(x),
        voluntary_wait_frames=int(voluntary.sum()),
        longest_consecutive_voluntary_wait_frames=longest_run(voluntary),
        waits_without_supported_risk_history=int(unsupported_wait.sum()),
        waits_with_all_eleven_motion_actions_legal=int((voluntary & (legal_count == 11)).sum()),
        active_wait_frames=int((voluntary & active).sum()),
        parent_stop_on_voluntary_wait_frames=int((voluntary & (parent_choice == 0)).sum()),
        unique_exact_head_inputs_during_voluntary_wait=len(unique_inputs),
        most_repeated_head_input=dict(count=int(len(repeated_indices)),
            first_sensor_time_s=float(data['labels'][repeated_indices[0], 0]),
            last_sensor_time_s=float(data['labels'][repeated_indices[-1], 0]),
            feature_sha256=hashlib.sha256(most_repeated.tobytes()).hexdigest(),
            any_ignored_scan_waypoint_goal_velocity_features_differ=bool((span > 0).any()),
            maximum_normalized_ignored_feature_span=float(span.max()),
            nearest_safe_source_max_abs_feature_distance=float(source_distances.min()),
            nearest_safe_source_choices=sources['choices'][nearest].tolist(),
            stop_probability_range=[float(stop_probability[repeated_indices].min()),
                float(stop_probability[repeated_indices].max())]),
        records=records,
        interpretation=['Risk-head features omit current scan, waypoint/goal direction and measured motion.',
            'Waiting timers saturate; track absence is unknown and cannot itself justify movement.',
            'Legal actions alone do not establish a useful complete safe recovery.',
            'Recorded geometry summaries do not prove that the obstacle left or a route is safe.',
            'Add observable release context and complete-safe recovery examples before fitting again.'],
        new_physical_steps=0, new_optimizer_updates=0, new_ppo_steps=0,
        failed_trace_not_admitted_as_positive=True, resource=budget.statistics(),
        goal_achieved=False, deployment_approved=False)
    write(output / 'RESULTS.json', result)
    write(output / 'STATUS.json', dict(status='complete', diagnostic_only=True))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    main(parser.parse_args().output)
