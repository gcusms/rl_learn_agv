"""Audit the current release actor's recorded wait failure without execution."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from experiments.release_actor import ReleaseActor
from neural_navigation.decision_feedback import FEEDBACK_REASONS
from neural_navigation.experiment_budget import ExperimentBudget
from neural_navigation.masked_actions import ACTIONS, COMMANDS


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'reports/stop_feedback_20261015'
MODEL = ROOT / 'models/stop_feedback_20261015/release_context_bc2000_s231016'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_arrays(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key].copy() for key in archive.files}


def stats(values):
    values = np.asarray(values)
    return {name: float(np.percentile(values, q)) for name, q in [('min', 0), ('p50', 50), ('p95', 95), ('max', 100)]} if len(values) else {}


def longest_run(values):
    best = run = 0
    for value in values:
        run = run + 1 if value else 0
        best = max(best, run)
    return best


def choose(commands):
    return np.abs(commands[:, None] - ACTIONS[None]).max(2).argmin(1)


def main(args):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    budget = ExperimentBudget(args.cpu_budget, memory_mib=1536, directory=output)
    report_path = BASE / 'release_validation_20261016/RESULTS.json'
    validation = json.loads(report_path.read_text())
    assert validation['completed']
    row = next(r for r in validation['rows'] if r['case_id'] == args.case_id)
    assert row['timeout'] and not row['safe_success'] and row['collisions'] == 0
    trace = ROOT / row['trace_path']
    assert digest(trace) == row['trace_sha256']
    inputs = [Path(__file__), report_path, trace, MODEL / 'checkpoint.pt',
        MODEL / 'parent_actor.pt', MODEL / 'demonstrations.npz', MODEL / 'portable/actor.pt',
        ROOT / 'training/experiments/release_actor.py', ROOT / 'training/experiments/long_risk_actor.py',
        ROOT / 'training/neural_navigation/decision_feedback.py']
    hashes = {str(p.relative_to(ROOT)): digest(p) for p in inputs}
    (output / 'PLAN.json').write_text(json.dumps(dict(case_id=args.case_id, inputs=hashes,
        new_training_steps=0, new_physical_steps=0, diagnostic_only=True,
        failed_trace_never_used_as_a_positive_label=True), indent=2) + '\n')
    data = load_arrays(trace)
    source = load_arrays(MODEL / 'demonstrations.npz')
    actor = ReleaseActor(torch.jit.load(str(MODEL / 'parent_actor.pt'), map_location='cpu')).eval()
    actor.load_state_dict(torch.load(MODEL / 'checkpoint.pt', map_location='cpu', weights_only=False)['actor'])
    portable = torch.jit.load(str(MODEL / 'portable/actor.pt'), map_location='cpu').eval()
    x, sx = torch.from_numpy(data['observations']), torch.from_numpy(source['observations'])
    with torch.no_grad():
        features, parent = actor.features(x)
        source_features, _ = actor.features(sx)
        active, source_active = actor.support(x).numpy(), actor.support(sx).numpy()
        masks = actor.masks(x)
        logits = actor.logits(x).masked_fill(~masks, -1e8)
        pstop = logits.softmax(1)[:, 0].numpy()
        commands = actor(x).numpy()
        np.testing.assert_array_equal(commands, portable(x).numpy())
        np.testing.assert_array_equal(commands, ACTIONS[data['choices']])
        np.testing.assert_array_equal(actor(sx).numpy(), ACTIONS[source['choices']])
    np.testing.assert_array_equal(masks.numpy(), data['masks'])
    np.testing.assert_allclose(data['commands'], COMMANDS[data['choices']], atol=2e-8, rtol=0)
    budget.tick()
    legal = data['masks'][:, 1:].sum(1)
    wait = (data['choices'] == 0) & (legal > 0)
    parent_choices = choose(parent.numpy())
    observations = data['observations']
    reason = observations[:, 399:419].argmax(1)
    history_valid = (observations[:, 439:481].reshape(-1, 3, 14)[:, :, 12] > .5).any(1)
    source_stop = source['choices'] == 0
    source_wait = source_stop & source['masks'][:, 1:].any(1)
    source_groups = []
    for sid in np.unique(source['source_ids']):
        selected = source['source_ids'] == sid
        source_groups.append(dict(source_id=int(sid), frames=int(selected.sum()),
            longest_voluntary_wait_frames=longest_run(source_wait[selected]),
            active_stop_labels=int((selected & source_active & source_stop).sum()),
            active_motion_labels=int((selected & source_active & ~source_stop).sum())))
    ablations = {}
    for name, removed in [('without_parent_bonus', None), ('without_wait_timers', slice(74, 76)),
            ('without_previous_stop_reason', slice(54, 74)), ('without_ranges', slice(88, 160)),
            ('without_scan_quality', slice(160, 304)), ('without_waypoint_goal', slice(304, 308)),
            ('without_measured_motion_and_execution', slice(308, 312))]:
        changed = features.clone()
        if removed is not None:
            changed[:, removed] = 0.
        with torch.no_grad():
            values = actor.head(changed)
            if name != 'without_parent_bonus':
                values = values + 2. * changed[:, 76:88]
            values = values.masked_fill(~masks, -1e8)
            action = torch.where(torch.from_numpy(active)[:, None], actor.actions[values.argmax(1)], parent).numpy()
            counterfactual_stop_probability = values.softmax(1)[:, 0].numpy()
        choices = choose(action)
        ablations[name] = dict(changed_choices=int((choices != data['choices']).sum()),
            recorded_wait_frames_changing_to_motion=int((wait & (choices != 0)).sum()),
            pstop_on_original_wait=stats(counterfactual_stop_probability[wait]))
        budget.tick()
    selected_frames = sorted(set([int(np.flatnonzero(wait)[0]), *range(0, len(x), 75), len(x) - 1]))
    neighbors = []
    train_features = source_features.numpy()
    train_active_indices = np.flatnonzero(source_active)
    for i in selected_frames:
        distance = np.sqrt(np.square(train_features[train_active_indices] - features[i].numpy()).mean(1))
        nearest_indices = train_active_indices[np.argsort(distance)[:3]]
        nearest_distances = np.sort(distance)[:3]
        neighbors.append(dict(frame=i, sensor_time_s=float(data['labels'][i, 0]),
            choice=int(data['choices'][i]), parent_choice=int(parent_choices[i]),
            pstop=float(pstop[i]), active=bool(active[i]), legal_motion_actions=int(legal[i]),
            previous_reason=FEEDBACK_REASONS[int(reason[i])],
            idle_seconds_clipped=(observations[i, 433:435] * 5).tolist(),
            risk_history_valid=bool(history_valid[i]),
            goal_robot_m=(observations[i, 218:220] * 12).tolist(),
            waypoint_robot_m=(observations[i, 216:218] * 2).tolist(),
            nearest_active_source_indices=nearest_indices.tolist(),
            nearest_active_source_ids=source['source_ids'][nearest_indices].tolist(),
            nearest_active_source_choices=source['choices'][nearest_indices].tolist(),
            nearest_feature_rms_distances=nearest_distances.tolist()))
        budget.tick()
    histogram = lambda a: {str(int(k)): int(v) for k, v in zip(*np.unique(a, return_counts=True))}
    result = dict(completed=True, at=datetime.now(timezone.utc).isoformat(), case_id=args.case_id,
        matched_eager_portable_recorded_frames=len(x), matched_training_source_labels=len(sx),
        policy_choices_equal_executed_commands=True, voluntary_wait_frames=int(wait.sum()),
        longest_voluntary_wait_frames=longest_run(wait),
        only_stop_frames=int((legal == 0).sum()),
        active_wait_frames=int((wait & active).sum()),
        wait_legal_motion_count_histogram=histogram(legal[wait]),
        frozen_parent_choices_on_wait=histogram(parent_choices[wait]),
        previous_reason_on_wait=dict(Counter(FEEDBACK_REASONS[int(i)] for i in reason[wait])),
        pstop_on_wait=stats(pstop[wait]),
        wait_without_supported_risk_history=int((wait & ~history_valid).sum()),
        training_coverage=dict(source_frames=len(sx), active_frames=int(source_active.sum()),
            action_counts=histogram(source['choices']),
            active_stop_labels=int((source_active & source_stop).sum()),
            active_motion_labels=int((source_active & ~source_stop).sum()),
            source_groups=source_groups,
            active_stop_wait_timer_seconds=stats(source['observations'][source_active & source_stop, 434] * 5),
            active_move_wait_timer_seconds=stats(source['observations'][source_active & ~source_stop, 434] * 5),
            wait_timer_at_cap_stop_labels=int((source_active & source_stop & (source['observations'][:, 434] >= .999)).sum()),
            wait_timer_at_cap_move_labels=int((source_active & ~source_stop & (source['observations'][:, 434] >= .999)).sum())),
        diagnostic_ablations=ablations, selected_frame_neighbors=neighbors,
        physical_episode=dict(safe_success=False, timeout=True, collisions=0, seconds=row['seconds'],
            remaining_distance_m=row['final_xy_error_m'], localization_xy_p95_m=row['localization_xy_p95_m']),
        new_training_steps=0, new_physical_steps=0, vehicle_started=False,
        failed_trace_admitted_as_positive=False, model_or_guard_changed=False,
        limitations=['A legal motion action is not proof of useful, complete or safe recovery.',
            'Ablated inputs are diagnostic and may be off-manifold; no counterfactual command was executed.',
            'Nearest feature distance is diagnostic, not a calibrated confidence or a safe action label.',
            'This formerly frozen sensor case becomes diagnostic development data after examination.',
            'No new full-loop or target-board validation occurred.'],
        resource=budget.statistics(), goal_achieved=False, deployment_approved=False)
    for name, expected in hashes.items():
        assert digest(ROOT / name) == expected, name
    (output / 'RESULTS.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['selected_frame_neighbors','diagnostic_ablations']}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--case-id', default='long_risk_sensor_05_1')
    parser.add_argument('--cpu-budget', type=float, default=.15)
    main(parser.parse_args())
