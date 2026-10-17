"""Check the batched forecast against the reference on frozen causal replays."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from neural_navigation.decision_feedback_runtime import StopFeedbackRuntime
from neural_navigation.experiment_budget import ExperimentBudget
from neural_navigation.long_motion_risk import batch_predicted_body_risk
from neural_navigation.masked_actions import ACTIONS, COMMANDS
from neural_navigation.motion_risk import predicted_body_risk


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'reports/stop_feedback_20261015'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def compare(actual, expected):
    assert len(actual) == len(expected)
    error = 0.
    for first, second in zip(actual, expected):
        for key in ('command', 'supported_tracks', 'predicted_margin_contact_s'):
            assert first[key] == second[key], (key, first, second)
        key = 'minimum_predicted_clearance_m'
        if second[key] is None:
            assert first[key] is None
        else:
            difference = abs(first[key] - second[key])
            assert difference <= 2e-12, (difference, first, second)
            error = max(error, difference)
    return error


def main(output, cores):
    output.mkdir(parents=True, exist_ok=False)
    budget = ExperimentBudget(cores, memory_mib=2048, directory=output)
    frozen = BASE / 'early_yield_holdout_replay_20261016'
    source = json.loads((frozen / 'FROZEN_FINISHED_CASES.json').read_text())
    reference = json.loads((frozen / 'RESULTS.json').read_text())
    assert source['completed'] and reference['completed']
    cases_path = BASE / 'early_yield_remaining_protocol_20261016/heldout_cases.json'
    cases = json.loads(cases_path.read_text())
    bundle = ROOT / 'models/stop_feedback_20261015/early_yield_remaining_bc600_s231016/bc_portable'
    manifest = json.loads((bundle / 'manifest.json').read_text())
    paths = [Path(__file__), frozen / 'FROZEN_FINISHED_CASES.json', frozen / 'RESULTS.json',
        cases_path, bundle / 'manifest.json', *[bundle / name for name in manifest['files']]]
    paths += [ROOT / row['trace_path'] for row in source['rows']]
    for folder in ('neural_navigation', 'navrl'):
        paths += list((ROOT / 'training' / folder).rglob('*.py'))
    inputs = {str(path.relative_to(ROOT)): digest(path) for path in set(paths)}
    write(output / 'PLAN.json', dict(inputs=inputs, maximum_replay_frames=103,
        horizon_s=5., period_s=.2, absolute_clearance_tolerance_m=2e-12,
        paired_current_process_cpu_measurements=True, new_physical_steps=0,
        new_optimizer_updates=0, no_action_selection_changes=True))
    torch.set_num_threads(1)
    kwargs = dict(horizon_s=5., period_s=.2, margin_m=.04,
        uncertainty_m_s=.025, adaptive_track_uncertainty=True)
    track = dict(time=1., velocity=[-.2, .1], points=[[1., .1], [.9, -.1]],
        hits=4, velocity_uncertainty_m_s=.04)
    edge_errors = []
    for tracks in ([], [track], [dict(track, time=-1.)], [dict(track, hits=2)],
            [dict(track, velocity=[0., 0.])], [track, dict(track, points=[[.2, -.1]])]):
        for velocity in (None, [.12, -.3]):
            for horizon in (5., .47):
                options = dict(kwargs, horizon_s=horizon, measured_velocity=velocity)
                first = batch_predicted_body_risk(tracks, [.3, -.2, .7], 1.2,
                    COMMANDS, [.393924904, .329460022], **options)
                second = predicted_body_risk(tracks, [.3, -.2, .7], 1.2,
                    COMMANDS, [.393924904, .329460022], **options)
                edge_errors.append(compare(first, second))
                budget.tick()
    records = []
    for row in source['rows']:
        case = next(case for case in cases if case['id'] == row['case_id'])
        trace = ROOT / row['trace_path']
        assert digest(trace) == row['trace_sha256']
        with np.load(trace, allow_pickle=False) as archive:
            data = {key: archive[key].copy() for key in archive.files}
        expected = [record for record in reference['records'] if record['case_id'] == case['id']]
        runtime = StopFeedbackRuntime(bundle, ROOT / 'training' / case['scene'] / 'decision_map.yaml',
            np.asarray(case['start']) + np.asarray(case.get('initial_prior_offset', [0, 0, 0])),
            case['goal'], 'shortcut')

        class Replay:
            def __init__(self, actor):
                self.actor, self.cursor = actor, 0

            def predict(self, observation):
                i = self.cursor
                np.testing.assert_allclose(observation, data['observations'][i], rtol=0, atol=2e-6)
                choice = int(data['choices'][i])
                np.testing.assert_array_equal(self.actor.predict(observation), ACTIONS[choice])
                predictor = runtime.guard.predictive_constraint
                timestamp = float(data['labels'][i, 0])
                assert abs(float(predictor.timestamp) - timestamp) < 1e-8
                args = (predictor.tracks, np.asarray(runtime.core.odom), timestamp,
                    COMMANDS, runtime.guard.half * 2)
                options = dict(kwargs, measured_velocity=np.asarray(runtime.core.measured_velocity))
                started = time.process_time()
                actual = batch_predicted_body_risk(*args, **options)
                batch_ms = (time.process_time() - started) * 1000
                started = time.process_time()
                current_reference = predicted_body_risk(*args, **options)
                reference_ms = (time.process_time() - started) * 1000
                saved_reference = [expected[i]['predictions']['5.0'][str(k)] for k in range(12)]
                error = max(compare(actual, current_reference), compare(actual, saved_reference))
                records.append(dict(case_id=case['id'], frame=i,
                    maximum_clearance_error_m=error, batch_cpu_ms=batch_ms,
                    reference_cpu_ms=reference_ms, supported_tracks=actual[0]['supported_tracks']))
                self.cursor += 1
                return ACTIONS[choice].copy()

        replay = Replay(runtime.policy.policy)
        runtime.policy.policy = replay
        for i in range(len(data['choices'])):
            result = runtime.step(data['scans'][i], data['odometry'][i], data['measured_velocity'][i],
                float(data['labels'][i, 0]), capture_observation=True)
            np.testing.assert_allclose(result['command'], data['commands'][i], rtol=0, atol=2e-8)
            np.testing.assert_allclose(runtime.core.estimate, data['labels'][i, 4:7], rtol=0, atol=2e-8)
            budget.tick()
            if i % 20 == 0:
                write(output / 'STATUS.json', dict(status='causal_parity_replay',
                    frames=len(records), case_id=case['id']))
        assert replay.cursor == len(expected)
    assert len(records) == 103
    for name, expected in inputs.items():
        assert digest(ROOT / name) == expected, name
    timings = {}
    for name in ('batch', 'reference'):
        values = [record[name + '_cpu_ms'] for record in records]
        timings[name] = dict(zip(('p50_ms', 'p95_ms', 'p99_ms', 'maximum_ms'),
            map(float, np.r_[np.percentile(values, [50, 95, 99]), max(values)])))
    write(output / 'RESULTS.json', dict(completed=True, at=datetime.now(timezone.utc).isoformat(),
        replayed_frames=len(records), edge_cases=len(edge_errors),
        action_forecasts_checked=len(records) * 12,
        maximum_clearance_error_m=max([record['maximum_clearance_error_m'] for record in records] + edge_errors),
        paired_cpu=timings, records=records, resource=budget.statistics(),
        new_physical_steps=0, new_optimizer_updates=0, new_ppo_steps=0,
        no_changed_action_or_guard=True, goal_achieved=False, deployment_approved=False))
    write(output / 'STATUS.json', dict(status='complete', frames=len(records)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cpu-cores', type=float, default=.35)
    args = parser.parse_args()
    try:
        main(args.output, args.cpu_cores)
    except Exception as error:
        if args.output.exists():
            write(args.output / 'FAILURE.json', dict(error=repr(error), automatically_restarted=False))
        raise
