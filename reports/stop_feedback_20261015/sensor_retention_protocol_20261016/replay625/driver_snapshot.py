"""Causally reconstruct inputs on two complete-safe learned missions.

This reuses existing physical success, not a fresh candidate success. Only raw
scan, recorded odometry, measured velocity and their timestamps enter runtime.
The old learned actor must reproduce all original decisions and executions.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np
import torch

from experiments.long_risk_actor import digest
from neural_navigation.experiment_budget import ExperimentBudget
from neural_navigation.masked_actions import ACTIONS, masks_from_observation
from neural_navigation.release_runtime import ReleaseRuntime


ROOT = Path(__file__).resolve().parents[2]


def write(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def main(args):
    output, bundle, specifications = [getattr(args, key).resolve() for key in
        ['output', 'bundle', 'specifications']]
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    budget = ExperimentBudget(.35, memory_mib=2048, directory=output)
    protocol_path = specifications.parent / 'PLAN.json'
    protocol = json.loads(protocol_path.read_text())
    for name, expected in protocol['inputs'].items():
        assert digest(ROOT / name) == expected, name
    specs = json.loads(specifications.read_text())
    assert len(specs) == 2
    manifest = json.loads((bundle / 'manifest.json').read_text())
    files = [Path(__file__), specifications, protocol_path, bundle / 'manifest.json']
    files += [ROOT / name for name in protocol['inputs']]
    files += [bundle / name for name in manifest['files']]
    for folder in ['neural_navigation', 'navrl']:
        files += list((ROOT / 'training' / folder).rglob('*.py'))
    for spec in specs:
        files += [ROOT / spec[key] for key in
            ['trace', 'reference_actor', 'physical_source_report']]
        files += [p for p in (ROOT / 'training' / spec['case']['scene']).iterdir() if p.is_file()]
    hashes = {str(p.relative_to(ROOT)): digest(p) for p in set(files)}
    write(output / 'PLAN.json', dict(inputs=hashes,
        maximum_recorded_sources=2, sensor_only_replay=True,
        no_initialization_action_prefix=True, original_439_prefix_atol=2e-6,
        actions_masks_and_execution_must_reproduce=True,
        source_full_navigation_success_is_not_candidate_success=True,
        simulator_truth_not_input=True, new_physical_episodes=0,
        new_optimizer_updates=0, new_ppo_steps=0))
    (output / 'driver_snapshot.py').write_text(Path(__file__).read_text())
    report = dict(completed=False, rows=[], new_physical_episodes=0,
        new_optimizer_updates=0, new_ppo_steps=0, goal_achieved=False,
        deployment_approved=False, not_candidate_navigation_success=True)

    def save():
        budget.tick()
        report.update(at=datetime.now(timezone.utc).isoformat(), resource=budget.statistics())
        write(output / 'RESULTS.json', report)

    def finish(reason):
        for name, expected in hashes.items():
            assert digest(ROOT / name) == expected, name
        report.update(completed=True, reason=reason, simulator_loaded='mujoco' in sys.modules)
        save()

    save()
    for spec in specs:
        row = spec['physical_source_row']
        source = json.loads((ROOT / spec['physical_source_report']).read_text())
        assert source['completed']
        matches = [r for r in source[spec['physical_source_rows_key']]
            if r['case_id'] == row['case_id'] and r.get('variant') == row.get('variant')]
        assert len(matches) == 1 and matches[0] == row
        assert row['safe_success'] and row['physical_arrival'] and row['protocol_valid']
        assert not any(row[k] for k in ['collisions', 'timeout', 'false_arrival', 'physical_guards'])
        path, reference_path = [ROOT / spec[key] for key in ['trace', 'reference_actor']]
        assert digest(path) == row['trace_sha256']
        assert digest(reference_path.parent / 'hardware.json') == digest(bundle / 'hardware.json')
        reference = torch.jit.load(str(reference_path), map_location='cpu').eval()
        with np.load(path, allow_pickle=False) as archive:
            trace = {k: archive[k].copy() for k in archive.files}
        count = len(trace['choices'])
        reference_dim = spec.get('reference_observation_dimensions', 439)
        assert reference_dim in (439, 625)
        assert count == row['frames'] and trace['observations'].shape == (count, reference_dim)
        if 'model_controlled' in trace:
            assert trace['model_controlled'].all(), 'Only full learned missions are admitted'
        assert trace['labels'][0, 0] == 0
        assert np.all(np.diff(trace['labels'][:, 0]) > 0)
        cursor = [0]

        class ReplayPolicy:
            def predict(self, observation):
                with torch.no_grad():
                    action = reference(torch.from_numpy(
                        np.asarray(observation[:reference_dim], np.float32)[None])).numpy()[0]
                np.testing.assert_array_equal(action, ACTIONS[trace['choices'][cursor[0]]])
                return action.copy()

        case = spec['case']
        prior = np.asarray(case['start']) + np.asarray(case.get('initial_prior_offset', [0, 0, 0]))
        runtime = ReleaseRuntime(bundle, ROOT / 'training' / case['scene'] / 'decision_map.yaml',
            prior, case['goal'], 'shortcut')
        runtime.policy.policy = ReplayPolicy()
        observations = []
        maximum = 0.
        try:
            for index in range(count):
                cursor[0] = index
                result = runtime.step(trace['scans'][index], trace['odometry'][index],
                    trace['measured_velocity'][index], float(trace['labels'][index, 0]),
                    capture_observation=True)
                observation = np.asarray(result['policy_observation'], np.float32)
                np.testing.assert_allclose(observation[:reference_dim], trace['observations'][index],
                    rtol=0, atol=2e-6)
                maximum = max(maximum,
                    float(np.max(abs(observation[:reference_dim] - trace['observations'][index]))))
                np.testing.assert_array_equal(masks_from_observation(observation[:282]), trace['masks'][index])
                np.testing.assert_array_equal(result['proposed_action'], ACTIONS[trace['choices'][index]])
                np.testing.assert_allclose(result['command'], trace['commands'][index], rtol=0, atol=2e-8)
                observations.append(observation)
                budget.tick()
                if index % 50 == 0:
                    write(output / 'STATUS.json', dict(status='causal_sensor_replay',
                        case_id=spec['id'], matched_frames=index + 1, total_frames=count,
                        completed_sources=len(report['rows'])))
            assert result['stop_reason'] == 'estimated_goal_reached'
        except Exception as error:
            write(output / 'FAILURE.json', dict(case_id=spec['id'], error=repr(error),
                matched_frames=len(observations), partial_source_not_admitted=True))
            finish('Source did not causally reproduce; no partial-source labels or fitting')
            return
        trace['original_reference_observations'] = trace['observations']
        trace['original_439_observations'] = trace['observations'][:, :439]
        trace['observations'] = np.asarray(observations)
        destination = output / (spec['id'] + '.npz')
        np.savez_compressed(destination, **trace)
        report['rows'].append(dict(case_id=spec['id'], frames=count,
            trace_path=str(destination.relative_to(ROOT)), trace_sha256=digest(destination),
            physical_source_report=spec['physical_source_report'],
            physical_source_rows_key=spec['physical_source_rows_key'],
            physical_source_trace=spec['trace'], physical_source_trace_sha256=row['trace_sha256'],
            physical_source_row=row, reference_actor=spec['reference_actor'],
            reference_actor_sha256=digest(reference_path), safe_success=True,
            reference_observation_dimensions=reference_dim,
            maximum_reference_observation_absolute_difference=maximum,
            maximum_439_prefix_absolute_difference=float(np.max(abs(
                trace['original_439_observations'] - trace['observations'][:, :439]))),
            original_masks_actions_and_execution_reproduced=True,
            complete_safe_source_admitted=True, no_initialization_action_prefix=True,
            original_success_is_reused_not_candidate_navigation_success=True))
        save()
        print(json.dumps(dict(case_id=spec['id'], matched_frames=count,
            maximum_439_prefix_absolute_difference=maximum,
            complete_safe_source_admitted=True, new_physical_episodes=0)), flush=True)
        del runtime, reference
    finish('Both complete-safe normal reference sensor histories reproduced; label admission remains separate')
    write(output / 'STATUS.json', dict(status='complete', converted_sources=len(report['rows'])))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ['output', 'bundle', 'specifications']:
        parser.add_argument('--' + key, type=Path, required=True)
    main(parser.parse_args())
