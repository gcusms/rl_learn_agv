"""Build measured release context from complete-safe, already finished traces."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from experiments.long_risk_actor import digest
from neural_navigation.experiment_budget import ExperimentBudget
from neural_navigation.masked_actions import ACTIONS, masks_from_observation
from neural_navigation.release_observation import augment, contract
from neural_navigation.sensors import LidarSpec


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'reports/stop_feedback_20261015'


def write(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def safe(row):
    return (row['safe_success'] and row['physical_arrival'] and row['protocol_valid']
        and not any(row[key] for key in ['collisions', 'timeout', 'false_arrival', 'physical_guards']))


def main(output):
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    budget = ExperimentBudget(.15, memory_mib=2048, directory=output)
    risk_sources = BASE / 'long_risk_sources_20261016'
    risk_report = json.loads((risk_sources / 'RESULTS.json').read_text())
    validation_path = BASE / 'long_risk_validation_20261016/RESULTS.json'
    validation = json.loads(validation_path.read_text())
    assert risk_report['completed'] and validation['completed']
    assert digest(risk_sources / 'demonstrations.npz') == risk_report['dataset_sha256']
    bundle = ROOT / 'models/stop_feedback_20261015/long_risk_bc1000_s231016/portable'
    hardware = json.loads((bundle / 'hardware.json').read_text())
    lidar = LidarSpec(hardware['lidar'])
    with np.load(risk_sources / 'demonstrations.npz', allow_pickle=False) as archive:
        original = {key: archive[key].copy() for key in archive.files}
    paths = [Path(__file__), ROOT / 'training/neural_navigation/release_observation.py',
        ROOT / 'training/neural_navigation/sensors.py', ROOT / 'training/neural_navigation/experiment_budget.py',
        risk_sources / 'RESULTS.json', risk_sources / 'demonstrations.npz', validation_path,
        bundle / 'hardware.json', bundle / 'actor.pt',
        BASE / 'long_risk_protocol_20261016/PLAN.json',
        BASE / 'long_risk_protocol_20261016/new_sensor_cases.json']
    sources = []
    for source_id, record in enumerate(risk_report['sources']):
        report_path = BASE / record['source'] / 'RESULTS.json'
        report = json.loads(report_path.read_text())
        row = report['rows'][0]
        assert report['completed'] and report['full_safe_expert_trajectory_exists'] and safe(row)
        indices = np.flatnonzero(original['source_ids'] == source_id)
        sources.append((row, original['observations'][indices], report_path, indices))
    for row in validation['rows']:
        if safe(row):
            sources.append((row, None, validation_path, None))
    assert len(sources) == 10
    paths += [report_path for _, _, report_path, _ in sources]
    paths += [ROOT / row['trace_path'] for row, _, _, _ in sources]
    inputs = {str(path.relative_to(ROOT)): digest(path) for path in set(paths)}
    write(output / 'PLAN.json', dict(inputs=inputs, contract=contract(),
        complete_safe_sources_only=True, failed_latest_trace_excluded=True,
        all_inputs_from_actual_recorded_scan_and_policy_observation=True,
        old_evaluation_seeds_are_now_development=True, fresh_four_seeds_not_used=True,
        new_optimizer_updates=0, new_physical_steps=0, new_ppo_steps=0))
    arrays = dict(observations=[], choices=[], masks=[], source_ids=[], sensor_times=[])
    summaries = []
    current_actor = torch.jit.load(str(bundle / 'actor.pt'), map_location='cpu').eval()
    for source_id, (row, old, report_path, indices) in enumerate(sources):
        trace_path = ROOT / row['trace_path']
        assert digest(trace_path) == row['trace_sha256']
        with np.load(trace_path, allow_pickle=False) as archive:
            trace = {key: archive[key].copy() for key in archive.files}
        if old is None:
            old = trace['observations']
            assert old.shape[1] == 481
            with torch.no_grad():
                np.testing.assert_array_equal(current_actor(torch.from_numpy(old)).numpy(), ACTIONS[trace['choices']])
        else:
            np.testing.assert_allclose(old[:, :439], trace['observations'], rtol=0, atol=2e-6)
            np.testing.assert_array_equal(original['choices'][indices], trace['choices'])
            np.testing.assert_array_equal(original['masks'][indices], trace['masks'])
            np.testing.assert_array_equal(original['sensor_times'][indices], trace['labels'][:, 0])
        np.testing.assert_array_equal(masks_from_observation(old[:, :282]), trace['masks'])
        assert trace['masks'][:, 0].all()
        assert trace['masks'][np.arange(len(old)), trace['choices']].all()
        for i, observation in enumerate(old):
            scan = trace['scans'][i]
            clean = np.nan_to_num(scan, nan=12., posinf=12., neginf=.1)
            pooled = np.array([part.min() for part in np.array_split(clean, 72)]) / lidar.range_max
            np.testing.assert_allclose(observation[144:216], np.clip(pooled, -1, 1), rtol=0, atol=1e-7)
            augmented = augment(observation, scan, lidar)
            np.testing.assert_array_equal(augmented[:481], observation)
            arrays['observations'].append(augmented)
            arrays['choices'].append(trace['choices'][i])
            arrays['masks'].append(trace['masks'][i])
            arrays['source_ids'].append(source_id)
            arrays['sensor_times'].append(trace['labels'][i, 0])
            if i % 40 == 0:
                budget.tick()
        summaries.append(dict(case_id=row['case_id'], source_report=str(report_path.relative_to(ROOT)),
            trace_path=row['trace_path'], trace_sha256=row['trace_sha256'], frames=len(old),
            safe_success=True, collision_free_complete_episode=True,
            source_is_offline_teacher=indices is not None))
        write(output / 'STATUS.json', dict(status='append_measured_release_context',
            finished_sources=len(summaries), frames=len(arrays['choices'])))
        budget.tick()
    for key, dtype in [('observations', np.float32), ('choices', np.int64), ('masks', bool),
            ('source_ids', np.int64), ('sensor_times', float)]:
        arrays[key] = np.asarray(arrays[key], dtype)
    assert arrays['observations'].shape == (1790, 625)
    np.savez_compressed(output / 'demonstrations.npz', **arrays)
    for name, expected in inputs.items():
        assert digest(ROOT / name) == expected, name
    budget.tick()
    write(output / 'RESULTS.json', dict(completed=True, frames=len(arrays['choices']),
        sources=summaries, dataset_sha256=digest(output / 'demonstrations.npz'),
        failed_latest_trace_excluded=True, new_physical_steps=0, new_optimizer_updates=0,
        new_ppo_steps=0, resource=budget.statistics(), goal_achieved=False, deployment_approved=False))
    write(output / 'STATUS.json', dict(status='complete', frames=len(arrays['choices'])))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    main(parser.parse_args().output)
