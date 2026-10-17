"""Admit one complete-safe offline release demonstration without relabeling."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import torch

from experiments.build_release_recovery_sources import training_indices
from experiments.long_risk_actor import digest
from experiments.release_fit_weights import fit_weights
from experiments.release_actor import ReleaseActor
from experiments.sensor_release_teacher import SensorReleaseTeacher, contract as teacher_contract
from neural_navigation.experiment_budget import ExperimentBudget
from neural_navigation.goal_pose_runtime import check_files
from neural_navigation.masked_actions import ACTIONS, COMMANDS, masks_from_observation
from neural_navigation.release_observation import augment, contract
from neural_navigation.sensors import LidarSpec


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'reports/stop_feedback_20261015'


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def main(args):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    budget = ExperimentBudget(.15, memory_mib=1536, directory=output)
    parent = BASE / 'release_recovery_sources_v2_20261016'
    teacher = BASE / 'sensor_release_source_20261016'
    reference = ROOT / 'models/stop_feedback_20261015/release_phase300_s231016'
    parent_result = json.loads((parent / 'RESULTS.json').read_text())
    source_result = json.loads((teacher / 'RESULTS.json').read_text())
    assert parent_result['completed'] and source_result['completed']
    assert source_result['full_safe_expert_trajectory_exists'] and len(source_result['rows']) == 1
    assert json.loads((teacher / 'teacher_contract.json').read_text()) == teacher_contract()
    assert digest(parent / 'demonstrations.npz') == parent_result['dataset_sha256']
    row = source_result['rows'][0]
    assert row['safe_success'] and row['physical_arrival'] and row['protocol_valid']
    assert row['not_a_learned_model_success']
    assert not any(row[k] for k in ['collisions', 'timeout', 'false_arrival', 'physical_guards'])
    path = ROOT / row['trace_path']
    assert digest(path) == row['trace_sha256']
    excluded = parent_result['excluded_failed_trace_sha256s']
    assert digest(path) not in excluded
    manifest = json.loads((reference / 'portable/manifest.json').read_text())
    assert manifest['observation_contract'] == contract()
    check_files(reference / 'portable', manifest)
    inputs = [Path(__file__), parent / 'RESULTS.json', parent / 'demonstrations.npz',
        teacher / 'RESULTS.json', teacher / 'PLAN.json', teacher / 'teacher_contract.json', path,
        reference / 'checkpoint.pt', reference / 'parent_actor.pt',
        reference / 'portable/manifest.json',
        ROOT / 'training/experiments/sensor_release_teacher.py',
        ROOT / 'training/experiments/build_release_recovery_sources.py',
        ROOT / 'training/experiments/release_fit_weights.py',
        ROOT / 'training/neural_navigation/release_observation.py']
    inputs += [reference / 'portable' / name for name in manifest['files']]
    hashes = {str(p.relative_to(ROOT)): digest(p) for p in inputs}
    write(output / 'PLAN.json', dict(inputs=hashes, original_source_frames=parent_result['frames'],
        old_source_arrays_preserved=True, teacher_replay_and_sensor_quality_checked=True,
        observation_contract=contract(), teacher_contract=teacher_contract(),
        failed_trace_hashes_excluded=excluded, no_new_physical_or_optimizer_steps=True))
    with np.load(parent / 'demonstrations.npz', allow_pickle=False) as data:
        original = {k:data[k].copy() for k in data.files}
    with np.load(path, allow_pickle=False) as data:
        trace = {k:data[k].copy() for k in data.files}
    observations, choices, masks = trace['observations'], trace['choices'], trace['masks']
    assert observations.shape == (len(choices), 625) and np.isfinite(observations).all()
    np.testing.assert_array_equal(masks_from_observation(observations[:, :282]), masks)
    assert masks[np.arange(len(masks)), choices].all()
    saved = torch.jit.load(str(reference / 'portable/actor.pt'), map_location='cpu').eval()
    actor = ReleaseActor(torch.jit.load(str(reference / 'parent_actor.pt'), map_location='cpu')).eval()
    actor.load_state_dict(torch.load(reference / 'checkpoint.pt', map_location='cpu', weights_only=False)['actor'])
    with torch.no_grad():
        x = torch.from_numpy(observations)
        np.testing.assert_array_equal(saved(x).numpy(), ACTIONS[trace['reference_choices']])
        support = actor.support(x).numpy()
    lidar = LidarSpec(json.loads((reference / 'portable/hardware.json').read_text())['lidar'])
    forward = np.array([abs(part.mean()+lidar.mount[3]) <= np.deg2rad(35.)
        for part in np.array_split(lidar.angles, 72)])
    selector = SensorReleaseTeacher(forward)
    replayed = []
    for i, (obs, scan) in enumerate(zip(observations, trace['scans'])):
        np.testing.assert_array_equal(augment(obs[:481], scan, lidar), obs)
        entry = selector.select(obs, int(trace['reference_choices'][i]), bool(support[i]))
        assert entry['choice'] == choices[i] and entry['teacher_controlled'] == trace['teacher_controlled'][i]
        assert entry['reason'] == trace['teacher_reasons'][i]
        replayed.append(entry)
        if i % 40 == 0:
            budget.tick()
    assert int(trace['teacher_controlled'].sum()) == row['teacher_controlled_frames'] > 0
    selected = training_indices(trace, row, support)
    np.testing.assert_allclose(trace['commands'][selected], COMMANDS[choices[selected]], rtol=0, atol=2e-8)
    assert not ((~support) & (choices != trace['reference_choices'])).any()
    new_id = int(original['source_ids'].max())+1
    extension = dict(observations=observations[selected], choices=choices[selected], masks=masks[selected],
        source_ids=np.full(len(selected), new_id, np.int64), sensor_times=trace['labels'][selected, 0])
    arrays = {key:np.concatenate([value, extension[key]]) for key, value in original.items()}
    for key, value in original.items():
        assert arrays[key][:len(value)].tobytes() == value.tobytes()
    with torch.no_grad():
        x = torch.from_numpy(arrays['observations'])
        features, baseline = actor.features(x)
        active = actor.support(x).numpy()
    expected = np.abs(baseline.numpy()[:, None]-ACTIONS[None]).max(2).argmin(1)
    assert not ((~active) & (expected != arrays['choices'])).any()
    _, weighting = fit_weights(features.numpy()[active], arrays['choices'][active],
        arrays['observations'][active, 434]*5, mode='recovery_phase')
    np.savez_compressed(output / 'demonstrations.npz', **arrays)
    sources = list(parent_result['sources'])
    sources.append(dict(case_id=row['case_id'], source_report=str((teacher / 'RESULTS.json').relative_to(ROOT)),
        trace_path=row['trace_path'], trace_sha256=row['trace_sha256'], frames=len(selected),
        complete_episode_frames=len(choices), excluded_inactive_arrival_frames=[len(choices)-1],
        safe_success=True, collision_free_complete_episode=True, source_is_offline_teacher=True,
        teacher_controlled_frames=row['teacher_controlled_frames'], teacher_inputs_replayed=True))
    for name, expected in hashes.items():
        assert digest(ROOT / name) == expected, name
    budget.tick()
    result = dict(completed=True, at=datetime.now(timezone.utc).isoformat(), frames=len(x),
        parent_frames=len(original['choices']), added_frames=len(selected), sources=sources,
        dataset_sha256=digest(output / 'demonstrations.npz'), original_prefix_byte_exact=True,
        complete_teacher_episode_replayed_frames=len(choices), failed_latest_trace_excluded=True,
        excluded_failed_trace_sha256s=excluded, original_fresh_four_now_development=True,
        training_weight_preflight=weighting, new_training_steps=0, new_physical_steps=0,
        new_ppo_steps=0, resource=budget.statistics(), goal_achieved=False, deployment_approved=False)
    write(output / 'RESULTS.json', result)
    (output / 'driver_snapshot.py').write_text(Path(__file__).read_text())
    print(json.dumps({k:v for k,v in result.items() if k != 'sources'}, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    main(parser.parse_args())
