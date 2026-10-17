"""Extend release training with already verified complete safe sensor traces."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from experiments.release_actor import ReleaseActor
from experiments.release_fit_weights import fit_weights
from neural_navigation.experiment_budget import ExperimentBudget
from neural_navigation.masked_actions import ACTIONS, COMMANDS, masks_from_observation
from neural_navigation.release_observation import augment, contract
from neural_navigation.sensors import LidarSpec


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'reports/stop_feedback_20261015'
MODEL = ROOT / 'models/stop_feedback_20261015/release_context_bc2000_s231016'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def training_indices(trace, row, support):
    """A terminal arrival stop is separate from the sampled policy action."""
    count = len(trace['choices'])
    if trace['commands'].shape != (count, 2) or not np.isfinite(trace['commands']).all():
        raise ValueError('Finite actual execution required')
    expected = COMMANDS[trace['choices']]
    difference = (np.abs(trace['commands'] - expected) > 2e-8).any(1)
    if (not count or row['stop_reasons'].get('estimated_goal_reached', 0) != 1
            or np.max(np.abs(trace['commands'][-1])) > 1e-8
            or support[-1] or np.linalg.norm(trace['observations'][-1, 218:220] * 12) > .3
            or difference[:-1].any()):
        raise ValueError('Only a verified inactive terminal arrival may be excluded; other execution mismatches reject the source')
    return np.arange(count - 1)


def main(args):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    budget = ExperimentBudget(.15, memory_mib=1536, directory=output)
    original = BASE / 'release_sources_20261016'
    source_result = json.loads((original / 'RESULTS.json').read_text())
    assert source_result['completed'] and digest(original / 'demonstrations.npz') == source_result['dataset_sha256']
    with np.load(original / 'demonstrations.npz', allow_pickle=False) as archive:
        arrays = {k: [archive[k].copy()] for k in archive.files}
    old_frames = len(arrays['choices'][0])
    validation_path = BASE / 'release_validation_20261016/RESULTS.json'
    validation = json.loads(validation_path.read_text())
    assert validation['completed']
    added = [r for r in validation['rows'] if r['safe_success']
        and r['group'] in ('repaired_wait_condition', 'frozen_new_noise')]
    assert len(added) == 4
    negative_hashes = {r['trace_sha256'] for r in validation['rows'] if not r['safe_success']}
    sources = list(source_result['sources'])
    source_ids_start = int(arrays['source_ids'][0].max()) + 1
    files = [Path(__file__), original / 'demonstrations.npz', original / 'RESULTS.json',
        validation_path, MODEL / 'checkpoint.pt', MODEL / 'parent_actor.pt', MODEL / 'portable/actor.pt',
        MODEL / 'portable/hardware.json', ROOT / 'training/neural_navigation/release_observation.py']
    files += [ROOT / r['trace_path'] for r in added]
    hashes = {str(p.relative_to(ROOT)): digest(p) for p in files}
    write(output / 'PLAN.json', dict(inputs=hashes, observation_contract=contract(),
        parent_frames=old_frames, complete_safe_added_cases=[r['case_id'] for r in added],
        failed_trace_sha256s_excluded=sorted(negative_hashes), old_prefix_kept_byte_exact=True,
        original_frozen_sensor_cases_are_now_development=True, new_training_steps=0, new_physics_episodes=0))
    lidar = LidarSpec(json.loads((MODEL / 'portable/hardware.json').read_text())['lidar'])
    saved = torch.jit.load(str(MODEL / 'portable/actor.pt'), map_location='cpu').eval()
    actor = ReleaseActor(torch.jit.load(str(MODEL / 'parent_actor.pt'), map_location='cpu')).eval()
    actor.load_state_dict(torch.load(MODEL / 'checkpoint.pt', map_location='cpu', weights_only=False)['actor'])
    for offset, row in enumerate(added):
        assert row['physical_arrival'] and row['protocol_valid'] and row['learned_model_only'] and row['no_teacher']
        assert not any(row[k] for k in ('collisions', 'timeout', 'false_arrival', 'physical_guards'))
        path = ROOT / row['trace_path']
        assert digest(path) == row['trace_sha256'] and digest(path) not in negative_hashes
        with np.load(path, allow_pickle=False) as archive:
            trace = {k: archive[k].copy() for k in archive.files}
        obs, choices, masks = trace['observations'], trace['choices'], trace['masks']
        assert obs.shape == (len(choices), 625) and np.isfinite(obs).all()
        np.testing.assert_array_equal(masks_from_observation(obs[:, :282]), masks)
        assert masks[np.arange(len(choices)), choices].all()
        with torch.no_grad():
            np.testing.assert_array_equal(saved(torch.from_numpy(obs)).numpy(), ACTIONS[choices])
            support = actor.support(torch.from_numpy(obs)).numpy()
        for observation, scan in zip(obs, trace['scans']):
            np.testing.assert_array_equal(augment(observation[:481], scan, lidar), observation)
        selected = training_indices(trace, row, support)
        np.testing.assert_allclose(trace['commands'][selected], COMMANDS[choices[selected]], rtol=0, atol=2e-8)
        for name, value in [('observations', obs[selected]), ('choices', choices[selected]), ('masks', masks[selected]),
                ('source_ids', np.full(len(selected), source_ids_start + offset, np.int64)),
                ('sensor_times', trace['labels'][selected, 0])]:
            arrays[name].append(value)
        sources.append(dict(case_id=row['case_id'], source_report=str(validation_path.relative_to(ROOT)),
            trace_path=row['trace_path'], trace_sha256=row['trace_sha256'], frames=len(selected),
            complete_episode_frames=len(obs), excluded_inactive_arrival_frames=[len(obs) - 1],
            safe_success=True, collision_free_complete_episode=True, source_is_offline_teacher=False))
        budget.tick()
    arrays = {k: np.concatenate(v) for k, v in arrays.items()}
    with np.load(original / 'demonstrations.npz', allow_pickle=False) as archive:
        for k in archive.files:
            assert arrays[k][:old_frames].tobytes() == archive[k].tobytes(), k
    x = torch.from_numpy(arrays['observations'])
    with torch.no_grad():
        features, parent = actor.features(x)
        active = actor.support(x).numpy()
    parent_choice = np.abs(parent.numpy()[:, None] - ACTIONS[None]).max(2).argmin(1)
    assert not ((~active) & (parent_choice != arrays['choices'])).any(), 'Source behavior outside trainable support'
    weight_modes = {}
    for mode in ['stop_move', 'recovery_phase']:
        _, weight_modes[mode] = fit_weights(features.numpy()[active], arrays['choices'][active],
            arrays['observations'][active, 434] * 5, mode=mode)
    np.savez_compressed(output / 'demonstrations.npz', **arrays)
    for name, expected in hashes.items():
        assert digest(ROOT / name) == expected, name
    budget.tick()
    write(output / 'driver_snapshot.py.sha256.json', dict(sha256=digest(Path(__file__))))
    (output / 'driver_snapshot.py').write_text(Path(__file__).read_text())
    result = dict(completed=True, at=datetime.now(timezone.utc).isoformat(), frames=len(x),
        parent_frames=old_frames, added_frames=len(x) - old_frames, sources=sources,
        dataset_sha256=digest(output / 'demonstrations.npz'), original_prefix_byte_exact=True,
        unique_full_observations=len({r.tobytes() for r in arrays['observations']}),
        failed_latest_trace_excluded=True, excluded_failed_trace_sha256s=sorted(negative_hashes),
        original_fresh_four_now_development=True, training_weight_preflight=weight_modes,
        new_training_steps=0, new_physical_steps=0, new_ppo_steps=0,
        resource=budget.statistics(), goal_achieved=False, deployment_approved=False)
    write(output / 'RESULTS.json', result)
    print(json.dumps({k:v for k,v in result.items() if k != 'sources'}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    main(parser.parse_args())
