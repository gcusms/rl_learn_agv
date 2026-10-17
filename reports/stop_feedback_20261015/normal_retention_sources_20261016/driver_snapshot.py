"""Append verified normal mission labels without dropping any old example."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import torch

from experiments.build_release_recovery_sources import training_indices
from experiments.guard_reason_release_actor import GuardReasonReleaseActor
from experiments.long_risk_actor import digest
from experiments.release_fit_weights import fit_weights
from neural_navigation.experiment_budget import ExperimentBudget
from neural_navigation.masked_actions import ACTIONS, COMMANDS, masks_from_observation
from neural_navigation.release_observation import augment
from neural_navigation.sensors import LidarSpec


ROOT = Path(__file__).resolve().parents[2]


def write(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def main(args):
    parent, replay, model, output = [getattr(args, key).resolve()
        for key in ['parent', 'replay', 'model', 'output']]
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    budget = ExperimentBudget(.10, memory_mib=1536, directory=output)
    previous = json.loads((parent / 'RESULTS.json').read_text())
    report = json.loads((replay / 'RESULTS.json').read_text())
    assert previous['completed'] and previous['frames'] == 3252
    assert report['completed'] and len(report['rows']) == 2 and report['new_physical_episodes'] == 0
    assert all(r['complete_safe_source_admitted'] for r in report['rows'])
    assert digest(parent / 'demonstrations.npz') == previous['dataset_sha256']
    files = [Path(__file__), parent / 'RESULTS.json', parent / 'demonstrations.npz',
        replay / 'RESULTS.json', replay / 'PLAN.json', model / 'checkpoint.pt', model / 'parent_actor.pt']
    for module in ['build_release_recovery_sources', 'guard_reason_release_actor',
            'context_release_actor', 'release_actor', 'long_risk_actor', 'release_fit_weights']:
        files.append(ROOT / ('training/experiments/' + module + '.py'))
    files += [ROOT / r[k] for r in report['rows'] for k in
        ['trace_path', 'physical_source_trace', 'physical_source_report', 'reference_actor']]
    hashes = {str(p.relative_to(ROOT)): digest(p) for p in set(files)}
    write(output / 'PLAN.json', dict(inputs=hashes, parent_frames=3252,
        maximum_added_frames=476, original_arrays_byte_exact=True,
        no_failed_candidate_suffix_labels=True, only_inactive_terminal_stops_excluded=True,
        source_full_success_and_causal_replay_required=True, new_optimizer_updates=0,
        new_physical_episodes=0, new_ppo_steps=0, deployment_approved=False))
    actor = GuardReasonReleaseActor(torch.jit.load(str(model / 'parent_actor.pt'), map_location='cpu')).eval()
    saved = torch.load(model / 'checkpoint.pt', map_location='cpu', weights_only=False)
    actor.load_state_dict(saved['actor'])
    with np.load(parent / 'demonstrations.npz', allow_pickle=False) as archive:
        original = {key: archive[key].copy() for key in archive.files}
    pieces = {key: [value] for key, value in original.items()}
    sources = list(previous['sources'])
    next_id = int(original['source_ids'].max()) + 1
    excluded_terminal_frames = []
    for offset, row in enumerate(report['rows']):
        for key, expected in [('trace_path', 'trace_sha256'),
                ('physical_source_trace', 'physical_source_trace_sha256'),
                ('reference_actor', 'reference_actor_sha256')]:
            assert digest(ROOT / row[key]) == row[expected]
        source = json.loads((ROOT / row['physical_source_report']).read_text())
        assert source['completed'] and row['physical_source_row'] in source[row['physical_source_rows_key']]
        assert row['original_masks_actions_and_execution_reproduced'] and row['no_initialization_action_prefix']
        with np.load(ROOT / row['trace_path'], allow_pickle=False) as archive:
            trace = {key: archive[key].copy() for key in archive.files}
        observations, choices, masks = [trace[key] for key in ['observations', 'choices', 'masks']]
        np.testing.assert_array_equal(masks_from_observation(observations[:, :282]), masks)
        assert observations.shape == (len(choices), 625) and np.isfinite(observations).all()
        assert masks[np.arange(len(choices)), choices].all()
        reference = torch.jit.load(str(ROOT / row['reference_actor']), map_location='cpu').eval()
        lidar = LidarSpec(json.loads((ROOT / row['reference_actor']).parent.joinpath('hardware.json').read_text())['lidar'])
        with torch.no_grad():
            tensor = torch.from_numpy(observations)
            np.testing.assert_array_equal(reference(tensor[:, :439]).numpy(), ACTIONS[choices])
            active = actor.support(tensor).numpy()
        for observation, scan in zip(observations, trace['scans']):
            np.testing.assert_array_equal(augment(observation[:481], scan, lidar), observation)
        selected = training_indices(trace, row['physical_source_row'], active)
        np.testing.assert_allclose(trace['commands'][selected], COMMANDS[choices[selected]], rtol=0, atol=2e-8)
        extension = dict(observations=observations[selected], choices=choices[selected], masks=masks[selected],
            source_ids=np.full(len(selected), next_id + offset, np.int64), sensor_times=trace['labels'][selected, 0])
        for key, value in extension.items():
            pieces[key].append(value)
        sources.append(dict(case_id=row['case_id'], frames=len(selected),
            complete_episode_frames=len(choices), trace_path=row['trace_path'], trace_sha256=row['trace_sha256'],
            source_report=str((replay / 'RESULTS.json').relative_to(ROOT)),
            physical_source_trace_sha256=row['physical_source_trace_sha256'],
            source_is_offline_teacher=True, reference_is_previous_learning_model=True,
            no_initialization_action_prefix=True, safe_success=True,
            original_safe_physics_and_sensor_only_replay_verified=True,
            not_new_physical_success=True, excluded_inactive_arrival_frames=[len(choices) - 1]))
        excluded_terminal_frames.append(dict(case_id=row['case_id'], frame=len(choices) - 1))
        budget.tick()
    arrays = {key: np.concatenate(values) for key, values in pieces.items()}
    for key, value in original.items():
        assert arrays[key][:len(value)].tobytes() == value.tobytes(), key
    assert len(arrays['choices']) == 3728
    with torch.no_grad():
        tensor = torch.from_numpy(arrays['observations'])
        features, baseline = actor.features(tensor)
        active = actor.support(tensor).numpy()
    nominal = np.abs(baseline.numpy()[:, None] - ACTIONS[None]).max(2).argmin(1)
    disagreements = np.flatnonzero((~active) & (nominal != arrays['choices']))
    admission = dict(original3252_arrays_byte_exact=True, total_frames=len(tensor),
        added_frames=476, active_frames=int(active.sum()),
        inactive_target_disagreements=disagreements.tolist(),
        excluded_terminal_frames=excluded_terminal_frames,
        no_nonterminal_frames_discarded=True)
    if len(disagreements):
        write(output / 'SOURCE_ADMISSION.json', admission)
        write(output / 'RESULTS.json', dict(completed=True, admitted=False,
            reason='Complete-safe reference differs outside trainable support; no silent filtering',
            **admission, resource=budget.statistics(), new_ppo_steps=0, deployment_approved=False))
        return
    # Preserve the control's duplicate/phase weighting rule on its original312
    # features. Current reason96 dimensions change neither phase nor loss rule.
    _, weighting = fit_weights(features.numpy()[active, :312], arrays['choices'][active],
        arrays['observations'][active, 434] * 5, mode='recovery_phase')
    admission['fit_weighting'] = weighting
    write(output / 'SOURCE_ADMISSION.json', admission)
    np.savez_compressed(output / 'demonstrations.npz', **arrays)
    for name, expected in hashes.items():
        assert digest(ROOT / name) == expected, name
    budget.tick()
    write(output / 'RESULTS.json', dict(completed=True, admitted=True,
        at=datetime.now(timezone.utc).isoformat(), frames=len(tensor), parent_frames=3252,
        added_frames=476, sources=sources, dataset_sha256=digest(output / 'demonstrations.npz'),
        original_prefix_byte_exact=True, failed_latest_trace_excluded=True,
        excluded_failed_trace_sha256s=previous['excluded_failed_trace_sha256s'],
        added_tasks_are_training_regressions_not_unseen_tests=True,
        training_weight_preflight=weighting, new_optimizer_updates=0,
        new_physical_episodes=0, new_ppo_steps=0, resource=budget.statistics(),
        goal_achieved=False, deployment_approved=False))
    (output / 'driver_snapshot.py').write_text(Path(__file__).read_text())
    print(json.dumps(dict(admitted=True, frames=len(tensor), added_frames=476,
        old3252_byte_exact=True, inactive_target_disagreements=0)), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ['parent', 'replay', 'model', 'output']:
        parser.add_argument('--' + key, type=Path, required=True)
    main(parser.parse_args())
