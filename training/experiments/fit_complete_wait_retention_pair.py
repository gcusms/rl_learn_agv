"""Bounded paired fit for a full-mission wait regression, with normal retention.

All previous labels remain byte-exact. Add one causally reconstructed complete
reference and four complete, efficient autonomous candidate missions. Neither
failed turn probes nor inefficient candidate waits are positive labels.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import shutil

import numpy as np
import torch
import torch.nn.functional as F

from experiments.build_release_recovery_sources import training_indices
from experiments.guard_reason_release_actor import GuardReasonReleaseActor, export
from experiments.long_risk_actor import digest
from experiments.phase_tail_loss import phase_tail_loss
from experiments.refit_phase_tail_pair import optimizer_digest, restore_fit, write
from experiments.release_fit_weights import fit_weights
from experiments.train_long_risk_actor import state_digest
from neural_navigation.experiment_budget import ExperimentBudget
from neural_navigation.masked_actions import ACTIONS, masks_from_observation
from neural_navigation.release_observation import augment
from neural_navigation.sensors import LidarSpec


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'reports/stop_feedback_20261015'


def load(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key].copy() for key in archive.files}


def main(args):
    protocol, fits = args.protocol.resolve(), args.output.resolve()
    protocol.mkdir(parents=True, exist_ok=False)
    fits.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    budget = ExperimentBudget(.1, memory_mib=1400, directory=fits)
    model = ROOT / 'models/stop_feedback_20261015/sensor_retention_pair_s231016/retained500'
    old_dataset = BASE / 'sensor_retention_protocol_20261016/demonstrations.npz'
    replay_folder = BASE / 'full_wait_regression_replay_20261016'
    replay_path = replay_folder / 'RESULTS.json'
    validation_path = BASE / 'sensor_retention_validation_20261016/RESULTS.json'
    task_path = BASE / 'context_feedback_longfit_protocol_20261016/tasks.json'
    old_fit = json.loads((model / 'RESULTS.json').read_text())
    replay = json.loads(replay_path.read_text())
    validation = json.loads(validation_path.read_text())
    tasks = json.loads(task_path.read_text())
    assert old_fit['completed'] and old_fit['correct_source_labels'] == old_fit['total_source_labels'] == 4067
    assert digest(model / 'checkpoint.pt') == old_fit['checkpoint_sha256']
    assert digest(old_dataset) == old_fit['source_sha256']
    assert replay['completed'] and replay['source_frames'] == 186
    assert replay['complete_reference_masks_actions_commands_reproduced']
    assert replay['maximum439_input_difference'] <= 2e-6
    assert validation['completed']
    references = {r['case_id']: r for r in tasks['reference']['normal18']}
    sources = [dict(row=replay['old_complete_safe_source_row'],
        trace=replay['source_trace_path'], sha256=replay['source_trace_sha256'],
        kind='causally_reconstructed_complete_reference439',
        actor=ROOT / 'models/stop_feedback_20261015/recovery_followup_bc400_s231016/bc_portable/actor.pt',
        observation_dimensions=439)]
    retained_ids = ['motion_profile_015', 'recovery_holdout_00', 'recovery_holdout_02', 'recovery_holdout_01']
    for case_id in retained_ids:
        row = next(r for r in validation['groups']['normal_other16']['rows'] if r['case_id'] == case_id)
        ref = references[case_id]
        assert row['pure_learned_model']
        assert all(row[k] <= ref[k] * 1.2 + 1e-8 for k in ('seconds', 'travel_m'))
        sources.append(dict(row=row, trace=row['trace_path'], sha256=row['trace_sha256'],
            kind='complete_efficient_native625_candidate_retention',
            actor=model / 'portable/actor.pt', observation_dimensions=625))
    files = [Path(__file__), replay_path, replay_folder / 'PLAN.json', validation_path,
        task_path, old_dataset, model / 'checkpoint.pt', model / 'RESULTS.json',
        model / 'parent_actor.pt', model / 'portable/hardware.json']
    for source in sources:
        files += [ROOT / source['trace'], source['actor']]
    for name in ['build_release_recovery_sources', 'guard_reason_release_actor', 'context_release_actor',
            'release_actor', 'long_risk_actor', 'phase_tail_loss', 'refit_phase_tail_pair',
            'release_fit_weights', 'train_long_risk_actor']:
        files.append(ROOT / ('training/experiments/' + name + '.py'))
    for folder in ['neural_navigation', 'navrl']:
        files += list((ROOT / 'training' / folder).rglob('*.py'))
    hashes = {str(p.relative_to(ROOT)): digest(p) for p in set(files)}
    plan = dict(inputs=hashes, old_labels=4067, prior_supervised_updates=3000,
        fits=['control500', 'retained500'], maximum_updates_per_fit=500,
        seed_restored_from_parent_checkpoint=True,
        same_initial_actor_adam_rng_loss_learning_rate_architecture_and_support=True,
        only_changed_factor='complete_early_safe_reference_and_four_efficient_mission_labels',
        old_label_prefix_byte_exact=True,
        excludes_inefficient58_6_second_candidate_and_all_failed_turn_probes=True,
        turn_probe_failures_are_not_complete_success_sources=True,
        exact_fit_all_original_and_added_labels_required_for_export=True,
        no_physical_simulation_or_runtime_teacher=True,
        actions_body4cm_masks_localizer_router_docking_unchanged=True,
        all_added_cases_are_now_training_regressions_not_generalization=True,
        fresh_noise2_from_previous_fit_remain_out_of_training_data=True,
        new_ppo_steps=0, deployment_approved=False)
    write(protocol / 'PLAN.json', plan)
    write(fits / 'PLAN.json', plan)
    (fits / 'driver_snapshot.py').write_text(Path(__file__).read_text())
    write(fits / 'STATUS.json', dict(status='admitting_complete_sources'))
    original = load(old_dataset)
    pieces = {key: [value] for key, value in original.items()}
    next_id = int(original['source_ids'].max()) + 1
    actor = GuardReasonReleaseActor(torch.jit.load(str(model / 'parent_actor.pt'), map_location='cpu')).eval()
    actor.load_state_dict(torch.load(model / 'checkpoint.pt', map_location='cpu', weights_only=False)['actor'])
    lidar = LidarSpec(json.loads((model / 'portable/hardware.json').read_text())['lidar'])
    source_rows = []
    for offset, source in enumerate(sources):
        row = source['row']
        assert row['safe_success'] and row['physical_arrival'] and row['protocol_valid']
        assert not any(row[k] for k in ('collisions', 'timeout', 'false_arrival', 'physical_guards'))
        trace_path = ROOT / source['trace']
        assert digest(trace_path) == source['sha256']
        trace = load(trace_path)
        observations, choices, masks = (trace[k] for k in ('observations', 'choices', 'masks'))
        assert observations.shape == (len(choices), 625) and np.isfinite(observations).all()
        assert trace['labels'][0, 0] == 0. and np.all(np.diff(trace['labels'][:, 0]) > 0.)
        np.testing.assert_array_equal(masks_from_observation(observations[:, :282]), masks)
        assert masks[:, 0].all() and masks[np.arange(len(choices)), choices].all()
        reference_actor = torch.jit.load(str(source['actor']), map_location='cpu').eval()
        with torch.no_grad():
            reference_action = reference_actor(torch.from_numpy(observations[:, :source['observation_dimensions']])).numpy()
            np.testing.assert_array_equal(reference_action, ACTIONS[choices])
            support = actor.support(torch.from_numpy(observations)).numpy()
        for observation, scan in zip(observations, trace['scans']):
            np.testing.assert_array_equal(augment(observation[:481], scan, lidar), observation)
        selected = training_indices(trace, row, support)
        extension = dict(observations=observations[selected], choices=choices[selected], masks=masks[selected],
            sensor_times=trace['labels'][selected, 0], source_ids=np.full(len(selected), next_id + offset, np.int64))
        for key, value in extension.items():
            pieces[key].append(value)
        source_rows.append(dict(case_id=row['case_id'], kind=source['kind'], frames=len(selected),
            full_episode_frames=len(choices), excluded_inactive_terminal_frames=1,
            trace_path=source['trace'], trace_sha256=source['sha256'],
            complete_safe_source_is_reused_not_new_success=True))
        budget.tick()
        del reference_actor
    arrays = {key: np.concatenate(values) for key, values in pieces.items()}
    for key, value in original.items():
        assert arrays[key][:len(value)].tobytes() == value.tobytes(), key
    assert len(arrays['choices']) == 4949
    dataset = protocol / 'demonstrations.npz'
    np.savez_compressed(dataset, **arrays)
    write(protocol / 'SOURCE_ADMISSION.json', dict(admitted=True, parent_labels=4067,
        added_labels=882, total_labels=4949, original_prefix_byte_exact=True,
        source_rows=source_rows, dataset_sha256=digest(dataset)))
    result = dict(completed=False, rows=[], new_ppo_steps=0,
        goal_achieved=False, deployment_approved=False)
    x = torch.from_numpy(arrays['observations'])
    initial_actor = initial_optimizer = None
    for name, count in [('control500', 4067), ('retained500', 4949)]:
        folder = fits / name
        folder.mkdir()
        actor = GuardReasonReleaseActor(torch.jit.load(str(model / 'parent_actor.pt'), map_location='cpu')).eval()
        saved, optimizer = restore_fit(model / 'checkpoint.pt', actor)
        current_actor, current_optimizer = state_digest(actor), optimizer_digest(optimizer)
        if initial_actor is None:
            initial_actor, initial_optimizer = current_actor, current_optimizer
        assert (current_actor, current_optimizer) == (initial_actor, initial_optimizer)
        frozen = state_digest(actor.parent)
        with torch.no_grad():
            features, baseline = actor.features(x)
            active = actor.support(x).numpy()
        nominal = np.abs(baseline.numpy()[:, None] - ACTIONS[None]).max(2).argmin(1)
        assert not ((~active) & (nominal != arrays['choices'])).any()
        selected = active.copy()
        selected[count:] = False
        weights, weighting = fit_weights(features.numpy()[selected, :312], arrays['choices'][selected],
            arrays['observations'][selected, 434] * 5, mode='recovery_phase')
        np.savez_compressed(folder / 'fixed_source_weights.npz', active=selected, weights=weights)
        feat = features[selected].detach()
        target = torch.from_numpy(arrays['choices'][selected])
        legal = torch.from_numpy(arrays['masks'][selected])
        waiting = torch.from_numpy(arrays['observations'][selected, 434] * 5)
        weight = torch.from_numpy(weights).to(feat.dtype)
        records = []
        for update in range(1, 501):
            logits = (actor.head(feat) + 2. * feat[:, 76:88]).masked_fill(~legal, -1e8)
            loss = phase_tail_loss(F.cross_entropy(logits, target, reduction='none'),
                target, waiting, weight, focus_share=.5)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.head.parameters(), 1.)
            optimizer.step()
            if update % 10 == 0:
                budget.tick()
            if update % 25 == 0:
                with torch.no_grad():
                    correct = (actor(x).numpy() == ACTIONS[arrays['choices']]).all(1)
                records.append(dict(new_update=update, loss=float(loss.detach()),
                    original4067_correct=int(correct[:4067].sum()),
                    added882_correct=int(correct[4067:].sum()), all4949_correct=int(correct.sum())))
                write(fits / 'STATUS.json', dict(status='paired_complete_mission_fit', model=name, **records[-1]))
        assert state_digest(actor.parent) == frozen
        with torch.no_grad():
            correct = (actor(x).numpy() == ACTIONS[arrays['choices']]).all(1)
        all_fit = bool(correct.all())
        own_fit = bool(correct[:count].all())
        shutil.copy2(model / 'parent_actor.pt', folder / 'parent_actor.pt')
        torch.save(dict(actor=actor.state_dict(), optimizer=optimizer.state_dict(),
            torch_rng=torch.get_rng_state(), numpy_rng=np.random.get_state(), python_rng=random.getstate(),
            parent_actor_sha256=digest(folder / 'parent_actor.pt'), contract=saved['contract'],
            head_contract=saved['head_contract'], learning_support=saved['learning_support'],
            supervised_updates=3500, new_supervised_updates=500, source_sha256=digest(dataset),
            new_ppo_steps=0, exact_physical_rollout_resume=False), folder / 'checkpoint.pt')
        if all_fit:
            invalid, near, stopped = (arrays['observations'][:16].copy() for _ in range(3))
            invalid[:, 435] = 0.
            near[:, 218:220] = 0.
            near[:, 302] = 1.
            stopped[:, 235:244] = stopped[:, 267:269] = 0.
            export(actor, ROOT / 'models/stop_feedback_20261015/long_risk_bc1000_s231016/portable',
                folder / 'portable', np.r_[arrays['observations'], invalid, near, stopped])
        row = dict(model=name, model_path=str(folder.relative_to(ROOT)), updates=3500,
            new_supervised_updates=500, cumulative_supervised_updates=3500,
            initial_actor_tensor_sha256=current_actor, initial_optimizer_sha256=current_optimizer,
            own_training_labels=count, own_training_fit_passed=own_fit,
            correct_source_labels=int(correct.sum()), total_source_labels=4949,
            correct_original4067_labels=int(correct[:4067].sum()),
            original3252_all_retained=bool(correct[:3252].all()),
            original3728_all_retained=bool(correct[:3728].all()),
            original4067_all_retained=bool(correct[:4067].all()),
            safe_stop_labels_all_retained=bool(correct[arrays['choices'] == 0].all()),
            wrong_source_indices=np.flatnonzero(~correct).tolist(),
            exported_for_physical_validation=all_fit, checkpoint_sha256=digest(folder / 'checkpoint.pt'),
            frozen_parent_tensor_sha256=frozen, optimizer_and_rng_saved=True,
            source_sha256=digest(dataset), records=records, fit_weighting=weighting,
            new_ppo_steps=0, deployment_approved=False)
        write(folder / 'RESULTS.json', dict(completed=True, **row))
        result['rows'].append(row)
        budget.tick()
        result['resource'] = budget.statistics()
        write(fits / 'RESULTS.json', result)
        del actor, optimizer, features, feat
    for name, expected in hashes.items():
        assert digest(ROOT / name) == expected, name
    candidates = [row for row in result['rows'] if row['exported_for_physical_validation']]
    result.update(completed=True, at=datetime.now(timezone.utc).isoformat(),
        selected_for_bounded_physical_validation=candidates[0]['model_path'] if candidates else None,
        inputs_verified=len(hashes), resource=budget.statistics())
    write(fits / 'RESULTS.json', result)
    write(fits / 'STATUS.json', dict(status='complete', selected=result['selected_for_bounded_physical_validation']))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    try:
        main(args)
    except Exception as error:
        if args.output.is_dir():
            write(args.output / 'FAILURE.json', dict(error=repr(error), automatically_restarted=False))
        raise
