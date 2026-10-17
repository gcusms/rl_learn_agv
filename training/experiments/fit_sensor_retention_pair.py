"""Retain complete-safe noise histories in a paired, bounded release-head fit.

Both fits start from identical model, Adam and RNG state. The treatment adds
two causally reproduced native625D trajectories; all prior3728 arrays remain
byte-exact. Runtime permissions, support, features and loss rule do not change.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import shutil
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

from experiments.build_release_recovery_sources import training_indices
from experiments.guard_reason_release_actor import GuardReasonReleaseActor, export
from experiments.long_risk_actor import digest
from experiments.phase_tail_loss import phase_tail_loss
from experiments.refit_phase_tail_pair import optimizer_digest, restore_fit, write
from experiments.release_fit_weights import fit_weights
from experiments.replay_normal_retention_sources import main as replay_sources
from experiments.train_long_risk_actor import state_digest
from neural_navigation.experiment_budget import ExperimentBudget
from neural_navigation.masked_actions import ACTIONS, masks_from_observation
from neural_navigation.release_observation import augment
from neural_navigation.sensors import LidarSpec


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'reports/stop_feedback_20261015'


def admitted_row(row):
    """Require full, autonomous, safe physical success before reusing labels."""
    if (not all(row.get(k) for k in ('safe_success', 'physical_arrival',
            'protocol_valid', 'learned_model_only', 'no_teacher'))
            or any(row.get(k, 1) for k in ('collisions', 'timeout',
                'false_arrival', 'physical_guards'))
            or row.get('recovery_test') or row.get('not_a_full_mission_score')):
        raise ValueError('Only complete-safe autonomous physical sources are eligible')


def load_arrays(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key].copy() for key in archive.files}


def main(args):
    protocol, fits = args.protocol.resolve(), args.output.resolve()
    protocol.mkdir(parents=True, exist_ok=False)
    fits.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    model = ROOT / 'models/stop_feedback_20261015/normal_retention_refit2_s231016/tail500'
    parent_source = BASE / 'normal_retention_sources_20261016'
    teacher = ROOT / 'models/stop_feedback_20261015/sensor_release_retained400_s231016/portable'
    task_path = BASE / 'context_feedback_longfit_protocol_20261016/tasks.json'
    report_path = BASE / 'retained_sensor_release_validation_20261016/RESULTS.json'
    source_report = json.loads(report_path.read_text())
    tasks = json.loads(task_path.read_text())
    parent_result = json.loads((parent_source / 'RESULTS.json').read_text())
    model_result = json.loads((model / 'RESULTS.json').read_text())
    assert parent_result['completed'] and parent_result['admitted']
    assert parent_result['frames'] == model_result['total_source_labels'] == 3728
    assert model_result['correct_source_labels'] == 3728
    assert digest(model / 'checkpoint.pt') == model_result['checkpoint_sha256']
    assert digest(parent_source / 'demonstrations.npz') == parent_result['dataset_sha256']
    assert source_report['completed']
    ids = ['sensor_release_frozen_noise_05_0', 'sensor_release_frozen_noise_05_1']
    specifications = []
    for case_id in ids:
        rows = [r for r in source_report['rows']
            if r['case_id'] == case_id and r['variant'] == 'candidate']
        assert len(rows) == 1
        row = rows[0]
        admitted_row(row)
        specifications.append(dict(id=case_id, trace=row['trace_path'],
            case=next(c for c in tasks['sensor5'] if c['id'] == case_id),
            physical_source_report=str(report_path.relative_to(ROOT)),
            physical_source_rows_key='rows', physical_source_row=row,
            reference_actor=str((teacher / 'actor.pt').relative_to(ROOT)),
            reference_observation_dimensions=625))
    # These sources are now training regressions. Freeze fresh noise tasks
    # before fitting; they remain separate from a new-map generalization test.
    fresh = []
    for index, seed in enumerate((261017121, 261017122)):
        case = dict(specifications[0]['case'], id=f'sensor_retention_fresh_{index}',
            seed=seed, pilot_category='noise_frozen_before_sensor_retention_fit_same_map')
        fresh.append(case)
    files = [Path(__file__), model / 'checkpoint.pt', model / 'RESULTS.json',
        model / 'parent_actor.pt', parent_source / 'RESULTS.json',
        parent_source / 'demonstrations.npz', report_path, task_path,
        teacher / 'actor.pt', teacher / 'hardware.json']
    files += [ROOT / spec['trace'] for spec in specifications]
    for name in ('replay_normal_retention_sources', 'guard_reason_release_actor',
            'context_release_actor', 'release_actor', 'long_risk_actor',
            'refit_phase_tail_pair', 'phase_tail_loss', 'release_fit_weights',
            'build_release_recovery_sources'):
        files.append(ROOT / f'training/experiments/{name}.py')
    hashes = {str(p.relative_to(ROOT)): digest(p) for p in files}
    plan = dict(at=datetime.now(timezone.utc).isoformat(), inputs=hashes,
        parent_labels=3728, maximum_added_labels=339, selected_complete_sources=ids,
        maximum_new_updates_per_fit=500, prior_supervised_updates=2500,
        fits=['control500', 'retained500'], seed=231016,
        only_changed_factor='add339complete_safe_native625_noise_labels',
        same_initial_actor_adam_rng_loss_architecture_and_learning_rate=True,
        phase_gradient_budgets=dict(stop=.5, ordinary=.25, resumed=.25),
        ordinary_tail_focus_share=.5, duplicate_weight_features=312,
        all_original_labels_and_safe_stop_labels_required=True,
        no_partial_or_failed_physics_trace_labels=True,
        fresh_noise_cases=fresh, new_ppo_steps=0, deployment_approved=False)
    write(protocol / 'PLAN.json', plan)
    write(protocol / 'specifications.json', specifications)
    write(protocol / 'fresh_cases.json', fresh)
    write(fits / 'PLAN.json', plan)
    shutil.copy2(__file__, fits / 'driver_snapshot.py')
    write(fits / 'STATUS.json', dict(status='causal_native625_replay'))
    replay = protocol / 'replay625'
    replay_sources(SimpleNamespace(output=replay, bundle=model / 'portable',
        specifications=protocol / 'specifications.json'))
    reproduced = json.loads((replay / 'RESULTS.json').read_text())
    if not reproduced['completed'] or len(reproduced['rows']) != 2:
        write(fits / 'RESULTS.json', dict(completed=True,
            reason='Native625 replay failed; no labels or fitting',
            selected_for_bounded_physical_validation=None, new_ppo_steps=0,
            deployment_approved=False))
        return
    budget = ExperimentBudget(.1, memory_mib=1536, directory=fits)
    actor = GuardReasonReleaseActor(torch.jit.load(str(model / 'parent_actor.pt'), map_location='cpu')).eval()
    actor.load_state_dict(torch.load(model / 'checkpoint.pt', map_location='cpu', weights_only=False)['actor'])
    original = load_arrays(parent_source / 'demonstrations.npz')
    pieces = {k: [v] for k, v in original.items()}
    source_rows = []
    next_id = int(original['source_ids'].max()) + 1
    lidar = LidarSpec(json.loads((teacher / 'hardware.json').read_text())['lidar'])
    for offset, row in enumerate(reproduced['rows']):
        assert row['complete_safe_source_admitted'] and row['reference_observation_dimensions'] == 625
        assert row['maximum_reference_observation_absolute_difference'] <= 2e-6
        assert row['original_masks_actions_and_execution_reproduced']
        admitted_row(row['physical_source_row'])
        path = ROOT / row['trace_path']
        assert digest(path) == row['trace_sha256']
        trace = load_arrays(path)
        observations, masks, labels = [trace[k] for k in ('observations', 'masks', 'choices')]
        assert observations.shape == (len(labels), 625) and np.isfinite(observations).all()
        np.testing.assert_array_equal(masks_from_observation(observations[:, :282]), masks)
        assert masks[:, 0].all() and masks[np.arange(len(labels)), labels].all()
        for observation, scan in zip(observations, trace['scans']):
            np.testing.assert_array_equal(augment(observation[:481], scan, lidar), observation)
        with torch.no_grad():
            active = actor.support(torch.from_numpy(observations)).numpy()
        selected = training_indices(trace, row['physical_source_row'], active)
        extension = dict(observations=observations[selected], masks=masks[selected], choices=labels[selected],
            sensor_times=trace['labels'][selected, 0], source_ids=np.full(len(selected), next_id + offset, np.int64))
        for key, value in extension.items():
            pieces[key].append(value)
        source_rows.append(dict(case_id=row['case_id'], frames=len(selected),
            full_episode_frames=len(labels), excluded_inactive_terminal_frame=len(labels)-1,
            complete_safe_native625_replay_verified=True, source_is_reused_not_new_success=True,
            trace_path=row['trace_path'], trace_sha256=row['trace_sha256']))
        budget.tick()
    arrays = {k: np.concatenate(v) for k, v in pieces.items()}
    for key, value in original.items():
        assert arrays[key][:len(value)].tobytes() == value.tobytes(), key
    assert len(arrays['choices']) == 4067
    dataset = protocol / 'demonstrations.npz'
    np.savez_compressed(dataset, **arrays)
    write(protocol / 'SOURCE_ADMISSION.json', dict(admitted=True,
        parent_frames=3728, added_frames=339, total_frames=4067,
        original_prefix_byte_exact=True, source_rows=source_rows,
        native625_full_replay_required=True, dataset_sha256=digest(dataset),
        excluded_failed_candidate_trace=True, all_nonterminal_frames_kept=True))
    result = dict(completed=False, rows=[], new_ppo_steps=0,
        goal_achieved=False, deployment_approved=False)
    x = torch.from_numpy(arrays['observations'])
    initial_actor = initial_optimizer = None
    for name, count in [('control500', 3728), ('retained500', 4067)]:
        folder = fits / name
        folder.mkdir()
        actor = GuardReasonReleaseActor(torch.jit.load(str(model / 'parent_actor.pt'), map_location='cpu')).eval()
        saved, optimizer = restore_fit(model / 'checkpoint.pt', actor)
        current_actor, current_optimizer = state_digest(actor), optimizer_digest(optimizer)
        if initial_actor is None:
            initial_actor, initial_optimizer = current_actor, current_optimizer
        assert (initial_actor, initial_optimizer) == (current_actor, current_optimizer)
        frozen = state_digest(actor.parent)
        with torch.no_grad():
            features, baseline = actor.features(x)
            active = actor.support(x).numpy()
        nominal = np.abs(baseline.numpy()[:, None] - ACTIONS[None]).max(2).argmin(1)
        assert not ((~active) & (nominal != arrays['choices'])).any()
        selected = active.copy(); selected[count:] = False
        w, weighting = fit_weights(features.numpy()[selected, :312], arrays['choices'][selected],
            arrays['observations'][selected, 434] * 5, mode='recovery_phase')
        np.savez_compressed(folder / 'fixed_source_weights.npz', active=selected, weights=w)
        feat = features[selected].detach()
        target = torch.from_numpy(arrays['choices'][selected])
        legal = torch.from_numpy(arrays['masks'][selected])
        waiting = torch.from_numpy(arrays['observations'][selected, 434] * 5)
        weights = torch.from_numpy(w).to(feat.dtype)
        records = []
        for update in range(1, 501):
            logits = (actor.head(feat) + 2. * feat[:, 76:88]).masked_fill(~legal, -1e8)
            loss = phase_tail_loss(F.cross_entropy(logits, target, reduction='none'),
                target, waiting, weights, focus_share=.5)
            optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.head.parameters(), 1.)
            optimizer.step()
            if update % 10 == 0:
                budget.tick()
            if update % 25 == 0:
                with torch.no_grad():
                    correct = (actor(x).numpy() == ACTIONS[arrays['choices']]).all(1)
                records.append(dict(new_update=update, loss=float(loss.detach()),
                    original3728_correct=int(correct[:3728].sum()),
                    added339_correct=int(correct[3728:].sum()), all4067_correct=int(correct.sum())))
                write(fits / 'STATUS.json', dict(status='paired_sensor_retention_fit', model=name, **records[-1]))
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
            supervised_updates=3000, new_supervised_updates=500,
            source_sha256=digest(dataset),
            training_source_sha256=digest(dataset if count == 4067 else parent_source / 'demonstrations.npz'),
            new_ppo_steps=0, exact_physical_rollout_resume=False), folder / 'checkpoint.pt')
        if all_fit:
            invalid, near, stopped = (arrays['observations'][:16].copy() for _ in range(3))
            invalid[:, 435] = 0.; near[:, 218:220] = 0.; near[:, 302] = 1.
            stopped[:, 235:244] = stopped[:, 267:269] = 0.
            export(actor, ROOT / 'models/stop_feedback_20261015/long_risk_bc1000_s231016/portable',
                folder / 'portable', np.r_[arrays['observations'], invalid, near, stopped])
        row = dict(model=name, model_path=str(folder.relative_to(ROOT)),
            updates=3000, new_supervised_updates=500, cumulative_supervised_updates=3000,
            initial_actor_tensor_sha256=initial_actor, initial_optimizer_sha256=initial_optimizer,
            own_training_labels=count, own_training_fit_passed=own_fit,
            correct_source_labels=int(correct.sum()), total_source_labels=4067,
            correct_original3728_labels=int(correct[:3728].sum()),
            original3252_all_retained=bool(correct[:3252].all()),
            original3728_all_retained=bool(correct[:3728].all()),
            safe_stop_labels_all_retained=bool(correct[arrays['choices'] == 0].all()),
            wrong_source_indices=np.flatnonzero(~correct).tolist(),
            exported_for_physical_validation=all_fit, checkpoint_sha256=digest(folder / 'checkpoint.pt'),
            frozen_parent_tensor_sha256=frozen, optimizer_and_rng_saved=True,
            source_sha256=digest(dataset),
            training_source_sha256=digest(dataset if count == 4067 else parent_source / 'demonstrations.npz'),
            records=records, fit_weighting=weighting,
            new_ppo_steps=0, deployment_approved=False)
        write(folder / 'RESULTS.json', dict(completed=True, **row))
        result['rows'].append(row)
        write(fits / 'RESULTS.json', result)
        del actor, optimizer, features, feat
        budget.tick()
    for path, expected in hashes.items():
        assert digest(ROOT / path) == expected, path
    candidates = [r for r in result['rows'] if r['exported_for_physical_validation']]
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
