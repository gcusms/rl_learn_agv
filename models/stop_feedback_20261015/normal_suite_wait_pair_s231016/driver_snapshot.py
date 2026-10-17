"""Retain the whole normal suite and test learner-visited wait continuations.

Both fits share the original4949 labels and complete normal-mission retention
sources. Only the treatment also sees the independently verified continuation
from the learner's actual waiting state. No failed prefix is a positive label.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import shutil
import subprocess
import time

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
DEPENDENCIES = {
    'rl-normal-suite-reference625-v2-20261016.service': 'd988fe03030d446cb93f9e15a04e10c7',
    'rl-learner-wait-continuation-20261016.service': '813e8ea20bee4175ab4a7f022ccbf916'}


def load(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key].copy() for key in archive.files}


def dependency_state(unit):
    process = subprocess.run(['systemctl', '--user', 'show', unit,
        '--property=ActiveState,MainPID,InvocationID,ExecMainStatus'],
        text=True, capture_output=True, check=True)
    return dict(line.split('=', 1) for line in process.stdout.splitlines() if '=' in line)


def main(args):
    protocol, output = args.protocol.resolve(), args.output.resolve()
    protocol.mkdir(parents=True, exist_ok=False)
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    model = ROOT / 'models/stop_feedback_20261015/full_mission_retention_pair_s231016/retained500'
    original_path = BASE / 'full_mission_retention_protocol_20261016/demonstrations.npz'
    replay_path = BASE / 'normal_suite_reference625_replay_v2_20261016/RESULTS.json'
    continuation_path = BASE / 'learner_wait_continuation_20261016/RESULTS.json'
    validation_path = BASE / 'full_mission_retention_validation_20261016/RESULTS.json'
    tasks_path = BASE / 'context_feedback_longfit_protocol_20261016/tasks.json'
    files = [Path(__file__), model / 'checkpoint.pt', model / 'parent_actor.pt', model / 'RESULTS.json',
        model / 'portable/actor.pt', model / 'portable/hardware.json', original_path, validation_path, tasks_path,
        ROOT / 'training/experiments/replay_reference_missions625.py',
        ROOT / 'training/experiments/verify_wait_continuation_source.py']
    for name in ['build_release_recovery_sources', 'guard_reason_release_actor', 'context_release_actor',
            'release_actor', 'long_risk_actor', 'phase_tail_loss', 'refit_phase_tail_pair',
            'release_fit_weights', 'train_long_risk_actor']:
        files.append(ROOT / ('training/experiments/' + name + '.py'))
    for folder in ['neural_navigation', 'navrl']:
        files += list((ROOT / 'training' / folder).rglob('*.py'))
    hashes = {str(p.relative_to(ROOT)): digest(p) for p in set(files)}
    plan = dict(inputs=hashes, dependencies=DEPENDENCIES, maximum_updates_per_fit=args.updates,
        prior_supervised_updates=3500, original_labels=4949,
        fits=['normal_control', 'wait_continuation'],
        only_changed_factor='verified_complete_continuation_from_actual_learner_wait',
        original_labels_byte_exact=True, complete_normal18_development_sources=True,
        choose_current_efficient_successes_before_older_reference_for_same_case=True,
        never_multiply_valid_reference_actions_into_conflicting_labels=True,
        loss='same_stop_ordinary_resumed_mass_and_half_tail_ordinary_loss',
        same_initial_model_adam_rng_architecture_lr_and_updates=True,
        deduplicate_by_complete408_features_not_truncated312=True,
        no_truth_or_future_actor_inputs=True, no_failed_prefix_positive_labels=True,
        no_new_physics=True, new_ppo_steps=0, deployment_approved=False)
    write(protocol / 'PLAN.json', plan)
    write(output / 'PLAN.json', plan)
    (output / 'driver_snapshot.py').write_text(Path(__file__).read_text())
    while True:
        states = {unit: dependency_state(unit) for unit in DEPENDENCIES}
        for unit, state in states.items():
            if state['InvocationID'] not in ('', DEPENDENCIES[unit]):
                raise RuntimeError('Source dependency restarted: ' + unit)
            if state['ActiveState'] == 'failed':
                raise RuntimeError('Source dependency failed: ' + unit)
        if all(s['ActiveState'] == 'inactive' and s['MainPID'] == '0' for s in states.values()):
            if any(s['ExecMainStatus'] != '0' for s in states.values()):
                raise RuntimeError('Source dependency did not finish normally')
            break
        write(output / 'STATUS.json', dict(status='waiting_for_pinned_source_collection', dependencies=states,
            optimizer_started=False, new_ppo_steps=0))
        time.sleep(5.)
    for name, expected in hashes.items():
        assert digest(ROOT / name) == expected, name
    replay = json.loads(replay_path.read_text())
    continuation = json.loads(continuation_path.read_text())
    validation = json.loads(validation_path.read_text())
    tasks = json.loads(tasks_path.read_text())
    assert replay['completed'] and len(replay['rows']) == 10
    assert continuation['completed'] and validation['completed']
    budget = ExperimentBudget(.1, memory_mib=1600, directory=output)
    original = load(original_path)
    assert len(original['choices']) == 4949
    original_fit = json.loads((model / 'RESULTS.json').read_text())
    assert digest(original_path) == original_fit['source_sha256']
    assert digest(model / 'checkpoint.pt') == original_fit['checkpoint_sha256']
    pieces = {key: [value] for key, value in original.items()}
    references = {r['case_id']: r for r in tasks['reference']['normal18']}
    current = {r['case_id']: r for group in ['normal_repaired2', 'normal_other16']
        for r in validation['groups'][group]['rows']}
    teacher_rows = {r['case_id']: r for r in replay['rows']}
    sources = []
    for case in tasks['normal18']:
        row = current.get(case['id'])
        ref = references[case['id']]
        efficient = bool(row and row['safe_success'] and row['protocol_valid']
            and not any(row[k] for k in ['collisions', 'false_arrival', 'physical_guards', 'timeout'])
            and all(row[k] <= ref[k]*1.2+1e-8 for k in ['seconds', 'travel_m']))
        teacher = teacher_rows.get(case['id'])
        assert efficient or teacher is not None, case['id']
        sources.append(dict(case_id=case['id'], kind='current_efficient_native625' if efficient else 'causal439_reference',
            row=row if efficient else ref, trace=row['trace_path'] if efficient else teacher['trace_path'],
            sha256=row['trace_sha256'] if efficient else teacher['trace_sha256'],
            actor=model / 'portable/actor.pt' if efficient else ROOT / teacher['reference_actor_path'],
            observation_dimensions=625 if efficient else 439))
    normal_count = None
    if continuation['source_admitted']:
        assert continuation['safe_success'] and continuation['reference_prefix_exact']
        assert continuation['minimum_post_release_dynamic_gap_m'] >= .04-1e-4
        assert not any(continuation[k] for k in ['collisions', 'false_arrival', 'physical_guards'])
        sources.append(dict(case_id='independent_dynamic_000_actual_wait35', kind='learner_visited_wait_continuation',
            row=None, trace=continuation['trace_path'], sha256=continuation['trace_sha256'],
            actor=model / 'parent_actor.pt', observation_dimensions=439))
    actor = GuardReasonReleaseActor(torch.jit.load(str(model / 'parent_actor.pt'), map_location='cpu')).eval()
    actor.load_state_dict(torch.load(model / 'checkpoint.pt', map_location='cpu', weights_only=False)['actor'])
    lidar = LidarSpec(json.loads((model / 'portable/hardware.json').read_text())['lidar'])
    admission = []
    for offset, source in enumerate(sources):
        path = ROOT / source['trace']
        assert digest(path) == source['sha256']
        trace = load(path)
        x = torch.from_numpy(trace['observations'])
        assert x.shape == (len(trace['choices']), 625) and torch.isfinite(x).all()
        np.testing.assert_array_equal(masks_from_observation(trace['observations'][:, :282]), trace['masks'])
        assert trace['masks'][np.arange(len(x)), trace['choices']].all()
        reference_actor = torch.jit.load(str(source['actor']), map_location='cpu').eval()
        with torch.no_grad():
            np.testing.assert_array_equal(reference_actor(x[:, :source['observation_dimensions']]).numpy(), ACTIONS[trace['choices']])
            support = actor.support(x).numpy()
        for observation, scan in zip(trace['observations'], trace['scans']):
            np.testing.assert_array_equal(augment(observation[:481], scan, lidar), observation)
        row = source['row']
        if row is None:
            row = dict(stop_reasons={'estimated_goal_reached': int((trace['stop_reasons']=='estimated_goal_reached').sum())})
        selected = training_indices(trace, row, support)
        extension = dict(observations=trace['observations'][selected], choices=trace['choices'][selected],
            masks=trace['masks'][selected], sensor_times=trace['labels'][selected, 0],
            source_ids=np.full(len(selected), int(original['source_ids'].max())+1+offset, np.int64))
        for key, value in extension.items():
            pieces[key].append(value)
        admission.append(dict(case_id=source['case_id'], kind=source['kind'], frames=len(selected),
            trace_path=source['trace'], trace_sha256=source['sha256'], complete_safe_source=True,
            excluded_inactive_terminal_frames=1, source_is_not_new_candidate_success=True))
        if offset == 17:
            normal_count = sum(len(values) for values in pieces['choices'])
        budget.tick()
        del reference_actor
    arrays = {key: np.concatenate(value) for key, value in pieces.items()}
    for key, value in original.items():
        assert arrays[key][:4949].tobytes() == value.tobytes(), key
    assert normal_count is not None
    dataset = protocol / 'demonstrations.npz'
    np.savez_compressed(dataset, **arrays)
    write(protocol / 'SOURCE_ADMISSION.json', dict(admitted=True, original_labels=4949,
        normal_suite_labels=normal_count, total_labels=len(arrays['choices']),
        normal18_cases=18, actual_wait_continuation_admitted=bool(continuation['source_admitted']),
        source_rows=admission, original_prefix_byte_exact=True, dataset_sha256=digest(dataset)))
    files = [replay_path, continuation_path, dataset] + [ROOT / s['trace'] for s in sources]
    files += [s['actor'] for s in sources]
    hashes.update({str(p.relative_to(ROOT)): digest(p) for p in files})
    write(output / 'DEPENDENCIES.json', dict(inputs=hashes))
    result = dict(completed=False, rows=[], new_ppo_steps=0, new_physical_episodes=0,
        goal_achieved=False, deployment_approved=False)
    x = torch.from_numpy(arrays['observations'])
    initial = None
    treatments = [('normal_control', normal_count)]
    if continuation['source_admitted']:
        treatments.append(('wait_continuation', len(x)))
    for name, count in treatments:
        folder = output / name
        folder.mkdir()
        actor = GuardReasonReleaseActor(torch.jit.load(str(model / 'parent_actor.pt'), map_location='cpu')).eval()
        saved, optimizer = restore_fit(model / 'checkpoint.pt', actor)
        fingerprints = (state_digest(actor), optimizer_digest(optimizer))
        if initial is None:
            initial = fingerprints
        assert initial == fingerprints
        frozen = state_digest(actor.parent)
        with torch.no_grad():
            features, baseline = actor.features(x)
            active = actor.support(x).numpy()
        nominal = np.abs(baseline.numpy()[:, None]-ACTIONS[None]).max(2).argmin(1)
        assert not ((~active) & (nominal != arrays['choices'])).any()
        selected = active.copy()
        selected[count:] = False
        target = torch.from_numpy(arrays['choices'][selected])
        legal = torch.from_numpy(arrays['masks'][selected])
        waiting = torch.from_numpy(arrays['observations'][selected, 434]*5)
        feat = features[selected].detach()
        weights, weighting = fit_weights(feat.numpy(), target.numpy(), waiting.numpy(), mode='recovery_phase')
        weight = torch.from_numpy(weights).to(feat.dtype)
        np.savez_compressed(folder / 'fixed_source_weights.npz', active=selected, weights=weights)
        records = []
        for update in range(1, args.updates+1):
            logits = (actor.head(feat)+2.*feat[:, 76:88]).masked_fill(~legal, -1e8)
            loss = phase_tail_loss(F.cross_entropy(logits, target, reduction='none'), target, waiting, weight)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.head.parameters(), 1.)
            optimizer.step()
            if update % 10 == 0:
                budget.tick()
            if update % 50 == 0 or update == args.updates:
                with torch.no_grad():
                    correct = (actor(x).numpy() == ACTIONS[arrays['choices']]).all(1)
                record = dict(update=update, loss=float(loss.detach()), original4949_correct=int(correct[:4949].sum()),
                    normal_suite_correct=int(correct[:normal_count].sum()), all_labels_correct=int(correct.sum()))
                records.append(record)
                write(output / 'STATUS.json', dict(status='paired_source_fit', model=name, **record))
        assert state_digest(actor.parent) == frozen
        with torch.no_grad():
            correct = (actor(x).numpy() == ACTIONS[arrays['choices']]).all(1)
        all_fit = bool(correct.all())
        shutil.copy2(model / 'parent_actor.pt', folder / 'parent_actor.pt')
        torch.save(dict(actor=actor.state_dict(), optimizer=optimizer.state_dict(),
            torch_rng=torch.get_rng_state(), numpy_rng=np.random.get_state(), python_rng=random.getstate(),
            parent_actor_sha256=digest(folder / 'parent_actor.pt'), contract=saved['contract'],
            head_contract=saved['head_contract'], learning_support=saved['learning_support'],
            supervised_updates=3500+args.updates, new_supervised_updates=args.updates,
            source_sha256=digest(dataset), new_ppo_steps=0, exact_physical_rollout_resume=False), folder / 'checkpoint.pt')
        if all_fit:
            invalid, near, stop = (arrays['observations'][:16].copy() for _ in range(3))
            invalid[:, 435] = 0.
            near[:, 218:220] = 0.
            stop[:, 235:244] = stop[:, 267:269] = 0.
            export(actor, ROOT / 'models/stop_feedback_20261015/long_risk_bc1000_s231016/portable',
                folder / 'portable', np.r_[arrays['observations'], invalid, near, stop])
        row = dict(model=name, model_path=str(folder.relative_to(ROOT)), updates=3500+args.updates,
            new_supervised_updates=args.updates, cumulative_supervised_updates=3500+args.updates,
            initial_actor_tensor_sha256=initial[0], initial_optimizer_sha256=initial[1],
            own_training_labels=count, own_training_fit_passed=bool(correct[:count].all()),
            correct_source_labels=int(correct.sum()), total_source_labels=len(x),
            original3252_all_retained=bool(correct[:3252].all()), original3728_all_retained=bool(correct[:3728].all()),
            original4949_all_retained=bool(correct[:4949].all()),
            safe_stop_labels_all_retained=bool(correct[arrays['choices']==0].all()),
            wrong_source_indices=np.flatnonzero(~correct).tolist(), records=records, weighting=weighting,
            source_sha256=digest(dataset), checkpoint_sha256=digest(folder / 'checkpoint.pt'),
            optimizer_and_rng_saved=True, exported_for_physical_validation=all_fit,
            new_ppo_steps=0, deployment_approved=False)
        write(folder / 'RESULTS.json', dict(completed=True, **row))
        result['rows'].append(row)
        result['resource'] = budget.statistics()
        write(output / 'RESULTS.json', result)
        del actor, optimizer, features, feat
    for name, expected in hashes.items():
        assert digest(ROOT / name) == expected, name
    candidates = [r for r in reversed(result['rows']) if r['exported_for_physical_validation']]
    budget.tick()
    result.update(completed=True, at=datetime.now(timezone.utc).isoformat(),
        selected_for_bounded_physical_validation=candidates[0]['model_path'] if candidates else None,
        input_hashes_verified=len(hashes), resource=budget.statistics())
    write(output / 'RESULTS.json', result)
    write(output / 'STATUS.json', dict(status='complete', selected=result['selected_for_bounded_physical_validation']))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--updates', type=int, default=1000)
    args = parser.parse_args()
    if not 1 <= args.updates <= 1000:
        parser.error('At most1000 additional updates per fit')
    try:
        main(args)
    except Exception as error:
        if args.output.is_dir():
            write(args.output / 'FAILURE.json', dict(error=repr(error), automatically_restarted=False))
        raise
