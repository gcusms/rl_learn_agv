"""Fit only independently admitted complete-safe remaining yield sources.

Existing 2,638 observations, masks and labels remain byte exact. The teacher
never participates in candidate validation, and failed sources never train.
"""
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
BASE = ROOT / 'reports/stop_feedback_20261015'
PARENT = ROOT / 'models/stop_feedback_20261015/early_yield_bc400_s231016'
TARGET = ROOT / 'models/stop_feedback_20261015/early_yield_remaining_bc600_s231016'
BUNDLE = ROOT / 'models/stop_feedback_20261015/baseline/portable'
DEPENDENCY = 'rl-early-yield-condition05-20261016.service'
SOURCES = [BASE / ('early_yield_condition' + key + '_20261016') for key in ('02', '05')]


def stamp():
    return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(name, value):
    path = OUT / name
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    temp.replace(path)


def state():
    raw = subprocess.check_output(['systemctl', '--user', 'show', DEPENDENCY,
        '-p', 'MainPID', '-p', 'ActiveState', '-p', 'Result', '-p', 'InvocationID'], text=True)
    return dict(line.split('=', 1) for line in raw.splitlines())


def main():
    if TARGET.exists() or (OUT / 'PLAN.json').exists():
        raise FileExistsError('Use a fresh bounded experiment; no restart')
    predecessor = state()
    paths = [Path(__file__), PARENT / 'policy_bc.zip', PARENT / 'demonstrations.npz',
        PARENT / 'continuation_state.pt', PARENT / 'RESULTS.json',
        BASE / 'early_yield_remaining_protocol_20261016/PLAN.json',
        BASE / 'early_yield_remaining_protocol_20261016/heldout_cases.json',
        BASE / 'early_yield_remaining_audit_20261016/RESULTS.json',
        BASE / 'actual106_memory_context_20261016/replay.py']
    for source in SOURCES:
        paths += [source / name for name in ('run.py', 'PLAN.json', 'case.json', 'expert_config.json')]
    manifest = json.loads((BUNDLE / 'manifest.json').read_text())
    paths += [BUNDLE / 'manifest.json', *[BUNDLE / name for name in manifest['files']]]
    for folder in ('neural_navigation', 'navrl', 'experiments'):
        paths += list((ROOT / 'training' / folder).rglob('*.py'))
    inputs = {str(path.relative_to(ROOT)): digest(path) for path in set(paths)}
    write('PLAN.json', dict(at=stamp(), inputs=inputs, predecessor=DEPENDENCY,
        predecessor_invocation=predecessor['InvocationID'], maximum_supervised_updates=600,
        seed=231016, parent_checkpoint_sha256=digest(PARENT / 'policy_bc.zip'),
        required_old_examples=2638, required_old_stop_examples=95,
        each_source_requires_complete_safe_arrival_and_sensor_yield_and_model_return=True,
        minimum_post_yield_sampled_clearance_m=.04,
        minimum_supported_model_motion_frames_after_release=10,
        failed_or_partial_sources_excluded=True, heldout_four_sensor_conditions_excluded=True,
        exact_old_input_mask_label_retention=True,
        unchanged_base_localizer_router_docking_body_actions_masks_memory_and_guard=True,
        changed_factor='Additional admitted complete-safe early-yield data, with a fixed600 maximum BC budget',
        new_ppo_steps=0, goal_achieved=False, deployment_approved=False))
    while True:
        current = state()
        if current['InvocationID'] not in ('', predecessor['InvocationID']):
            raise RuntimeError('Physical predecessor restarted')
        if current['ActiveState'] in ('inactive', 'failed'):
            if current['ActiveState'] != 'inactive' or current['Result'] != 'success' or current['MainPID'] != '0':
                raise RuntimeError('Physical source pipeline failed')
            break
        write('STATUS.json', dict(at=stamp(), status='waiting_for_complete_sources',
            optimization_started=False, predecessor=current))
        time.sleep(5)

    def verify():
        for name, expected in inputs.items():
            assert digest(ROOT / name) == expected, name

    verify()
    import importlib.util
    import numpy as np
    import torch
    from experiments.experiment_provenance import record_provenance
    from experiments.recovery_demonstrations import convert_observations, initialize_from_demonstrations
    from experiments.stop_recovery_policy import StopRecoveryPPO, export_recovery
    from experiments.train_prefix_recovery import frozen_base_digest
    from neural_navigation.experiment_budget import ExperimentBudget
    from neural_navigation.masked_actions import COMMANDS, masks_from_observation
    torch.set_num_threads(1)
    torch.manual_seed(231016)
    np.random.seed(231016)
    budget = ExperimentBudget(.35, memory_mib=3072, directory=OUT)
    model = StopRecoveryPPO.load(PARENT / 'policy_bc.zip', device='cpu')
    model.policy.set_training_mode(False)
    frozen = frozen_base_digest(model.policy)
    assert frozen == 'e5502406c4ee8e42c4da09dd6affa022cb76e7f9b01316b1eaf0d2551431938e'
    assert model.num_timesteps == 0
    continuation = torch.load(PARENT / 'continuation_state.pt', map_location='cpu', weights_only=False)
    model.policy.optimizer.load_state_dict(continuation['optimizer'])
    with np.load(PARENT / 'demonstrations.npz', allow_pickle=False) as archive:
        old = {key: archive[key].copy() for key in ('observations', 'masks', 'choices')}
    assert old['observations'].shape == (2638, 436) and int((old['choices'] == 0).sum()) == 95
    with torch.no_grad():
        previous = model.policy.get_distribution(torch.from_numpy(old['observations']), old['masks']).distribution.probs.argmax(-1).numpy()
    np.testing.assert_array_equal(previous, old['choices'])
    spec = importlib.util.spec_from_file_location('remaining_yield_geometry', BASE / 'actual106_memory_context_20261016/replay.py')
    geometry = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(geometry)
    arrays = []
    admissions = []
    half = np.asarray(json.loads((BUNDLE / 'hardware.json').read_text())['robot_footprint_m']) / 2
    for source in SOURCES:
        report_path = source / 'RESULTS.json'
        report = json.loads(report_path.read_text())
        assert report['completed'] and len(report['rows']) == 1
        inputs[str(report_path.relative_to(ROOT))] = digest(report_path)
        row = report['rows'][0]
        admission = dict(source=str(source.relative_to(ROOT)), case_id=row['case_id'],
            source_report_sha256=digest(report_path), admitted=False,
            expert_success_is_not_learned_success=True)
        admissions.append(admission)
        if (not report.get('full_safe_expert_trajectory_exists') or not row['safe_success']
                or not row['physical_arrival'] or not row['protocol_valid']
                or any(row[key] for key in ('false_arrival', 'collisions', 'physical_guards', 'timeout'))):
            admission['reason'] = 'No complete safe sensor-yield source'
            continue
        path = ROOT / row['trace_path']
        assert digest(path) == row['trace_sha256']
        inputs[str(path.relative_to(ROOT))] = digest(path)
        with np.load(path, allow_pickle=False) as archive:
            trace = {key: archive[key].copy() for key in archive.files}
        starts = [e['time_s'] for e in row['events'] if e['type'] == 'begin_yield']
        releases = [e['time_s'] for e in row['events'] if e['type'] == 'release_to_model']
        if not starts or not releases:
            admission['reason'] = 'No confirmed return to the model after yield'
            continue
        times = trace['labels'][:, 0]
        begin, release = min(starts), min(releases)
        x = convert_observations(trace['observations'], trace['masks'], trace['commands'], times)
        np.testing.assert_array_equal(masks_from_observation(x[:, :282]), trace['masks'])
        assert trace['masks'][np.arange(len(times)), trace['choices']].all()
        away = np.linalg.norm(x[:, 218:220] * 12., axis=1) > .3
        np.testing.assert_allclose(trace['commands'][away], COMMANDS[trace['choices'][away]], rtol=0, atol=2e-8)
        case = json.loads((source / 'case.json').read_text())
        scene = json.loads((ROOT / 'training' / case['scene'] / 'physical_scene.json').read_text())
        walls = np.asarray(scene['walls'], float)
        radii = np.asarray([human['radius_m'] for human in scene['dynamic_obstacles']])
        minimum = float(min(geometry.physical_gap(trace['labels'][i, 1:4], half, walls,
            trace['human_positions'][i], radii) for i in np.flatnonzero(times >= begin - 1e-8)))
        admission.update(minimum_post_yield_sampled_gap_m=minimum,
            sampled_geometry_is_not_continuous_hardware_safety=True)
        if minimum < .04:
            admission['reason'] = 'Post-yield sampled geometry below unchanged4cm margin'
            continue
        with torch.no_grad():
            supported = model.policy.recovery_support(torch.from_numpy(x)).numpy()
        keep = supported & (times >= begin - 1e-8)
        post = supported & trace['model_controlled'] & (times >= release - 1e-8) & (trace['choices'] != 0)
        if int(post.sum()) < 10 or not np.any(trace['choices'][keep] == 0) or not np.any(trace['choices'][keep] != 0):
            admission['reason'] = 'Insufficient supported waiting or model resumption'
            continue
        arrays.append(dict(observations=x[keep], masks=trace['masks'][keep], choices=trace['choices'][keep]))
        admission.update(admitted=True, reason='Complete safe sensor-yield and model-resumption trajectory',
            first_yield_s=begin, first_release_s=release, trace_path=row['trace_path'],
            trace_sha256=row['trace_sha256'], new_examples=int(keep.sum()),
            stop_targets=int((trace['choices'][keep] == 0).sum()),
            movement_targets=int((trace['choices'][keep] != 0).sum()),
            supported_model_motion_after_release=int(post.sum()),
            no_truth_or_future_in_model_or_teacher_inputs=True)
        budget.tick()
    write('SOURCE_ADMISSION.json', dict(at=stamp(), sources=admissions,
        admitted_sources=sum(row['admitted'] for row in admissions),
        no_failed_or_partial_positive_sources=True, source_file_hashes=inputs))
    if not arrays:
        verify()
        write('RESULTS.json', dict(completed=True, skipped=True, reason='No independently admitted source; no optimizer started',
            sources=admissions, additional_supervised_updates=0, new_ppo_steps=0,
            exported_for_bounded_evaluation=False, goal_achieved=False, deployment_approved=False))
        write('STATUS.json', dict(at=stamp(), status='complete_without_fitting', optimization_started=False))
        return
    data = {key: np.concatenate([old[key], *[array[key] for array in arrays]]) for key in old}
    for key in old:
        np.testing.assert_array_equal(data[key][:2638], old[key])
    seen = {}
    for i, (features, mask, label) in enumerate(zip(data['observations'], data['masks'], data['choices'])):
        key = hashlib.sha256(features.tobytes() + mask.tobytes()).digest()
        before = seen.setdefault(key, (int(label), i))
        if before[0] != int(label):
            raise ValueError('Conflicting exact labels: ' + repr((before[1], i)))
    TARGET.mkdir()
    np.savez_compressed(TARGET / 'demonstrations.npz', **data)
    (TARGET / 'config.json').write_text(json.dumps(dict(seed=231016, maximum_updates=600,
        source_admissions=admissions, old_examples=2638, old_stop_examples=95), indent=2) + '\n')
    record_provenance(TARGET, BUNDLE, [Path(__file__)])
    write('STATUS.json', dict(at=stamp(), status='bounded_supervised_fit',
        examples=len(data['choices']), optimization_started=True, new_ppo_steps=0))
    fitted = initialize_from_demonstrations(model, data['observations'], data['masks'], data['choices'], budget, maximum_updates=600)
    model.policy.set_training_mode(False)
    with torch.no_grad():
        predicted = model.policy.get_distribution(torch.from_numpy(data['observations']), data['masks']).distribution.probs.argmax(-1).numpy()
    old_correct = int((predicted[:2638] == old['choices']).sum())
    old_stops = int((predicted[:2638][old['choices'] == 0] == 0).sum())
    all_correct = int((predicted == data['choices']).sum())
    assert frozen_base_digest(model.policy) == frozen and model.num_timesteps == 0
    accepted = old_correct == 2638 and old_stops == 95 and all_correct == len(predicted)
    model.demonstration_initialization = dict(fitted, parent_checkpoint_sha256=digest(PARENT / 'policy_bc.zip'),
        sources=admissions)
    model.save(TARGET / 'policy_bc.zip')
    torch.save(dict(optimizer=model.policy.optimizer.state_dict(), torch_rng=torch.get_rng_state(),
        numpy_rng=np.random.get_state()), TARGET / 'continuation_state.pt')
    if accepted:
        export_recovery(model, BUNDLE, TARGET / 'bc_portable')
        parent_manifest = json.loads((PARENT / 'bc_portable/manifest.json').read_text())
        child_manifest = json.loads((TARGET / 'bc_portable/manifest.json').read_text())
        for name, expected in parent_manifest['files'].items():
            if name != 'actor.pt':
                assert child_manifest['files'][name] == expected, name
    verify()
    result = dict(completed=True, at=stamp(), sources=admissions, fitting=fitted,
        old_examples_correct=old_correct, old_stop_labels_preserved=old_stops,
        total_examples=len(predicted), all_examples_correct=all_correct,
        additional_supervised_updates=fitted['updates'], new_ppo_steps=0,
        checkpoint_sha256=digest(TARGET / 'policy_bc.zip'), frozen_base_sha256=frozen,
        exported_for_bounded_evaluation=accepted, resource=budget.statistics(),
        goal_achieved=False, deployment_approved=False)
    (TARGET / 'RESULTS.json').write_text(json.dumps(result, indent=2) + '\n')
    write('RESULTS.json', result)
    write('STATUS.json', dict(at=stamp(), status='complete', exported_for_bounded_evaluation=accepted))


if __name__ == '__main__':
    os.chdir(ROOT)
    try:
        main()
    except Exception as error:
        write('FAILURE.json', dict(at=stamp(), error=repr(error), automatically_restarted=False))
        raise
