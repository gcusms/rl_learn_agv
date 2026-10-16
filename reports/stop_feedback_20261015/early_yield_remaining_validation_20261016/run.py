"""Up to16 serial learned-only and paired sensor-condition checks.

No teacher, motion insertion, model update, guard change or future simulator
state enters candidate control. Preserve all declared failures and gates.
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
MODEL = ROOT / 'models/stop_feedback_20261015/early_yield_remaining_bc600_s231016'
PARENT = ROOT / 'models/stop_feedback_20261015/early_yield_bc400_s231016'
FIT = BASE / 'early_yield_remaining_fit_20261016'
PROTOCOL = BASE / 'early_yield_remaining_protocol_20261016'
DEPENDENCY = 'rl-early-yield-remaining-fit-20261016.service'


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
    if (OUT / 'PLAN.json').exists():
        raise FileExistsError('Use a new bounded validation, never restart')
    predecessor = state()
    paths = [Path(__file__), FIT / 'run.py', FIT / 'PLAN.json', PROTOCOL / 'PLAN.json',
        PROTOCOL / 'heldout_cases.json',
        BASE / 'proactive_augmented_protocol_20261016/fresh_cases.json',
        BASE / 'w11_recovery_bc800_navigation_20261016/cases.json',
        BASE / 'actual95_learned_validation_20261016/prefixes.json',
        BASE / 'w11_recovery_bc800_w11_20261016/specifications.json']
    for folder in ('neural_navigation', 'navrl', 'experiments'):
        paths += list((ROOT / 'training' / folder).rglob('*.py'))
    inputs = {str(path.relative_to(ROOT)): digest(path) for path in set(paths)}
    write('PLAN.json', dict(at=stamp(), inputs=inputs, predecessor=DEPENDENCY,
        predecessor_invocation=predecessor['InvocationID'], maximum_new_physical_episodes=16,
        ordered_groups=['original_collision_conditions4', 'paired_sensor_seed_holdouts4x2',
            'retained_fresh33', 'normal_w11', 'retained_actual95', 'retained_actual90'],
        original_four_conditions_must_all_be_safe_before_expansion=True,
        retained_condition04_and_safe_teacher_time_and_travel_factor=1.2,
        heldout_four_seeds_excluded_from_training=True,
        heldout_candidate_requires4_safe_and_paired_safety_efficiency_nonregression=True,
        do_not_compare_success_time_to_collision_time=True,
        no_teacher_or_controller_patch_in_learned_control=True,
        unchanged_body_margin_actions_masks_blind_memory_guard_and_deadlines=True,
        development_maps_are_not_unseen_maps=True,
        full18_extra12_heading_wait_regressions_required_after_this=True,
        new_ppo_steps=0, new_optimizer_updates=0, goal_achieved=False, deployment_approved=False))
    while True:
        current = state()
        if current['InvocationID'] not in ('', predecessor['InvocationID']):
            raise RuntimeError('Fit predecessor restarted')
        if current['ActiveState'] in ('inactive', 'failed'):
            if current['ActiveState'] != 'inactive' or current['Result'] != 'success' or current['MainPID'] != '0':
                raise RuntimeError('Source-gated fitting failed')
            break
        write('STATUS.json', dict(at=stamp(), status='waiting_for_admitted_fit',
            physical_simulation_started=False, predecessor=current))
        time.sleep(5)
    fit_path = FIT / 'RESULTS.json'
    fit = json.loads(fit_path.read_text())
    assert fit['completed']
    if not fit['exported_for_bounded_evaluation']:
        write('RESULTS.json', dict(completed=True, skipped=True,
            reason='No admitted exactly-retained candidate; no physics started',
            new_physical_steps=0, fitting_sha256=digest(fit_path),
            goal_achieved=False, deployment_approved=False))
        write('STATUS.json', dict(at=stamp(), status='complete_without_physics'))
        return
    import numpy as np
    import torch
    from experiments.evaluate_docking_navigation import evaluate
    from experiments.evaluate_feedback_recovery import evaluate_recovery
    from experiments.experiment_provenance import record_provenance
    from experiments.paired_navigation_gate import compare
    from experiments.stop_recovery_policy import StopRecoveryPPO
    from experiments.train_prefix_recovery import frozen_base_digest
    from neural_navigation.experiment_budget import ExperimentBudget
    from neural_navigation.masked_actions import ACTIONS, masks_from_observation
    torch.set_num_threads(1)
    bundle = MODEL / 'bc_portable'
    manifest = json.loads((bundle / 'manifest.json').read_text())
    assert manifest['observation_contract']['dimensions'] == 439
    known_path = BASE / 'recovery_followup_validation_20261016/RESULTS.json'
    broad_path = BASE / 'recovery_followup_broad_regression_20261016/RESULTS.json'
    prior_path = BASE / 'early_yield_validation_20261016/RESULTS.json'
    known = json.loads(known_path.read_text())
    broad = json.loads(broad_path.read_text())
    prior = json.loads(prior_path.read_text())
    assert known['completed'] and broad['completed'] and prior['completed']
    final_paths = [fit_path, known_path, broad_path, prior_path, MODEL / 'policy_bc.zip',
        MODEL / 'demonstrations.npz', MODEL / 'continuation_state.pt', bundle / 'manifest.json',
        *[bundle / name for name in manifest['files']], PARENT / 'bc_portable/manifest.json']
    parent_manifest = json.loads((PARENT / 'bc_portable/manifest.json').read_text())
    final_paths += [PARENT / 'bc_portable' / name for name in parent_manifest['files']]
    teacher_rows = []
    for folder in ('early_yield_physical_20261016', 'early_yield_condition02_20261016', 'early_yield_condition05_20261016'):
        path = BASE / folder / 'RESULTS.json'
        data = json.loads(path.read_text())
        assert data['completed']
        final_paths.append(path)
        teacher_rows.extend(row for row in data['rows'] if row['safe_success'])
    inputs.update({str(path.relative_to(ROOT)): digest(path) for path in final_paths})

    def verify():
        for name, expected in inputs.items():
            assert digest(ROOT / name) == expected, name

    verify()
    assert digest(MODEL / 'policy_bc.zip') == fit['checkpoint_sha256']
    budget = ExperimentBudget(.35, memory_mib=3072, directory=OUT)
    model = StopRecoveryPPO.load(MODEL / 'policy_bc.zip', device='cpu')
    model.policy.set_training_mode(False)
    assert model.num_timesteps == 0 and frozen_base_digest(model.policy) == fit['frozen_base_sha256']
    continuation = torch.load(MODEL / 'continuation_state.pt', map_location='cpu', weights_only=False)
    model.policy.optimizer.load_state_dict(continuation['optimizer'])
    assert continuation['torch_rng'].dtype == torch.uint8 and continuation['numpy_rng'][0] == 'MT19937'
    with np.load(MODEL / 'demonstrations.npz', allow_pickle=False) as archive:
        data = {key: archive[key].copy() for key in ('observations', 'masks', 'choices')}
    with np.load(PARENT / 'demonstrations.npz', allow_pickle=False) as archive:
        for key in data:
            np.testing.assert_array_equal(data[key][:2638], archive[key])
    assert data['observations'].shape == (len(data['choices']), 436)
    assert int((data['choices'][:2638] == 0).sum()) == 95
    np.testing.assert_array_equal(masks_from_observation(data['observations'][:, :282]), data['masks'])
    assert data['masks'][:, 0].all() and data['masks'][np.arange(len(data['choices'])), data['choices']].all()
    with torch.no_grad():
        predicted = model.policy.get_distribution(torch.from_numpy(data['observations']), data['masks']).distribution.probs.argmax(-1).numpy()
    np.testing.assert_array_equal(predicted, data['choices'])
    budget.tick()
    actor = torch.jit.load(str(bundle / 'actor.pt'), map_location='cpu').eval()
    selected = np.unique(np.r_[np.flatnonzero(data['choices'] == 0),
        np.arange(2638, len(predicted)), np.linspace(0, 2637, 128, dtype=int)])
    for i in selected:
        x = data['observations'][i]
        runtime = np.r_[x[:300], np.zeros(3, np.float32), x[300:]].astype(np.float32)
        with torch.no_grad():
            command = actor(torch.from_numpy(runtime[None])).numpy()[0]
        np.testing.assert_array_equal(command, ACTIONS[predicted[i]])
        budget.tick()
    write('MODEL_ADMISSION.json', dict(at=stamp(), inputs=inputs, checkpoint_sha256=fit['checkpoint_sha256'],
        optimizer_and_rng_loaded=True, all_demonstrations_reproduced=len(predicted),
        old_stop_labels_preserved=95, exported_decisions_checked=len(selected), no_optimizer_updates=True))
    del actor, model
    physical = OUT / 'physical'
    physical.mkdir()
    record_provenance(physical, bundle, [Path(__file__)])
    fresh = json.loads((BASE / 'proactive_augmented_protocol_20261016/fresh_cases.json').read_text())
    cases = json.loads((BASE / 'w11_recovery_bc800_navigation_20261016/cases.json').read_text())
    holdouts = json.loads((PROTOCOL / 'heldout_cases.json').read_text())
    report = dict(completed=False, rows=[], gates={}, new_ppo_steps=0, new_optimizer_updates=0,
        actual106_not_claimed_fixed=True, broader_regression_still_required=True,
        goal_achieved=False, deployment_approved=False)

    def save():
        report['resource'] = budget.statistics()
        write('RESULTS.json', report)

    def finish(reason):
        verify()
        report.update(completed=True, at=stamp(), reason=reason)
        save()
        write('STATUS.json', dict(at=stamp(), status='complete', reason=reason))

    def run(case, group, candidate_bundle=bundle, variant='early_yield_remaining_bc600', recovery=False):
        verify()
        write('STATUS.json', dict(at=stamp(), status='learned_physical_validation', group=group,
            case_id=case['id'], variant=variant, finished=len(report['rows'])))
        row, trace = (evaluate_recovery if recovery else evaluate)(case, candidate_bundle, budget)
        path = physical / (group + '_' + case['id'] + '_' + variant + '.npz')
        np.savez_compressed(path, **trace)
        row.update(group=group, variant=variant, trace_path=str(path.relative_to(ROOT)),
            trace_sha256=digest(path), learned_model_only_after_initialization=True)
        report['rows'].append(row)
        save()
        return row

    fitted_ids = {'proactive_fresh_condition_01'} | {
        item['case_id'] for item in fit['sources'] if item['admitted']}
    originals = []
    original_efficiency = []
    for suffix in ('01', '02', '04', '05'):
        case = dict(next(case for case in fresh if case['id'] == 'proactive_fresh_condition_' + suffix))
        case['pilot_category'] = ('positive_yield_source_in_current_or_parent_fit'
            if case['id'] in fitted_ids else 'not_fitted_condition_same_development_map')
        row = run(case, 'original_collision_conditions')
        originals.append(row)
        reference = next((item for item in teacher_rows if item['case_id'] == row['case_id']), None)
        if suffix == '04':
            reference = next(item for item in prior['rows'] if item['case_id'] == row['case_id'])
        if reference is not None:
            original_efficiency.append(dict(case_id=row['case_id'], reference_seconds=reference['seconds'],
                reference_travel_m=reference['travel_m'],
                passed=bool(row['safe_success'] and row['seconds'] <= reference['seconds'] * 1.2 + 1e-8
                    and row['travel_m'] <= reference['travel_m'] * 1.2 + 1e-8)))
    report['gates']['original_four_safe'] = all(row['safe_success'] for row in originals)
    report['gates']['original_safe_reference_efficiency'] = original_efficiency
    if not report['gates']['original_four_safe'] or not all(row['passed'] for row in original_efficiency):
        finish('Original collision conditions or declared efficiency gate failed; no expansion')
        return
    before, after = [], []
    for case in holdouts:
        before.append(run(case, 'heldout_sensor_conditions', PARENT / 'bc_portable', 'prior_1b55'))
        after.append(run(case, 'heldout_sensor_conditions'))
    paired = compare(before, after)
    report['gates']['heldout_pair'] = paired
    report['gates']['heldout_candidate4_safe'] = all(row['safe_success'] for row in after)
    if not report['gates']['heldout_candidate4_safe'] or not paired['all_regression_checks_passed']:
        finish('Unfitted sensor seeds did not all retain safety/efficiency; no expansion')
        return
    for key, collection, group in [('localization_nav_33', fresh, 'retained_fresh33'),
            ('train_failure_w11_e7', cases, 'normal_w11')]:
        row = run(next(case for case in collection if case['id'] == key), group)
        parents = broad['groups']['fresh12']['rows'] if group == 'retained_fresh33' else broad['full18_rows']
        parent = next(item for item in parents if item['case_id'] == key)
        paired = compare([parent], [row])
        report['gates'][group] = paired
        if not paired['all_regression_checks_passed']:
            finish('Previously retained normal navigation regressed')
            return
    for filename, release, group in [('actual95_learned_validation_20261016/prefixes.json', 95., 'retained_actual95'),
            ('w11_recovery_bc800_w11_20261016/specifications.json', 90., 'retained_actual90')]:
        specifications = json.loads((BASE / filename).read_text())
        row = run(next(item for item in specifications if item['release_s'] == release), group, recovery=True)
        parent = next(item for item in known['rows'] if item.get('release_sensor_timestamp_s') == release)
        paired = compare([parent], [row])
        report['gates'][group] = paired
        if not paired['all_regression_checks_passed']:
            finish('Previously retained learned recovery regressed')
            return
    report['bounded_gates_passed'] = True
    finish('Bounded gates passed; complete regressions and actual106 prevention/recovery remain required')


if __name__ == '__main__':
    os.chdir(ROOT)
    try:
        main()
    except Exception as error:
        write('FAILURE.json', dict(at=stamp(), error=repr(error), automatically_restarted=False))
        raise
