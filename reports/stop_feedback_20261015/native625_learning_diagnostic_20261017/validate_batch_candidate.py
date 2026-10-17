"""Serial closed-loop checks after the pinned native625 PPO pilot finishes.

Waiting loads no model. An unavailable or safety-rejected PPO artifact is not
evaluated. Training motion labels may change, while all safe STOP labels and
the physical navigation regression gates remain mandatory.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import time


ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / 'reports/stop_feedback_20261015'
DEPENDENCY = 'rl-native625-batch-credit-20261017.service'
INVOCATION = 'dcad1b530ad94e64b7a9c7ad39e71611'
MODEL = ROOT / 'models/stop_feedback_20261015/native625_batch_credit_s231017'


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')
    temporary.replace(path)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def state():
    result = subprocess.run(['systemctl', '--user', 'show', DEPENDENCY,
        '--property=MainPID,ActiveState,InvocationID,ExecMainStatus'],
        text=True, capture_output=True, check=True)
    return dict(line.split('=', 1) for line in result.stdout.splitlines() if '=' in line)


def main(args):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    tasks_path = BASE / 'context_feedback_longfit_protocol_20261016/tasks.json'
    late_path = BASE / 'parent_wait_protocol_20261016/specifications.json'
    tasks, late = read(tasks_path), read(late_path)
    raw_wait_trace = BASE / 'context_broad_diagnostic_20261016/physical/hard3_w11_training_stall_t106.npz'
    case = tasks['actual106']['case']
    wait = dict(id='actual_training_initialization90', case=case, trace=str(raw_wait_trace), release_s=90.)
    hard = [tasks['actual106'], *late]
    files = [Path(__file__), tasks_path, late_path, raw_wait_trace]
    for folder in ['experiments', 'neural_navigation', 'navrl']:
        files += list((ROOT / 'training' / folder).rglob('*.py'))
    files += [ROOT / 'training/neural_navigation/static_tracking_kernel.cpp']
    files += [ROOT / spec['trace'] for spec in hard + tasks['actual95_actual90'] + [tasks['delayed_wait']]]
    scenes = {c['scene'] for c in tasks['normal18'] + tasks['sensor5'] + tasks['heading2']}
    scenes |= {spec['case']['scene'] for spec in hard + tasks['actual95_actual90'] + [tasks['delayed_wait']]}
    for scene in scenes:
        files += [p for p in (ROOT / 'training' / scene).iterdir() if p.is_file()]
    hashes = {str(path.relative_to(ROOT)): digest(path) for path in set(files)}
    write(output / 'PLAN.json', dict(inputs=hashes, exact_dependency=DEPENDENCY,
        expected_invocation=INVOCATION, maximum_physical_episodes=33,
        order=['parent_actual90', 'ppo_actual90', 'ppo_hard3',
            'normal_priority2', 'sensor_priority2', 'source_recovery3', 'heading2',
            'remaining_sensor3', 'remaining_normal16'],
        expand_only_if_early_or_hard_gain_or_all_hard_safe=True,
        compare_normal18_with_fixed_reference_and_new_parent=True,
        same_body4cm_actions_masks_sensors_dynamics_and_deadlines=True,
        source_motion_changes_not_automatically_a_failure=True,
        all_reference_stop_labels_required=True,
        no_optimizer_updates=True, all_tasks_development_not_unseen_generalization=True,
        prefix_is_initialization_not_learned_full_mission=True,
        no_automatic_deployment=True))
    while True:
        observed = state()
        if observed['InvocationID'] not in ('', INVOCATION):
            raise RuntimeError('Pinned PPO dependency was replaced or restarted')
        if observed['ActiveState'] in ('failed', 'inactive') and observed['MainPID'] == '0':
            break
        write(output / 'STATUS.json', dict(status='waiting_for_exact_ppo_predecessor',
            dependency=observed, new_physical_episodes=0, new_ppo_steps=0))
        time.sleep(5.)
    stage_path = BASE / 'native625_batch_credit_pilot_20261017/PLAN.json'
    ready = bool(observed['ActiveState'] == 'inactive' and observed['ExecMainStatus'] == '0')
    training_path = BASE / 'native625_batch_credit_pilot_20261017/RESULTS.json'
    training = read(training_path) if ready else {}
    ready = ready and bool(training.get('completed') and training.get('new_ppo_steps', 0) > 0
        and training.get('diagnostic_portable_exported') and training.get('all_reference_stop_labels_retained'))
    if not ready:
        write(output / 'RESULTS.json', dict(completed=True, physical_validation_started=False,
            reason='Exact PPO predecessor did not produce a source-STOP-preserving complete candidate',
            stage_result_sha256=digest(stage_path) if stage_path.exists() else None,
            training_result_sha256=digest(training_path) if training_path.exists() else None,
            new_physical_episodes=0, new_ppo_steps=0, goal_achieved=False, deployment_approved=False))
        write(output / 'STATUS.json', dict(status='not_started_predecessor_gate_failed'))
        return

    import numpy as np
    import torch
    from experiments.evaluate_docking_navigation import evaluate
    from experiments.guard_reason_release_actor import GuardReasonReleaseActor
    from experiments.paired_navigation_gate import compare
    from experiments.release_recovery import evaluate_recovery
    from neural_navigation.experiment_budget import ExperimentBudget
    from neural_navigation.goal_pose_runtime import check_files
    from neural_navigation.masked_actions import ACTIONS

    parent_validation_path = BASE / 'normal_suite_wait_validation_v2_20261016/RESULTS.json'
    parent_validation = read(parent_validation_path)
    assert parent_validation['completed'] and parent_validation['normal18_pair']['all_regression_checks_passed']
    parent = ROOT / parent_validation['selected_model']
    bundle = MODEL / 'portable'
    for folder in [bundle, parent / 'portable']:
        check_files(folder, read(folder / 'manifest.json'))
        assert not read(folder / 'manifest.json')['deployment_approved']
    parent_rows = {r['case_id']:r for g in parent_validation['groups'].values() for r in g['rows']}
    anchors_path = BASE / 'normal_suite_wait_protocol_v2_20261016/demonstrations.npz'
    checkpoint = MODEL / 'checkpoint.pt'
    files = [stage_path, training_path, parent_validation_path, anchors_path, checkpoint, parent / 'parent_actor.pt']
    for folder in [bundle, parent / 'portable']:
        manifest = read(folder / 'manifest.json')
        files += [folder / 'manifest.json'] + [folder / name for name in manifest['files']]
    hashes.update({str(p.relative_to(ROOT)):digest(p) for p in files})
    for name, expected in hashes.items():
        assert digest(ROOT / name) == expected, name
    write(output / 'DEPENDENCIES.json', dict(inputs=hashes,
        completed_training_steps=training['new_ppo_steps'],
        motion_source_correct=training['correct_source_labels'], source_total=training['total_source_labels']))
    torch.set_num_threads(1)
    budget = ExperimentBudget(.35, memory_mib=2800, directory=output)
    with np.load(anchors_path, allow_pickle=False) as source:
        x, labels = source['observations'].copy(), source['choices'].copy()
    actor = GuardReasonReleaseActor(torch.jit.load(str(parent / 'parent_actor.pt'), map_location='cpu')).eval()
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    assert saved['ppo_decisions'] == training['new_ppo_steps']
    actor.load_state_dict(saved['actor'])
    portable = torch.jit.load(str(bundle / 'actor.pt'), map_location='cpu').eval()
    with torch.no_grad():
        expected = actor(torch.from_numpy(x)).numpy()
        np.testing.assert_array_equal(portable(torch.from_numpy(x)).numpy(), expected)
    matches = (expected == ACTIONS[labels]).all(1)
    assert bool(matches[labels == 0].all())
    assert int(matches.sum()) == training['correct_source_labels']
    del actor, portable, saved, x, labels
    budget.tick()
    physical = output / 'physical'
    physical.mkdir()
    report = dict(completed=False, physical_validation_started=True, groups={}, gates={},
        new_physical_episodes=0, new_optimizer_updates=0, new_ppo_steps=0,
        validated_candidate_ppo_steps=training['new_ppo_steps'],
        goal_achieved=False, deployment_approved=False)

    def save():
        budget.tick()
        report.update(at=datetime.now(timezone.utc).isoformat(), resource=budget.statistics())
        write(output / 'RESULTS.json', report)

    def finish(reason):
        for name, expected in hashes.items():
            assert digest(ROOT / name) == expected, name
        report.update(completed=True, reason=reason, inputs_verified=len(hashes))
        save()
        write(output / 'STATUS.json', dict(status='complete', reason=reason,
            completed_episodes=report['new_physical_episodes']))

    def run(spec, group, recovery=False, target=bundle):
        write(output / 'STATUS.json', dict(status='physical_validation', group=group,
            case_id=spec['id'], completed_episodes=report['new_physical_episodes']))
        row, trace = (evaluate_recovery if recovery else evaluate)(spec, target, budget)
        path = physical / (group+'_'+spec['id']+'.npz')
        np.savez_compressed(path, **trace)
        row.update(trace_path=str(path.relative_to(ROOT)), trace_sha256=digest(path),
            variant=str(target.parent.relative_to(ROOT)), no_optimizer_updates=True,
            development_reused_not_independent_generalization=True)
        report['groups'].setdefault(group, dict(rows=[]))['rows'].append(row)
        report['new_physical_episodes'] += 1
        save()
        print(json.dumps(dict(group=group, **row)), flush=True)
        return row

    def safe(row):
        return row['protocol_valid'] and not any(row[k] for k in ['collisions', 'false_arrival', 'physical_guards'])

    def retained(row, reference):
        time_key = 'post_release_seconds' if row.get('recovery_test') else 'seconds'
        return safe(row) and row['safe_success'] and all(
            row[k] <= reference[k]*1.2+1e-8 for k in [time_key, 'travel_m'])

    save()
    reference = run(wait, 'parent_actual90', True, target=parent / 'portable')
    if not safe(reference):
        finish('Parent physical safety failed at actual90; inspect before expansion')
        return
    row = run(wait, 'ppo_actual90', True)
    if not safe(row) or not row['safe_success'] or (reference['safe_success'] and not retained(row, reference)):
        finish('Candidate actual90 recovery or retention failed; no wider expansion')
        return
    early_gain = bool(row['safe_success'] and not reference['safe_success'])
    report['early_recovery_comparison'] = dict(gained=early_gain, parent_safe_success=reference['safe_success'], candidate_safe_success=row['safe_success'])
    for specification in hard:
        row = run(specification, 'hard3', True)
        if not safe(row):
            finish('PPO hard-state safety failure; no wider expansion')
            return
    hard_rows = report['groups']['hard3']['rows']
    gains = [r['case_id'] for r in hard_rows if r['safe_success'] and not parent_rows[r['case_id']]['safe_success']]
    losses = [r['case_id'] for r in hard_rows if not r['safe_success'] and parent_rows[r['case_id']]['safe_success']]
    report['hard_state_comparison'] = dict(gained=gains, lost=losses,
        all_safe_success=all(r['safe_success'] for r in hard_rows))
    if losses or not (early_gain or gains or all(r['safe_success'] for r in hard_rows)):
        finish('No early/hard recovery gain or a lost hard success; full regression expansion not justified')
        return
    priority_ids = ['independent_dynamic_000', 'train_failure_w11_e7']
    sensor_ids = ['sensor_release_frozen_noise_05_0', 'sensor_release_frozen_noise_05_1']
    groups = [('normal18', [c for name in priority_ids for c in tasks['normal18'] if c['id']==name], False),
        ('sensor5', [c for name in sensor_ids for c in tasks['sensor5'] if c['id']==name], False),
        ('source_recovery3', [*tasks['actual95_actual90'], tasks['delayed_wait']], True),
        ('heading2', tasks['heading2'], False),
        ('sensor5', [c for c in tasks['sensor5'] if c['id'] not in sensor_ids], False),
        ('normal18', [c for c in tasks['normal18'] if c['id'] not in priority_ids], False)]
    fixed_source = [*tasks['reference']['actual95_actual90'], *tasks['reference']['delayed_wait']]
    references = {g:{r['case_id']:r for r in source} for g,source in [
        ('normal18', tasks['reference']['normal18']), ('sensor5', tasks['reference']['sensor5']),
        ('heading2', tasks['reference']['heading2']), ('source_recovery3', fixed_source)]}
    for group, specifications, recovery in groups:
        for specification in specifications:
            row = run(specification, group, recovery)
            okay = retained(row, references[group][row['case_id']]) and retained(row, parent_rows[row['case_id']])
            report['gates'][row['case_id']] = dict(passed=bool(okay))
            if not okay:
                finish(group+' safety or efficiency retention failure; no wider expansion')
                return
    normal = report['groups']['normal18']['rows']
    assert len(normal) == 18 and report['new_physical_episodes'] == 33
    report['normal18_against_fixed'] = compare(tasks['reference']['normal18'], normal)
    report['normal18_against_parent'] = compare([parent_rows[r['case_id']] for r in normal], normal)
    report['all_development_retention_passed'] = bool(
        report['normal18_against_fixed']['larger_training_gate_passed']
        and report['normal18_against_parent']['larger_training_gate_passed'])
    finish('Declared post-PPO development checks finished; held-out and real-car acceptance remain separate')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    try:
        main(args)
    except Exception as error:
        if args.output.is_dir():
            write(args.output / 'FAILURE.json', dict(error=repr(error), automatically_restarted=False))
        raise
