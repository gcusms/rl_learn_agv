"""Bounded broad physics diagnosis; unresolved hard states remain explicit."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess

import numpy as np
import torch

from experiments.evaluate_docking_navigation import evaluate
from experiments.long_risk_actor import digest
from experiments.paired_navigation_gate import compare
from experiments.release_recovery import evaluate_recovery
from neural_navigation.experiment_budget import ExperimentBudget
from neural_navigation.goal_pose_runtime import check_files


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'reports/stop_feedback_20261015'


def write(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def main(args):
    output, prior, protocol = [getattr(args, k).resolve() for k in ['output', 'prior', 'protocol']]
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    budget = ExperimentBudget(.35, memory_mib=3072, directory=output)
    process = subprocess.check_output(['systemctl', '--user', 'show', args.predecessor_unit,
        '-p', 'MainPID', '-p', 'ActiveState'], text=True)
    assert 'MainPID=0' in process and 'ActiveState=inactive' in process
    previous = json.loads((prior / 'RESULTS.json').read_text())
    assert previous['completed'] and previous['safe_physical_diagnostics'] == 3
    assert all(v['all_regression_checks_passed'] for v in previous['comparisons'].values())
    for name, expected in json.loads((prior / 'PLAN.json').read_text())['inputs'].items():
        assert digest(ROOT / name) == expected, name
    bundle = prior / 'diagnostic_bundle'
    manifest = json.loads((bundle / 'manifest.json').read_text())
    assert manifest['diagnostic_only'] and not manifest['deployment_approved']
    check_files(bundle, manifest)
    tasks = json.loads((protocol / 'tasks.json').read_text())
    late_path = BASE / 'parent_wait_protocol_20261016/specifications.json'
    late = json.loads(late_path.read_text())
    assert len(late) == 2
    hard = [tasks['actual106'], *late]
    files = [Path(__file__), prior / 'RESULTS.json', prior / 'PLAN.json', prior / 'MODEL_CHECKS.json',
        protocol / 'tasks.json', late_path, bundle / 'manifest.json']
    files += [bundle / name for name in manifest['files']]
    files += [ROOT / spec['trace'] for spec in [*hard, *tasks['prefixes2']]]
    for folder in ['neural_navigation', 'navrl', 'experiments']:
        files += list((ROOT / 'training' / folder).rglob('*.py'))
    files += [ROOT / 'training/neural_navigation/static_tracking_kernel.cpp']
    for case in [*tasks['heading2'], *tasks['sensor5'], *tasks['normal18'], *tasks['extra12'],
            *[s['case'] for s in [*hard, *tasks['prefixes2']]]]:
        files += [p for p in (ROOT / 'training' / case['scene']).iterdir() if p.is_file()]
    hashes = {str(p.relative_to(ROOT)): digest(p) for p in set(files)}
    write(output / 'PLAN.json', dict(at=datetime.now(timezone.utc).isoformat(), inputs=hashes,
        maximum_new_physical_episodes=42,
        ordered_groups=['hard3', 'heading2', 'sensor5', 'normal18', 'extra12', 'prefixes2'],
        hard3_specifications=hard, hard_timeouts_remain_unresolved=True,
        stop_if_hard_group_has_collision_false_arrival_guard_or_invalid_protocol=True,
        stop_on_heading_sensor_or_normal_safety_efficiency_failure=True,
        all_tasks_are_development=True, no_full_mission_claim_for_initialized_prefixes=True,
        original_source_fit_gate_remains_failed=True, diagnostic_only=True,
        original_actions_masks_body_margin_memory_deadlines=True,
        no_teacher_after_initialization=True, new_ppo_steps=0, new_optimizer_updates=0,
        no_automatic_promotion=True, deployment_approved=False))
    (output / 'driver_snapshot.py').write_text(Path(__file__).read_text())
    physical = output / 'physical'; physical.mkdir()
    report = dict(completed=False, groups={}, gates={}, new_physical_episodes=0,
        diagnostic_only=True, original_source_fit_gate_remains_failed=True,
        new_ppo_steps=0, new_optimizer_updates=0, deployment_approved=False, goal_achieved=False)

    def save():
        budget.tick()
        report.update(at=datetime.now(timezone.utc).isoformat(), resource=budget.statistics())
        write(output / 'RESULTS.json', report)

    def finish(reason):
        for name, expected in hashes.items():
            assert digest(ROOT / name) == expected, name
        assert report['new_physical_episodes'] <= 42
        report.update(completed=True, reason=reason, inputs_verified=len(hashes))
        save()
        write(output / 'COMPLETION_CHECKS.json', dict(completed=True, inputs_verified=len(hashes),
            diagnostics_do_not_approve_deployment=True))
        write(output / 'STATUS.json', dict(status='complete', reason=reason,
            completed_episodes=report['new_physical_episodes']))

    def run(specification, group, recovery=False):
        write(output / 'STATUS.json', dict(status='bounded_physics_diagnosis', group=group,
            case_id=specification['id'], completed_episodes=report['new_physical_episodes']))
        print(json.dumps(dict(starting=specification['id'], group=group)), flush=True)
        row, trace = (evaluate_recovery if recovery else evaluate)(specification, bundle, budget)
        path = physical / (group + '_' + specification['id'] + '.npz')
        np.savez_compressed(path, **trace)
        row.update(trace_path=str(path.relative_to(ROOT)), trace_sha256=digest(path),
            variant='context_release2000_s231016', pure_learned_model=True, diagnostic_only=True)
        report['groups'].setdefault(group, dict(rows=[]))['rows'].append(row)
        report['new_physical_episodes'] += 1
        save(); print(json.dumps(row), flush=True)
        return row

    def paired(group):
        rows = report['groups'][group]['rows']
        value = compare(tasks['reference'][group], rows)
        report['groups'][group]['paired'] = value
        passed = all(r['safe_success'] for r in rows) and value['all_regression_checks_passed']
        report['gates'][group] = dict(passed=bool(passed))
        save()
        return passed

    save()
    for spec in hard:
        run(spec, 'hard3', True)
    hard_rows = report['groups']['hard3']['rows']
    report['hard_states_all_solved'] = all(r['safe_success'] for r in hard_rows)
    safe_to_expand = all(r['protocol_valid'] and not any(r[k] for k in
        ['collisions', 'false_arrival', 'physical_guards']) for r in hard_rows)
    report['gates']['hard_diagnostic_safety'] = dict(passed=bool(safe_to_expand),
        all_hard_states_solved=report['hard_states_all_solved'])
    if not safe_to_expand:
        finish('Hard-state physical safety regression; no wider expansion')
        return
    for case in tasks['heading2']:
        run(case, 'heading2')
    if not paired('heading2'):
        finish('Heading safety or efficiency regression; no wider expansion')
        return
    for case in tasks['sensor5']:
        row = run(case, 'sensor5')
        reference = next(r for r in tasks['reference']['sensor5'] if r['case_id'] == row['case_id'])
        passed = row['safe_success'] and all(row[k] <= reference[k] * 1.2 + 1e-8
            for k in ['seconds', 'travel_m'])
        report['gates'][case['id']] = dict(passed=bool(passed))
        if not passed:
            finish('Sensor retention safety or efficiency regression; no wider expansion')
            return
    for case in tasks['normal18']:
        run(case, 'normal18')
    if not paired('normal18'):
        finish('Normal18 safety or efficiency regression; no wider expansion')
        return
    for case in tasks['extra12']:
        run(case, 'extra12')
    for spec in tasks['prefixes2']:
        run(spec, 'prefixes2', True)
    paired('extra12'); paired('prefixes2')
    report['all_declared_diagnostics_safe'] = all(r['safe_success']
        for group in report['groups'].values() for r in group['rows'])
    finish('All declared diagnostics finished; unresolved hard/extra failures and fit rejection remain explicit')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['prior', 'protocol', 'output']:
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--predecessor-unit', default='rl-context-physics-diagnostic-20261016.service')
    args = parser.parse_args()
    try:
        main(args)
    except Exception as error:
        if args.output.is_dir():
            write(args.output / 'FAILURE.json', dict(error=repr(error), automatically_restarted=False))
        raise
