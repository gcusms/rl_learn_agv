"""Serial full preservation gate for the new v2 observation semantics.

The tasks are reused development regressions. This gate neither establishes
new-map generalization nor solves the known late106 and sensor01 failures.
"""
from datetime import datetime
import hashlib
import json
from pathlib import Path
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT/'reports/stop_feedback_20261015'
OUT = BASE/'guard_forecast_migration_validation_20261017'
MODEL = ROOT/'models/stop_feedback_20261015/guard_aligned_migration_s231017'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')
    temporary.replace(path)


def main():
    import numpy as np
    import torch
    from experiments.evaluate_docking_navigation import evaluate
    from experiments.release_recovery import evaluate_recovery
    from experiments.paired_navigation_gate import compare
    from neural_navigation.experiment_budget import ExperimentBudget
    from neural_navigation.goal_pose_runtime import check_files
    from neural_navigation.aligned_release_runtime import RUNTIME_NAME, contract

    OUT.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    budget = ExperimentBudget(.35, memory_mib=2800, directory=OUT)
    task_path = BASE/'context_feedback_longfit_protocol_20261016/tasks.json'
    parent_path = BASE/'normal_suite_wait_validation_v2_20261016/RESULTS.json'
    fit_path = MODEL/'RESULTS.json'
    tasks, parent, fit = [json.loads(p.read_text()) for p in [task_path, parent_path, fit_path]]
    assert fit['completed'] and fit['all_source_labels_fit'] and fit['exported_for_bounded_physical_validation']
    assert digest(MODEL/'checkpoint.pt') == fit['checkpoint_sha256']
    bundle = MODEL/'portable'
    manifest = json.loads((bundle/'manifest.json').read_text())
    assert manifest['required_runtime'] == RUNTIME_NAME and manifest['observation_contract'] == contract()
    check_files(bundle, manifest)
    parent_rows = {row['case_id']: row for group in parent['groups'].values() for row in group['rows']}
    recoveries = [*tasks['actual95_actual90'], tasks['delayed_wait']]
    paths = [Path(__file__), task_path, parent_path, fit_path, MODEL/'checkpoint.pt', bundle/'manifest.json']
    paths += [bundle/name for name in manifest['files']]
    paths += [ROOT/spec['trace'] for spec in recoveries]
    for folder in ['experiments', 'neural_navigation', 'navrl']:
        paths += list((ROOT/'training'/folder).rglob('*.py'))
    scenes = {case['scene'] for case in tasks['normal18']+tasks['sensor5']+tasks['heading2']}
    scenes |= {spec['case']['scene'] for spec in recoveries}
    for scene in scenes:
        paths += [p for p in (ROOT/'training'/scene).iterdir() if p.is_file()]
    hashes = {str(p.relative_to(ROOT)): digest(p) for p in set(paths)}
    write(OUT/'PLAN.json', dict(at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(), inputs=hashes,
        maximum_new_physical_episodes=28, groups=dict(normal18=18, sensor5=5, source_recovery3=3, heading2=2),
        compare_to_source_bc4500_and_fixed_older_reference=True, original_per_case_limits=1.2,
        original_aggregate_limits=dict(travel_increase=.02, seconds_increase=.05),
        same_625_length_but_explicit_v2_semantics=True, original_masks_hardware_and_4cm_margin=True,
        no_teacher_after_release=True, initialization_not_full_mission_score=True,
        no_fresh_seeds_consumed=True, no_parallel_physical_job=True, new_ppo_steps=0,
        goal_achieved=False, deployment_approved=False))
    (OUT/'physical').mkdir()
    report = dict(completed=False, selected_model=str(MODEL.relative_to(ROOT)), groups={}, gates={},
        new_physical_episodes=0, new_optimizer_updates=0, new_ppo_steps=0,
        goal_achieved=False, deployment_approved=False)

    def save():
        budget.tick()
        report.update(resource=budget.statistics(), at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat())
        write(OUT/'RESULTS.json', report)

    def finish(reason):
        for name, expected in hashes.items():
            assert digest(ROOT/name) == expected, name
        report.update(completed=True, reason=reason, inputs_verified=len(hashes))
        save()
        write(OUT/'STATUS.json', dict(status='complete', reason=reason, completed_episodes=report['new_physical_episodes']))

    def retained(row, reference):
        key = 'post_release_seconds' if row.get('recovery_test') else 'seconds'
        return bool(row['safe_success'] and row['protocol_valid']
            and not any(row[k] for k in ['collisions', 'false_arrival', 'physical_guards'])
            and all(row[k] <= reference[k]*1.2+1e-8 for k in [key, 'travel_m']))

    fixed = {group: {row['case_id']: row for row in rows} for group, rows in [
        ('normal18', tasks['reference']['normal18']), ('sensor5', tasks['reference']['sensor5']),
        ('source_recovery3', [*tasks['reference']['actual95_actual90'], *tasks['reference']['delayed_wait']]),
        ('heading2', tasks['reference']['heading2'])]}
    save()
    for group, specifications, recovery in [('normal18', tasks['normal18'], False),
            ('sensor5', tasks['sensor5'], False), ('source_recovery3', recoveries, True),
            ('heading2', tasks['heading2'], False)]:
        for specification in specifications:
            write(OUT/'STATUS.json', dict(status='v2_full_preservation_physics', group=group,
                case_id=specification['id'], completed_episodes=report['new_physical_episodes']))
            row, trace = (evaluate_recovery if recovery else evaluate)(specification, bundle, budget)
            path = OUT/'physical'/(group+'_'+specification['id']+'.npz')
            np.savez_compressed(path, **trace)
            row.update(trace_path=str(path.relative_to(ROOT)), trace_sha256=digest(path),
                model=str(MODEL.relative_to(ROOT)), v2_runtime=True, no_optimizer_updates=True)
            report['groups'].setdefault(group, dict(rows=[]))['rows'].append(row)
            report['new_physical_episodes'] += 1
            passed = retained(row, fixed[group][row['case_id']]) and retained(row, parent_rows[row['case_id']])
            report['gates'][row['case_id']] = dict(passed=passed)
            save()
            print(json.dumps(dict(group=group, **row)), flush=True)
            if not passed:
                finish('V2 migration lost fixed success or efficiency; stop before new PPO')
                return
        if group == 'normal18':
            rows = report['groups'][group]['rows']
            report['normal18_pair'] = compare([parent_rows[r['case_id']] for r in rows], rows)
            report['normal18_against_fixed'] = compare(tasks['reference']['normal18'], rows)
            if not (report['normal18_pair']['all_regression_checks_passed']
                    and report['normal18_against_fixed']['all_regression_checks_passed']):
                finish('V2 migration aggregate normal18 efficiency failed; stop before new PPO')
                return
    report['all_fixed_retention_passed'] = True
    finish('All28 v2 preservation tasks passed; eligible for bounded PPO, not deployment')


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        if OUT.exists():
            write(OUT/'FAILURE.json', dict(error=repr(error), automatically_restarted=False))
        raise
