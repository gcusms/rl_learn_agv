"""Independent, gated physical validation of the queued matched-v2 PPO.

Reused development tasks and previously unexecuted seeds are reported apart.
No successful fit, training episode or process exit substitutes for navigation.
"""
from datetime import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import time
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT/'reports/stop_feedback_20261015'
OUT = BASE/'guard_aligned_mixed_ppo_resume_validation_20261017'
MODEL = ROOT/'models/stop_feedback_20261015/guard_aligned_mixed_ppo_resume_s231017'
BASELINE = ROOT/'models/stop_feedback_20261015/normal_suite_wait_pair_v2_s231016/wait_continuation/portable'
DEPENDENCY = 'rl-guard-aligned-ppo-resume-20261017.service'
INVOCATION = 'f4a81eee5fc14358a91653e882f19f46'


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')
    temporary.replace(path)


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    task_path = BASE/'context_feedback_longfit_protocol_20261016/tasks.json'
    parent_path = BASE/'normal_suite_wait_validation_v2_20261016/RESULTS.json'
    known_path = BASE/'native625_local_speedup_retention_20261017/RESULTS.json'
    known_cases_path = BASE/'native625_frozen_holdout_20261016/CASES.json'
    fresh_dir = BASE/'memory_joint_retention_protocol_20261017'
    fresh_path, freeze_path = fresh_dir/'CASES.json', fresh_dir/'FREEZE.json'
    paths = [Path(__file__), task_path, parent_path, known_path, known_cases_path, fresh_path, freeze_path,
        BASE/'guard_aligned_mixed_ppo_resume_20261017/PLAN.json']
    for folder in ['experiments', 'neural_navigation', 'navrl']:
        paths += list((ROOT/'training'/folder).rglob('*.py'))
    hashes = {str(p.relative_to(ROOT)): digest(p) for p in set(paths)}
    write(OUT/'PLAN.json', dict(at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(), inputs=hashes,
        dependency=DEPENDENCY, expected_invocation=INVOCATION, maximum_new_physical_episodes=53,
        ordered_groups=dict(fixed_preservation=28, observed_full_mission_diagnostics=6,
            same_state_hard_recoveries=3, previously_unexecuted_seed_pairs=8),
        fresh_case_sha256=digest(fresh_path), fresh_freeze_sha256=digest(freeze_path),
        seeds_not_maps_are_fresh=True, no_training_on_fresh_outcomes=True,
        preserve_both_source_bc4500_and_older_fixed_reference=True,
        at_least_one_gained_safe_success_before_fresh_expansion=True,
        late106_requirement_never_waived=True, no_automatic_model_promotion=True,
        no_parallel_physical_job=True, soft_cpu_cores=.35, memory_hard_max_mib=3072,
        new_ppo_steps=0, goal_achieved=False, deployment_approved=False))
    while True:
        checked = subprocess.run(['systemctl', '--user', 'show', DEPENDENCY,
            '--property=ActiveState,MainPID,InvocationID,ExecMainStatus'], capture_output=True, text=True, check=True)
        state = dict(line.split('=', 1) for line in checked.stdout.splitlines() if '=' in line)
        if state['InvocationID'] not in ('', INVOCATION):
            raise RuntimeError('The exact PPO dependency changed invocation')
        if state['MainPID'] == '0':
            training_path = BASE/'guard_aligned_mixed_ppo_resume_20261017/RESULTS.json'
            training = read(training_path) if training_path.exists() else {}
            if (state['ActiveState'] != 'inactive' or state['ExecMainStatus'] != '0'
                    or not training.get('completed') or not training.get('training_started')):
                write(OUT/'RESULTS.json', dict(completed=True, evaluation_started=False,
                    reason='PPO did not complete an admitted training run', training_reason=training.get('reason'),
                    new_physical_episodes=0, new_ppo_steps=0, goal_achieved=False, deployment_approved=False))
                write(OUT/'STATUS.json', dict(status='not_started_no_admitted_ppo'))
                return
            paths = [training_path, MODEL/'checkpoint.pt', MODEL/'parent_actor.pt', MODEL/'RESULTS.json']
            for bundle in [MODEL/'portable', BASELINE]:
                manifest = read(bundle/'manifest.json')
                paths += [bundle/'manifest.json']+[bundle/name for name in manifest['files']]
            hashes.update({str(p.relative_to(ROOT)): digest(p) for p in paths})
            break
        write(OUT/'STATUS.json', dict(status='waiting_for_exact_matched_v2_ppo', dependency=state,
            neural_and_physics_libraries_not_loaded=True, new_physical_episodes=0))
        time.sleep(5.)
    for name, expected in hashes.items():
        assert digest(ROOT/name) == expected, name

    import gc
    import numpy as np
    import torch
    from experiments.evaluate_docking_navigation import evaluate
    from experiments.guard_reason_release_actor import GuardReasonReleaseActor
    from experiments.release_recovery import evaluate_recovery
    from experiments.paired_navigation_gate import compare
    from neural_navigation.experiment_budget import ExperimentBudget
    from neural_navigation.goal_pose_runtime import check_files
    from neural_navigation.aligned_release_runtime import RUNTIME_NAME, contract

    torch.set_num_threads(1)
    budget = ExperimentBudget(.35, memory_mib=2800, directory=OUT)
    fit = read(MODEL/'RESULTS.json')
    assert fit['completed'] and fit['all_reference_stop_labels_retained']
    assert digest(MODEL/'checkpoint.pt') == fit['checkpoint_sha256']
    bundle = MODEL/'portable'
    manifest = read(bundle/'manifest.json')
    assert manifest['required_runtime'] == RUNTIME_NAME and manifest['observation_contract'] == contract()
    check_files(bundle, manifest)
    actor = GuardReasonReleaseActor(torch.jit.load(str(MODEL/'parent_actor.pt'), map_location='cpu')).eval()
    actor.load_state_dict(torch.load(MODEL/'checkpoint.pt', map_location='cpu', weights_only=False)['actor'])
    portable = torch.jit.load(str(bundle/'actor.pt'), map_location='cpu').eval()
    anchors_path = BASE/'guard_forecast_alignment_protocol_20261017/aligned_demonstrations.npz'
    with np.load(anchors_path, allow_pickle=False) as data:
        x = torch.from_numpy(data['observations'].copy())
    with torch.no_grad():
        np.testing.assert_array_equal(actor(x).numpy(), portable(x).numpy())
    del actor, portable, x
    gc.collect()
    tasks, parent, known = [read(p) for p in [task_path, parent_path, known_path]]
    parent_rows = {r['case_id']: r for g in parent['groups'].values() for r in g['rows']}
    known_rows = [r for r in known['frozen_rows'] if r['variant'] == 'parent']
    assert len(known_rows) == 6
    known_lookup = {c['id']: c for c in read(known_cases_path)}
    known_cases = [known_lookup[r['case_id']] for r in known_rows]
    hard_reference = parent['groups']['hard3']['rows']
    hard_specs = [dict(id=r['case_id'], case=tasks['actual106']['case'],
        release_s=r['release_sensor_timestamp_s'], trace=r['trace_path']) for r in hard_reference]
    fresh = read(fresh_path)
    assert len(fresh) == 8
    recoveries = [*tasks['actual95_actual90'], tasks['delayed_wait']]
    paths = [anchors_path]+[ROOT/s['trace'] for s in recoveries+hard_specs]
    scenes = {c['scene'] for c in tasks['normal18']+tasks['sensor5']+tasks['heading2']+known_cases+fresh}
    scenes |= {s['case']['scene'] for s in recoveries+hard_specs}
    for scene in scenes:
        paths += [p for p in (ROOT/'training'/scene).iterdir() if p.is_file()]
    hashes.update({str(p.relative_to(ROOT)): digest(p) for p in paths})
    write(OUT/'DEPENDENCIES.json', dict(inputs=hashes, training_steps=fit['new_ppo_steps'],
        portable_checkpoint_parity_rows=3799, fresh_freeze=read(freeze_path)))
    (OUT/'physical').mkdir()
    report = dict(completed=False, evaluation_started=True, model_path=str(MODEL.relative_to(ROOT)),
        groups={}, gates={}, fresh_rows=[], new_physical_episodes=0, new_optimizer_updates=0,
        new_ppo_steps=0, evaluated_candidate_ppo_steps=fit['new_ppo_steps'],
        goal_achieved=False, deployment_approved=False)

    def save():
        budget.tick()
        report.update(resource=budget.statistics(), at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat())
        write(OUT/'RESULTS.json', report)

    def finish(reason):
        for name, expected in hashes.items():
            assert digest(ROOT/name) == expected, name
        assert report['new_physical_episodes'] <= 53
        report.update(completed=True, reason=reason, inputs_verified=len(hashes))
        save()
        write(OUT/'STATUS.json', dict(status='complete', reason=reason, completed_episodes=report['new_physical_episodes']))

    def run(spec, group, recovery=False, components=bundle, variant='candidate'):
        write(OUT/'STATUS.json', dict(status='post_ppo_physics', group=group, case_id=spec['id'],
            variant=variant, completed_episodes=report['new_physical_episodes']))
        row, trace = (evaluate_recovery if recovery else evaluate)(spec, components, budget)
        path = OUT/'physical'/(group+'_'+spec['id']+'_'+variant+'.npz')
        np.savez_compressed(path, **trace)
        row.update(trace_path=str(path.relative_to(ROOT)), trace_sha256=digest(path), variant=variant)
        if group == 'fresh8':
            report['fresh_rows'].append(row)
        else:
            report['groups'].setdefault(group, dict(rows=[]))['rows'].append(row)
        report['new_physical_episodes'] += 1
        save()
        print(json.dumps(dict(group=group, **row)), flush=True)
        return row

    def retained(row, reference):
        key = 'post_release_seconds' if row.get('recovery_test') else 'seconds'
        return bool(row['safe_success'] and row['protocol_valid']
            and not any(row[k] for k in ['collisions', 'false_arrival', 'physical_guards'])
            and all(row[k] <= reference[k]*1.2+1e-8 for k in [key, 'travel_m']))

    fixed = {g: {r['case_id']: r for r in rows} for g, rows in [
        ('normal18', tasks['reference']['normal18']), ('sensor5', tasks['reference']['sensor5']),
        ('source_recovery3', [*tasks['reference']['actual95_actual90'], *tasks['reference']['delayed_wait']]),
        ('heading2', tasks['reference']['heading2'])]}
    for group, specs, recovery in [('normal18', tasks['normal18'], False), ('sensor5', tasks['sensor5'], False),
            ('source_recovery3', recoveries, True), ('heading2', tasks['heading2'], False)]:
        for spec in specs:
            row = run(spec, group, recovery)
            passed = retained(row, fixed[group][row['case_id']]) and retained(row, parent_rows[row['case_id']])
            report['gates'][row['case_id']] = dict(passed=passed)
            if not passed:
                finish('Post-PPO fixed success or efficiency lost; expansion stopped')
                return
        if group == 'normal18':
            rows = report['groups'][group]['rows']
            report['normal18_pair'] = compare([parent_rows[r['case_id']] for r in rows], rows)
            report['normal18_against_fixed'] = compare(tasks['reference']['normal18'], rows)
            if not (report['normal18_pair']['all_regression_checks_passed']
                    and report['normal18_against_fixed']['all_regression_checks_passed']):
                finish('Post-PPO normal18 aggregate efficiency failed')
                return
    report['all_fixed_retention_passed'] = True
    gains = []
    for spec, reference in zip(known_cases, known_rows):
        row = run(spec, 'observed6')
        if any(row[k] for k in ['collisions', 'false_arrival', 'physical_guards']):
            finish('Known development case still unsafe; no fresh expansion')
            return
        if reference['safe_success'] and not retained(row, reference):
            finish('Known complete-mission success or efficiency lost')
            return
        if row['safe_success'] and not reference['safe_success']:
            gains.append(row['case_id'])
    for spec, reference in zip(hard_specs, hard_reference):
        row = run(spec, 'hard3', True)
        if any(row[k] for k in ['collisions', 'false_arrival', 'physical_guards']):
            finish('Hard recovery safety failed; no fresh expansion')
            return
        if reference['safe_success'] and not retained(row, reference):
            finish('Previously successful hard recovery lost')
            return
        if row['safe_success'] and not reference['safe_success']:
            gains.append(row['case_id'])
    report['gained_development_safe_successes'] = gains
    report['hard3_all_safe'] = all(r['safe_success'] for r in report['groups']['hard3']['rows'])
    if not gains:
        finish('No gained safe success; do not consume prospective seeds')
        return
    for spec in fresh:
        for variant, components in [('parent', BASELINE), ('candidate', bundle)]:
            row = run(spec, 'fresh8', components=components, variant=variant)
            if variant == 'candidate' and any(row[k] for k in ['collisions', 'false_arrival', 'physical_guards']):
                finish('Prospective seed candidate safety failed; remaining pairs stopped')
                return
    before = [r for r in report['fresh_rows'] if r['variant'] == 'parent']
    after = [r for r in report['fresh_rows'] if r['variant'] == 'candidate']
    report['fresh_seed_pair_comparison'] = compare(before, after)
    finish('Declared post-PPO protocol complete; full objective/deployment require separate completion audit')


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        if OUT.exists():
            write(OUT/'FAILURE.json', dict(error=repr(error), automatically_restarted=False))
        raise
