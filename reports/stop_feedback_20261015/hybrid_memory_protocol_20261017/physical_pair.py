"""Bounded, serialized memory-only development comparisons.

Wait for the exact preceding validation and replay handles. Never restart them.
Unsolved old failures stay failures; prior success/safety losses stop expansion.
Passing this preflight does not waive the full original validation protocol.
"""
from datetime import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
BASE = HERE.parent
OUT = BASE/'hybrid_memory_pair_preflight_20261017'
BUNDLE = ROOT/'models/stop_feedback_20261015/guard_aligned_migration_s231017/portable'
PREDECESSOR = 'rl-v2-behavior-retention-validate-20261017.service'
EXPECTED_INVOCATION = '2aad2fe0cdc34caca8796f013b79a7d3'
PRIOR_PID = 2180324
PRIOR_START_TICKS = '49348108'
AUDIT_PID = 2182906
AUDIT_START_TICKS = '49399335'


def read(path): return json.loads(path.read_text())
def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def now(): return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()
def write(path, value):
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')
    temporary.replace(path)


def same_process(pid, start_ticks):
    try:
        fields = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
        return fields[19] == start_ticks and fields[0] != 'Z'
    except FileNotFoundError:
        return False


def unit_state():
    result = subprocess.run(['systemctl', '--user', 'show', PREDECESSOR,
        '--property=MainPID,ActiveState,SubState,InvocationID,LoadState'],
        capture_output=True, text=True, check=True)
    return dict(line.split('=', 1) for line in result.stdout.splitlines() if '=' in line)


def main():
    assert not OUT.exists(), 'Fresh output required; no automatic restart'
    OUT.mkdir(parents=True)
    previous = BASE/'v2_behavior_retention_validation_20261017'
    subset_path = HERE/'SUBSET_REPLAY.json'
    tasks_path = BASE/'context_feedback_longfit_protocol_20261016/tasks.json'
    reference_path = BASE/'normal_suite_wait_validation_v2_20261016/RESULTS.json'
    tasks, reference = read(tasks_path), read(reference_path)
    hard = {row['case_id']:row for row in reference['groups']['hard3']['rows']}
    order = ['legal_wait_parent_t90', 'legal_wait_parent_t95', 'w11_training_stall_t106']
    specifications = [dict(recovery=True, spec=dict(id=key, case=tasks['actual106']['case'],
        release_s=hard[key]['release_sensor_timestamp_s'], trace=hard[key]['trace_path']),
        reference=hard[key]) for key in order]
    for case_id in ['train_failure_w11_e7', 'motion_profile_015']:
        spec = next(c for c in tasks['normal18'] if c['id']==case_id)
        old = next(r for r in tasks['reference']['normal18'] if r['case_id']==case_id)
        specifications.append(dict(recovery=False, spec=spec, reference=old))
    files = [Path(__file__), HERE/'physical_worker.py', HERE/'hybrid_memory.py',
        HERE/'audit_subset.py', HERE/'TEST_RESULTS.json', HERE/'REPLAY.json',
        tasks_path, reference_path, previous/'PLAN.json']
    files += [ROOT/s['spec']['trace'] for s in specifications if s['recovery']]
    for folder in ['neural_navigation', 'experiments', 'navrl']:
        files += list((ROOT/'training'/folder).rglob('*.py'))
    manifest = read(BUNDLE/'manifest.json')
    files += [BUNDLE/'manifest.json']+[BUNDLE/name for name in manifest['files']]
    for s in specifications:
        case = s['spec']['case'] if s['recovery'] else s['spec']
        files += [f for f in (ROOT/'training'/case['scene']).iterdir() if f.is_file()]
    hashes = {str(path.relative_to(ROOT)):digest(path) for path in set(files)}
    plan = dict(at=now(), cases=specifications, inputs=hashes,
        frozen_bundle=str(BUNDLE.relative_to(ROOT)), policy_training=False,
        diagnostic_overlay_only=True, factor='memory_free_evidence_rule_only',
        predecessor_unit=PREDECESSOR, expected_invocation=EXPECTED_INVOCATION,
        predecessor_process=dict(pid=PRIOR_PID, start_ticks=PRIOR_START_TICKS),
        subset_audit_process=dict(pid=AUDIT_PID, start_ticks=AUDIT_START_TICKS),
        maximum_new_physical_episodes=10, fresh_worker_process_per_episode=True,
        pair_order=['control', 'hybrid'], scope='reused development comparisons; not unseen-map acceptance',
        install_memory_before_first_sensor_step=True, admissible_prefix_and_physical_parity_required=True,
        no_teacher_after_release=True, unchanged4cm_margin_footprint_actions_blind_rules_dynamics_deadline=True,
        stop_on_collision_false_arrival_or_physical_guard=True,
        stop_on_prior_success_or_original_20percent_efficiency_loss=True,
        unsolved97_6_or106_is_not_waived=True,
        original_fixed28_and_conditional8_pairs_still_required_after_preflight=True,
        soft_cpu_cores=.35, hard_service_memory_mib=3072, rss_budget_mib=2800,
        no_parallel_physical_jobs=True, no_automatic_promotion=True,
        goal_achieved=False, deployment_approved=False)
    write(OUT/'PLAN.json', plan)
    report = dict(completed=False, at=now(), rows=[], pairs=[], new_physical_episodes=0,
        new_ppo_steps=0, new_optimizer_updates=0, goal_achieved=False, deployment_approved=False)
    write(OUT/'RESULTS.json', report)
    while True:
        state = unit_state()
        if state.get('InvocationID') not in ('', EXPECTED_INVOCATION):
            raise RuntimeError('Predecessor invocation changed; do not overlap or infer a restart')
        validator_live = same_process(PRIOR_PID, PRIOR_START_TICKS)
        audit_live = same_process(AUDIT_PID, AUDIT_START_TICKS)
        write(OUT/'STATUS.json', dict(status='waiting_for_exact_predecessors', at=now(),
            validator_live=validator_live, subset_audit_live=audit_live, unit_state=state))
        if not validator_live and state.get('MainPID')=='0' and not audit_live:
            break
        time.sleep(15)
    assert not (previous/'FAILURE.json').exists(), 'Predecessor infrastructure failed; inspect before any new physical job'
    prior_result = read(previous/'RESULTS.json')
    assert prior_result['completed'], 'Missing authoritative terminal validation result'
    subset = read(subset_path)
    assert subset['completed'] and subset['capacity_never_active']
    assert subset['all_retained_points_and_uncertainty_timestamps_match_original']
    assert all(row['hybrid_subset_of_original_at_every_probe'] for row in subset['rows'])
    assert read(HERE/'TEST_RESULTS.json')['passed']
    for name, expected in hashes.items(): assert digest(ROOT/name)==expected, name
    report['predecessor_terminal'] = dict(at=now(), reason=prior_result.get('reason'),
        completed_episodes=prior_result['new_physical_episodes'], result_sha256=digest(previous/'RESULTS.json'))
    report['subset_replay_sha256'] = digest(subset_path)
    (OUT/'jobs').mkdir()
    (OUT/'physical').mkdir()

    def finish(reason):
        for name, expected in hashes.items(): assert digest(ROOT/name)==expected, name
        report.update(completed=True, reason=reason, at=now(), verified_input_hashes=len(hashes))
        write(OUT/'RESULTS.json', report)
        write(OUT/'STATUS.json', dict(status='complete', at=now(), reason=reason,
            completed_episodes=report['new_physical_episodes']))

    def safe(row):
        return row['safe_success'] and row['protocol_valid'] and not any(row[k] for k in ['collisions', 'false_arrival', 'physical_guards'])

    def retained(row, old):
        key = 'post_release_seconds' if row.get('recovery_test') else 'seconds'
        return safe(row) and all(row[k]<=old[k]*1.2+1e-8 for k in [key,'travel_m'])

    for item in specifications:
        pair = []
        for variant in ['control','hybrid']:
            stem = item['spec']['id']+'_'+variant
            job_path = OUT/'jobs'/(stem+'.json')
            result_path = OUT/'jobs'/(stem+'_RESULT.json')
            write(job_path, dict(spec=item['spec'], recovery=item['recovery'], variant=variant,
                bundle=str(BUNDLE), trace=str(OUT/'physical'/(stem+'.npz')), output=str(result_path)))
            write(OUT/'STATUS.json', dict(status='physical_case', at=now(),
                case_id=item['spec']['id'], variant=variant, completed_episodes=report['new_physical_episodes']))
            log = OUT/'jobs'/(stem+'.log')
            with log.open('w') as stream:
                worker = subprocess.Popen([sys.executable,'-u',str(HERE/'physical_worker.py'),str(job_path)],
                    stdout=stream, stderr=subprocess.STDOUT, cwd=ROOT)
                write(OUT/'ACTIVE_WORKER.json', dict(pid=worker.pid, job=str(job_path.relative_to(ROOT)),
                    log=str(log.relative_to(ROOT))))
                status = worker.wait()
            if status or not result_path.exists():
                raise RuntimeError(f'Physical worker failed with status {status}: {log.relative_to(ROOT)}')
            row = read(result_path)
            pair.append(row)
            report['rows'].append(row)
            report['new_physical_episodes'] += 1
            report['at'] = now()
            write(OUT/'RESULTS.json', report)
            print(json.dumps(dict(case_id=row['case_id'], variant=variant,
                safe_success=row['safe_success'], seconds=row['seconds'], travel_m=row['travel_m'])), flush=True)
            if any(row[k] for k in ['collisions','false_arrival','physical_guards']) or not row['protocol_valid']:
                finish('Unsafe or protocol-invalid development episode; stop expansion')
                return
            if item['reference']['safe_success'] and not retained(row,item['reference']):
                finish('Prior original success or efficiency lost; stop expansion')
                return
        control, hybrid = pair
        comparison = dict(case_id=item['spec']['id'], recovery=item['recovery'],
            control_safe_success=safe(control), hybrid_safe_success=safe(hybrid),
            newly_safe=not safe(control) and safe(hybrid),
            original_success_retained=not item['reference']['safe_success'] or retained(hybrid,item['reference']),
            paired_control_success_retained=not safe(control) or retained(hybrid,control))
        report['pairs'].append(comparison)
        if not comparison['paired_control_success_retained']:
            finish('Memory overlay lost matched control success or efficiency; stop expansion')
            return
    report['all_five_hybrid_safe'] = all(pair['hybrid_safe_success'] for pair in report['pairs'])
    report['newly_safe_case_ids'] = [pair['case_id'] for pair in report['pairs'] if pair['newly_safe']]
    finish('Bounded memory comparison complete; original full retention, late recovery and prospective gates remain required')


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        if OUT.exists(): write(OUT/'FAILURE.json', dict(at=now(), error=repr(error), no_auto_restart=True))
        raise
