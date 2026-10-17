"""Two missing frozen recovery checks, without resuming training or promotion."""
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
BASE = HERE.parent
OUT = BASE / 'deployment_acceptance_supplement_20261017'
MODEL = ROOT / 'models/stop_feedback_20261015/full_navigation_joint_linear_s231017'
WORKER = BASE / 'full_navigation_joint_linear_20261017/worker.py'
CHILD = None


def now():
    return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(path)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stop(signum, frame):
    if CHILD is not None and CHILD.poll() is None:
        CHILD.terminate()
    raise SystemExit(128 + signum)


def safe(row):
    return (row['safe_success'] and row['protocol_valid'] and
            not any(row[k] for k in ['collisions', 'false_arrival', 'physical_guards']))


def main():
    global CHILD
    assert not OUT.exists(), 'Never overwrite an earlier evaluation'
    OUT.mkdir()
    (OUT / 'jobs').mkdir()
    (OUT / 'physical').mkdir()
    prior = read(BASE / 'full_navigation_joint_linear_validation_20261017/RESULTS.json')
    original_plan = read(BASE / 'full_navigation_joint_linear_validation_20261017/PLAN.json')
    hashes = original_plan['inputs']
    additional = [Path(__file__), HERE / 'audit.py']
    pinned = {str(p.relative_to(ROOT)): digest(p) for p in additional}

    def verify():
        for name, expected in {**hashes, **pinned}.items():
            assert digest(ROOT / name) == expected, name

    verify()
    active = subprocess.run(
        ['systemctl', '--user', 'list-units', 'rl-*', '--state=active,activating,deactivating',
         '--no-pager', '--plain', '--no-legend'], capture_output=True, text=True, check=True)
    units = [line.split()[0] for line in active.stdout.splitlines() if line.strip()]
    assert units == ['rl-deployment-acceptance-supplement-20261017.service'], units
    state = subprocess.run(['systemctl', '--user', 'show', units[0], '-p', 'MainPID',
                            '-p', 'InvocationID', '-p', 'MemoryMax', '-p', 'Nice'],
                           capture_output=True, text=True, check=True)
    assert f'MainPID={os.getpid()}\n' in state.stdout
    parent = read(BASE / 'normal_suite_wait_validation_v2_20261016/RESULTS.json')
    tasks = read(BASE / 'context_feedback_longfit_protocol_20261016/tasks.json')
    refs = {r['case_id']: r for r in parent['groups']['hard3']['rows']}
    ids = ['legal_wait_parent_t95', 'legal_wait_parent_t90']
    assert prior['groups']['hard3']['rows'][0]['case_id'] == 'w11_training_stall_t106'
    assert not safe(prior['groups']['hard3']['rows'][0])
    plan = dict(at=now(), user_request='直接进行部署验收需要多长时间，看看是否可以改善？',
                scope='Only two previously unexecuted frozen recovery checks; no optimization resume',
                case_ids=ids, model=str(MODEL.relative_to(ROOT)), original_inputs=hashes,
                protocol_inputs=pinned, original_formal_report=str((BASE / 'full_navigation_joint_linear_validation_20261017/RESULTS.json').relative_to(ROOT)),
                maximum_new_physical_episodes=2, maximum_physics_steps_per_episode=700,
                soft_cpu_cores=.35, memory_hard_limit_mib=3072, nice=19,
                actual_service_state=state.stdout, expected_minutes=[10, 15],
                prior35_actual_wall_minutes=51.6508, new_ppo_steps=0, new_training_updates=0,
                no_prospective_fresh8_expansion=True, no_model_replacement=True,
                original_body_margin_actions_unknown_blind_memory_dynamics_deadlines_retained=True,
                goal_achieved=False, deployment_approved=False)
    write(OUT / 'PLAN.json', plan)
    report = dict(completed=False, at=now(), model=plan['model'], rows=[], gates={},
                  new_ppo_steps=0, new_training_updates=0, original_deployment_replaced=False,
                  prior106_failure_retained=True, fresh8_executed=False,
                  goal_achieved=False, deployment_approved=False)
    write(OUT / 'RESULTS.json', report)
    for case_id in ids:
        verify()
        ref = refs[case_id]
        assert digest(ROOT / ref['trace_path']) == ref['trace_sha256']
        spec = dict(id=case_id, case=tasks['actual106']['case'],
                    release_s=ref['release_sensor_timestamp_s'], trace=ref['trace_path'])
        job_path = OUT / 'jobs' / (case_id + '.json')
        result_path = OUT / 'jobs' / (case_id + '_RESULT.json')
        job = dict(spec=spec, recovery=True, variant='candidate', bundle=str(MODEL / 'portable'),
                   trace=str(OUT / 'physical' / (case_id + '.npz')), output=str(result_path))
        write(job_path, job)
        write(OUT / 'STATUS.json', dict(at=now(), stage='bounded_physical_recovery_check',
                                      case_id=case_id, completed_episodes=len(report['rows']),
                                      training_running=False, deployment_approved=False))
        with (OUT / 'jobs' / (case_id + '.log')).open('w') as stream:
            CHILD = subprocess.Popen([sys.executable, '-u', str(WORKER), str(job_path)],
                                     cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
            write(OUT / 'ACTIVE_WORKER.json', dict(at=now(), pid=CHILD.pid,
                                                 case_id=case_id, job=str(job_path.relative_to(ROOT))))
            code = CHILD.wait(timeout=900)
        assert code == 0 and result_path.exists(), (case_id, code)
        row = read(result_path)
        passed = safe(row)
        if ref['safe_success']:
            passed = passed and all(row[k] <= ref[k] * 1.2 + 1e-8
                                    for k in ['post_release_seconds', 'travel_m'])
        report['rows'].append(row)
        report['gates'][case_id] = dict(passed=passed, prior_safe_success=ref['safe_success'],
                                      prior_post_release_seconds=ref['post_release_seconds'],
                                      prior_travel_m=ref['travel_m'])
        report['at'] = now()
        write(OUT / 'RESULTS.json', report)
        print(json.dumps(dict(case_id=case_id, safe_success=row['safe_success'],
                              passed=passed, seconds=row['seconds'],
                              post_release_seconds=row['post_release_seconds'])), flush=True)
    write(OUT / 'STATUS.json', dict(at=now(), stage='independent_saved_trace_audit',
                                  completed_episodes=2, training_running=False))
    CHILD = subprocess.Popen([sys.executable, '-u', str(HERE / 'audit.py')], cwd=ROOT)
    assert CHILD.wait(timeout=180) == 0
    audit = read(OUT / 'TRACE_AUDIT.json')
    assert audit['completed'] and audit['prefix_sensor_and_physical_parity_verified']
    verify()
    report.update(completed=True, at=now(), completed_new_physical_episodes=len(report['rows']),
                  all_supplement_gates_passed=all(g['passed'] for g in report['gates'].values()),
                  prefix_audit=str((OUT / 'TRACE_AUDIT.json').relative_to(ROOT)),
                  verified_original_inputs=len(hashes), verified_protocol_inputs=len(pinned),
                  reason='Bounded supplement finished; original106 failure still prevents full acceptance; no training or deployment changes')
    write(OUT / 'RESULTS.json', report)
    write(OUT / 'STATUS.json', dict(at=now(), stage='complete', completed_episodes=2,
                                  training_running=False, deployment_approved=False,
                                  original106_unresolved=True))
    (OUT / 'ACTIVE_WORKER.json').unlink(missing_ok=True)


if __name__ == '__main__':
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        main()
    except BaseException as error:
        if CHILD is not None and CHILD.poll() is None:
            CHILD.terminate()
            try:
                CHILD.wait(timeout=10)
            except subprocess.TimeoutExpired:
                CHILD.kill()
                CHILD.wait()
        if OUT.exists():
            write(OUT / 'FAILURE.json', dict(at=now(), error=repr(error),
                                           interrupted=isinstance(error, SystemExit),
                                           no_auto_restart=True, deployment_approved=False))
        raise
