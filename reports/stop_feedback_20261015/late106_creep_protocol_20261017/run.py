"""Replace the memory-failed beam with bounded sequential fresh processes."""
from datetime import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
BASE = HERE.parent
OUT = BASE / 'late106_creep_teacher_20261017'
OWN_UNIT = 'rl-late106-creep-teacher-20261017.service'


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(name, **value):
    value.update(at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
        new_ppo_steps=0, new_training_updates=0, goal_achieved=False, deployment_approved=False)
    path = OUT / name
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def main():
    assert not OUT.exists()
    for unit, invocation, code in [
        ('rl-full-navigation-joint-linear-20261017.service', '4410ede38b4a4369a6c1dfe388ab40fd', '0'),
        ('rl-full-navigation-joint-linear-mechanism-20261017.service', '12f8bee1621e4efda6da7f4a2bcbc002', '0'),
        ('rl-late106-joint-beam-20261017.service', '0c167cc622a54f3780596b824afd4eb8', '1')]:
        result = subprocess.run(['systemctl', '--user', 'show', unit,
            '-p', 'MainPID', '-p', 'ActiveState', '-p', 'InvocationID', '-p', 'ExecMainStatus'],
            capture_output=True, text=True, check=True)
        state = dict(line.split('=', 1) for line in result.stdout.splitlines() if '=' in line)
        assert state['InvocationID'] in ('', invocation), state
        assert state['MainPID'] == '0' and state['ActiveState'] in ('inactive', 'failed')
        assert state['ExecMainStatus'] == code, state
    running = subprocess.run(['systemctl', '--user', 'list-units', '--type=service',
        '--state=running', '--no-legend', '--no-pager', 'rl-*'],
        capture_output=True, text=True, check=True)
    assert all(line.split()[0] == OWN_UNIT for line in running.stdout.splitlines() if line.strip())
    validation_path = BASE / 'full_navigation_joint_linear_validation_20261017/RESULTS.json'
    validation = read(validation_path)
    assert validation['completed'] and validation['all_fixed28_passed']
    assert validation['model'] == 'models/stop_feedback_20261015/full_navigation_joint_linear_s231017'
    failure = next(r for r in validation['groups']['hard3']['rows'] if r['case_id'] == 'w11_training_stall_t106')
    assert failure['timeout'] and not any(failure[k] for k in ('collisions', 'physical_guards', 'false_arrival'))
    old_failure = BASE / 'late106_joint_beam_20261017/FAILURE.json'
    assert 'conservative RSS budget' in read(old_failure)['error']
    folder = BASE / 'full_navigation_joint_linear_20261017'
    paths = [Path(__file__), HERE / 'worker.py', validation_path, old_failure,
        folder / 'FIXED28_AUDIT.json', folder / 'KNOWN6_PHYSICAL_AUDIT.json',
        folder / 'NEURAL_MECHANISM.json', folder / 'CURRENT106_FAILURE_AUDIT.json',
        folder / 'FIT_RESULTS.json']
    for name in ('FIXED28_AUDIT.json', 'KNOWN6_PHYSICAL_AUDIT.json', 'NEURAL_MECHANISM.json', 'CURRENT106_FAILURE_AUDIT.json'):
        value = read(folder / name)
        assert value['completed'] and value['model'] == validation['model']
    tasks_path = BASE / 'context_feedback_longfit_protocol_20261016/tasks.json'
    spec = read(tasks_path)['actual106']
    source = ROOT / failure['trace_path']
    assert digest(source) == failure['trace_sha256']
    bundle = ROOT / validation['model'] / 'portable'
    paths += [tasks_path, source, bundle / 'manifest.json']
    paths += [bundle / name for name in read(bundle / 'manifest.json')['files']]
    for package in ('experiments', 'neural_navigation', 'navrl'):
        paths += list((ROOT / 'training' / package).rglob('*.py'))
    paths += [p for p in (ROOT / 'training' / spec['case']['scene']).iterdir() if p.is_file()]
    hashes = {str(p.relative_to(ROOT)): digest(p) for p in set(paths)}
    OUT.mkdir()
    write('PLAN.json', inputs=hashes, source_spec=spec,
        replaces_memory_failed_beam_only=True, no_snapshot_or_branch_copy=True,
        targets_clockwise_radians=[.4, .8, 1.2], forced_turn_max_seconds=6.,
        sensor_masked_turn_priority=[9, 5, 7], forward_m_s_range=[0., .05],
        maximum_probe_processes=3, maximum_winner_replays=1,
        maximum_new_physical_steps=2800, original_prefix_frames=530,
        original_deadline_s=140., original_12actions_body4cm_unknown_blind_memory_dynamics_unchanged=True,
        complete_fresh_process_winner_replay_required=True,
        prior_snapshot_parity_replaced_by_fresh_process_exact_original_prefix_and_full_replay=True,
        offline_teacher_not_learned_success=True, failed_probes_not_infeasibility=True,
        no_automatic_training=True, soft_cpu_cores=.35, hard_memory_mib=3072)
    rows = []

    def run(name, mode, target=None, expected=None):
        for path, sha in hashes.items():
            assert digest(ROOT / path) == sha, path
        job = dict(mode=mode, target_rad=target, case=spec['case'], bundle=str(bundle.relative_to(ROOT)),
            source_trace=str(source.relative_to(ROOT)), inputs=hashes,
            directory=str((OUT / name).relative_to(ROOT)))
        if expected is not None:
            job.update(expected_trace=expected['trace_path'], expected_sha256=expected['trace_sha256'])
        job_path = OUT / (name + '.json')
        job_path.write_text(json.dumps(job, ensure_ascii=False, indent=2) + '\n')
        with (OUT / (name + '.log')).open('x') as stream:
            child = subprocess.Popen([sys.executable, '-u', str(HERE / 'worker.py'), str(job_path)],
                cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
            stat = Path('/proc') / str(child.pid) / 'stat'
            ticks = stat.read_text().rsplit(')', 1)[1].split()[19]
            write('STATUS.json', stage=name, child_pid=child.pid, child_start_ticks=ticks,
                completed_jobs=len(rows), one_model_or_physics_worker=True)
            code = child.wait()
        assert code == 0, (name, code, 'No automatic restart')
        result = read(OUT / name / 'RESULTS.json')
        rows.append(result)
        assert sum(row['new_physical_steps'] for row in rows) <= 2800
        write('INTERIM.json', completed_jobs=len(rows), rows=rows)
        return result

    teacher = None
    for index, angle in enumerate((.4, .8, 1.2)):
        row = run('probe_' + str(index), 'probe', angle)
        if row['safe_teacher_success']:
            teacher = row
            break
    replay = run('winner_replay', 'replay', expected=teacher) if teacher is not None else None
    if replay is not None:
        assert replay['safe_teacher_success'] and replay['complete_fresh_process_teacher_replay_exact']
    write('RESULTS.json', completed=True, safe_teacher_found=teacher is not None,
        complete_fresh_replay_passed=replay is not None,
        rows=rows, new_physical_steps=sum(row['new_physical_steps'] for row in rows),
        no_learned_success_or_infeasibility_claim=True,
        no_automatic_training_admission=True, no_deployment_replacement=True)
    write('STATUS.json', stage='complete', safe_teacher_found=teacher is not None)


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        if OUT.exists():
            write('FAILURE.json', error=repr(error), no_auto_restart=True)
        raise
