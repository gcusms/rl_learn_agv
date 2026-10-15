"""Freeze completed artifacts and finished validation rows for the dev handoff."""
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
from zoneinfo import ZoneInfo

ROOT = Path('/home/ubuntu/rl_learn/.publish-handoff-20261015')
OUT = ROOT / 'reports/handoff_dev_20261016'
BASE = ROOT / 'reports/stop_feedback_20261015'
OUT.mkdir(exist_ok=False)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')


def copy(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    data = source.read_bytes()
    target.write_bytes(data)
    assert digest(target) == hashlib.sha256(data).hexdigest()


bc = ROOT / 'models/stop_feedback_20261015/early_transfer_bc800_s231015'
bc_status = json.loads((bc / 'status.json').read_text())
bc_complete = bc_status['status'] == 'offline_bc_complete'
if bc_complete and not (bc / 'METADATA_FINALIZATION.json').exists():
    subprocess.run([sys.executable, str(BASE / 'early_decision_transfer_20261016/finalize_metadata.py')],
                   cwd=ROOT, check=True)
if bc_complete:
    bc_result = json.loads((bc / 'RESULTS.json').read_text())
    assert bc_result['completed'] and bc_result['additional_ppo_steps'] == 0
    assert bc_result['environment_steps'] == 0
else:
    bc_result = None
write(OUT / 'OFFLINE_BC_STATUS.json', dict(status_at_capture=bc_status,
    completed_checkpoint_included=bc_complete,
    unfinished_model_directory_included=False,
    data_and_fit_protocol_included=True,
    source_jobs_do_not_migrate=True))

reports = {}
for name, expected in [
        ('w11_recovery_bc800_w11_20261016', 3),
        ('w11_recovery_parent800_early_20261016', 1),
        ('w11_recovery_bc800_fresh_20261016', 24),
        ('w11_control2000_navigation_20261016', 18),
        ('w11_retained2000_navigation_20261016', 18),
        ('rotation_escape_t106_model_20261016', 1),
        ('rotation_escape_t106_probe_20261016', 1),
        ('early_transfer_bc800_quick_20261016', 3)]:
    source = BASE / name
    path = source / 'RESULTS.json'
    if not path.exists():
        reports[name] = dict(started=False, expected_rows=expected)
        continue
    raw = path.read_bytes()
    report = json.loads(raw)
    target = OUT / 'validation' / name
    target.mkdir(parents=True)
    (target / 'RESULTS.json').write_bytes(raw)
    for filename in ['cases.json', 'specifications.json', 'PROVENANCE.json', 'SUMMARY.json']:
        if (source / filename).exists():
            copy(source / filename, target / filename)
    traces = {digest(path): path for path in source.glob('*.npz')}
    for row in report['rows']:
        trace = traces[row['trace_sha256']]
        copy(trace, target / trace.name)
    reports[name] = dict(started=True, completed=report['completed'],
        expected_rows=expected, finished_rows=len(report['rows']),
        safe=sum(bool(r['safe_success']) for r in report['rows']),
        collisions=sum(int(r['collisions']) for r in report['rows']),
        timeouts=sum(bool(r['timeout']) for r in report['rows']),
        report=str((target / 'RESULTS.json').relative_to(ROOT)),
        report_sha256=digest(target / 'RESULTS.json'),
        variants=sorted({r['variant'] for r in report['rows']}))

units = []
for unit in ['rl-w11-retention-pair-20261016.service',
             'rl-w11-paired-v2-after-validation-20261016.service',
             'rl-rotation-escape-t106-20261016.service',
             'rl-early-transfer-bc-20261016.service',
             'rl-early-transfer-quick-20261016.service']:
    raw = subprocess.check_output(['systemctl', '--user', 'show', unit,
        '-p', 'Id', '-p', 'ActiveState', '-p', 'SubState', '-p', 'MainPID',
        '-p', 'InvocationID', '-p', 'Result'], text=True)
    units.append(dict(line.split('=', 1) for line in raw.splitlines() if '=' in line))

audit_path = BASE / 'retention_pair_20261016/COMPLETED_TRAINING_AUDIT.json'
audit = json.loads(audit_path.read_text())
assert audit['completed']
models = {}
for variant in ['control', 'retained']:
    path = ROOT / 'models/stop_feedback_20261015' / ('w11_' + variant + '2000_s231015')
    status = json.loads((path / 'status.json').read_text())
    assert status['status'] == 'complete' and status['additional_ppo_steps'] == 2000
    models[variant] = dict(folder=str(path.relative_to(ROOT)), policy='policy.zip',
        policy_sha256=digest(path / 'policy.zip'), status=status,
        causal_sampling_audit='reports/stop_feedback_20261015/w11_' + variant + '2000_audit_20261016/RESULTS.json')

stamp = datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()
snapshot = dict(schema='agv_handoff_snapshot_v13', branch='dev', captured_at=stamp,
    previous_branch='work', previous_commit=subprocess.check_output(
        ['git', 'rev-parse', 'work'], cwd=ROOT, text=True).strip(),
    source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
    independent_completed_ppo=models, offline_bc_included=bc_complete,
    offline_bc_result=bc_result, validation=reports, source_units=units,
    original_deployment_model='models/quality_recovery_20261014/policy_2000.zip',
    original_deployment_sha256=digest(ROOT / 'models/quality_recovery_20261014/policy_2000.zip'),
    original_deployment_replaced=False, local_jobs_migrate=False,
    exact_physical_rollout_resume_supported=False,
    training_audit=str(audit_path.relative_to(ROOT)),
    known_remaining=[
        'Both 2k PPO models are trained; their predeclared35-task evaluations are incomplete.',
        'The retained run has one actual training collision; its training episode ratio is not a matched validation score.',
        'Neither completed 2k run changed a prior-BC stop into motion on its own recorded training states.',
        'Additional safe demonstration data have2355labels but do not prove new physical navigation capability.',
        'The rotation option changes guard masks and remains a separate diagnostic, not the standard runtime.',
        'Source systemd jobs, waits and post-cutoff outputs do not migrate through Git.',
        'No unseen-map, real-world absolute localization, AMCL superiority or Orin Nano acceptance was demonstrated.'
    ], goal_achieved=False, deployment_approved=False)
write(OUT / 'SNAPSHOT.json', snapshot)
copy(Path(__file__), OUT / 'capture.py')
for name in ['retention_pair_20261016/STATUS.json',
             'rotation_escape_t106_protocol_20261016/STATUS.json',
             'early_decision_transfer_20261016/QUICK_STATUS.json']:
    source = BASE / name
    if source.exists():
        copy(source, OUT / 'source_status' / name)

for source, target in [
        ('README.md', 'docs/handoff/WORK_README.md'),
        ('docs/handoff/FILES.json', 'docs/handoff/WORK_FILES.json'),
        ('docs/handoff/VERIFICATION.json', 'docs/handoff/WORK_VERIFICATION.json')]:
    destination = ROOT / target
    if destination.exists():
        raise ValueError('Prior handoff archive exists; refusing to overwrite it')
    copy(ROOT / source, destination)

print(json.dumps(dict(captured_at=stamp, branch='dev', models={
    k: dict(folder=v['folder'], policy_sha256=v['policy_sha256']) for k,v in models.items()},
    offline_bc_included=bc_complete, validation=reports), ensure_ascii=False, indent=2))
