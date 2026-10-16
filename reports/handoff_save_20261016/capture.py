"""Capture finished evaluation rows; running source-PC jobs are not migrated."""
from datetime import datetime
import hashlib
import json
from pathlib import Path
import subprocess
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
BASE = ROOT / 'reports/stop_feedback_20261015'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')


def copy_bytes(source, target, expected=None):
    data = source.read_bytes()
    if expected is not None and sha(data) != expected:
        raise ValueError('Source changed or trace is incomplete: ' + str(source))
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)
    return sha(data)


def main():
    if (OUT / 'SNAPSHOT.json').exists():
        raise FileExistsError('This is a fixed handoff; capture into a new directory.')
    started = datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()
    reports = {}
    for name, expected in [
        ('w11_control2000_navigation_20261016', 18),
        ('w11_control2000_prefixes_20261016', 2),
        ('w11_control2000_w11_20261016', 3),
        ('w11_control2000_fresh_20261016', 12),
        ('w11_retained2000_navigation_20261016', 18),
        ('w11_retained2000_prefixes_20261016', 2),
        ('w11_retained2000_w11_20261016', 3),
        ('w11_retained2000_fresh_20261016', 12),
        ('early_transfer_diagnostic_navigation_20261016', 6),
        ('early_transfer_diagnostic_reference_20261016', 3),
        ('reference_control_wait_t32_20261016', 1),
    ]:
        source = BASE / name
        path = source / 'RESULTS.json'
        if not path.exists():
            reports[name] = dict(started=False, expected_rows=expected, finished_rows=0)
            continue
        raw = path.read_bytes()
        report = json.loads(raw)
        rows = report['rows']
        if report['completed'] and len(rows) != expected:
            raise ValueError('Unexpected number of complete rows: ' + name)
        target = OUT / 'validation' / name
        target.mkdir(parents=True)
        (target / 'RESULTS.json').write_bytes(raw)
        for filename in ['cases.json', 'specifications.json', 'PROVENANCE.json', 'SUMMARY.json']:
            if (source / filename).is_file():
                copy_bytes(source / filename, target / filename)
        traces = {sha(p.read_bytes()): p for p in source.glob('*.npz')}
        trace_files = {}
        for row in rows:
            digest = row['trace_sha256']
            original = traces[digest]
            copy_bytes(original, target / original.name, digest)
            trace_files[original.name] = digest
        reports[name] = dict(started=True, completed=report['completed'],
            expected_rows=expected, finished_rows=len(rows),
            safe=sum(bool(r['safe_success']) for r in rows),
            collisions=sum(int(r['collisions']) for r in rows),
            timeouts=sum(bool(r['timeout']) for r in rows),
            report=str((target / 'RESULTS.json').relative_to(ROOT)),
            report_sha256=sha(raw), trace_files=trace_files,
            variants=sorted({r['variant'] for r in rows}))

    units = []
    for unit in ['rl-w11-retention-pair-20261016.service',
                 'rl-post-aux-fixed-20261016.service',
                 'rl-reference-wait-t32-20261016.service']:
        raw = subprocess.check_output(['systemctl', '--user', 'show', unit,
            '-p', 'Id', '-p', 'ActiveState', '-p', 'SubState', '-p', 'MainPID',
            '-p', 'InvocationID', '-p', 'Result'], text=True)
        units.append(dict(line.split('=', 1) for line in raw.splitlines() if '=' in line))
    for folder, filenames in [
        ('retention_pair_20261016', ['STATUS.json']),
        ('post_aux_diagnostics_fixed_20261016', ['STATUS.json']),
        ('reference_wait_t32_protocol_20261016', ['STATUS.json']),
        ('early_decision_transfer_20261016', ['QUICK_STATUS.json']),
    ]:
        for filename in filenames:
            if (BASE / folder / filename).is_file():
                copy_bytes(BASE / folder / filename, OUT / 'source_status' / folder / filename)

    models = {}
    for name, folder, policy in [
        ('deployment', 'models/quality_recovery_20261014', 'policy_2000.zip'),
        ('reference_bc', 'models/stop_feedback_20261015/w11_recovery_bc800_s231015', 'policy_bc.zip'),
        ('control_2k', 'models/stop_feedback_20261015/w11_control2000_s231015', 'policy.zip'),
        ('retained_2k', 'models/stop_feedback_20261015/w11_retained2000_s231015', 'policy.zip'),
        ('diagnostic_bc', 'models/stop_feedback_20261015/early_transfer_bc800_s231015', 'policy_bc.zip'),
    ]:
        models[name] = dict(folder=folder, checkpoint=policy,
            sha256=sha((ROOT / folder / policy).read_bytes()),
            additional_ppo_steps=2000 if name in ('control_2k', 'retained_2k') else 0,
            approved_as_new_deployment=False)

    write(OUT / 'SNAPSHOT.json', dict(schema='agv_handoff_snapshot_v14', branch='save',
        capture_started_at=started,
        captured_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
        source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        previous_handoff_branch='dev',
        models=models, validation=reports, source_units=units,
        original_deployment_replaced=False, local_jobs_migrate=False,
        exact_physical_rollout_resume_supported=False,
        goal_achieved=False, deployment_approved=False,
        remaining=[
            'Finish the retained-model fixed evaluations and compare original successes, not only total arrivals.',
            'Finish the diagnostic BC/reference tasks and the t32 reference continuation; no result is assumed in advance.',
            'The control 2k regressed on the original W11 success and all three historical W11 recoveries.',
            'The retained 2k recovered the full-start W11, but it has other failures; this is not deployment acceptance.',
            'Long-wait useful demonstrations and actual learnable-state exposure remain insufficient.',
            'Localization networks/router are frozen in recovery training; real localization and Orin latency remain unaccepted.',
            'Source-PC services and post-snapshot outputs are not transferred by Git.',
        ]))
    print(json.dumps(dict(captured_at=json.loads((OUT/'SNAPSHOT.json').read_text())['captured_at'],
        validation={name:{k:v for k,v in item.items() if k not in ('trace_files', 'report_sha256')}
                    for name,item in reports.items()}), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
