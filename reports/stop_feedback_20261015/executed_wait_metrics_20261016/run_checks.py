"""Preserve historical metrics and check issued-zero counts on physical traces."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

from experiments.audit_decision_waits import measure
from neural_navigation.goal_pose import contract


ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
BASE = Path('reports/stop_feedback_20261015')


def main():
    os.chdir(ROOT)
    plan = json.loads((OUT/'PLAN.json').read_text())
    path = Path(plan['legacy_auditor'])
    if hashlib.sha256(path.read_bytes()).hexdigest() != plan['legacy_auditor_sha256']:
        raise ValueError('Frozen legacy metric implementation changed')
    spec = importlib.util.spec_from_file_location('frozen_wait_auditor', path)
    legacy = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(legacy)
    test = subprocess.run([sys.executable, '-m', 'unittest', '-v',
        'experiments.test_audit_decision_waits'], capture_output=True, text=True)
    (OUT/'tests.txt').write_text(test.stdout+test.stderr)
    if test.returncode:
        raise ValueError('Executed wait implementation checks failed')
    sources = [
        ('proactive_augmented_bc800_gate_20261016', None, False),
        ('w11_recovery_bc800_navigation_20261016', 1, False),
        ('w11_recovery_bc800_early_20261016', None, True),
    ]
    rows = []
    for folder, limit, recovery in sources:
        directory = BASE/folder
        report_path = directory/'RESULTS.json'
        raw = ((OUT/'CANDIDATE_PARTIAL.json').read_bytes()
               if limit is not None else report_path.read_bytes())
        report = json.loads(raw)
        cases = ({s['id']:s['case'] for s in json.loads(
            (BASE/'w11_success_augmented_protocol_20261016/w11_t90_evaluation.json').read_text())}
            if recovery else {c['id']:c for c in json.loads(
                ((BASE/'w11_success_augmented_protocol_20261016/gate_cases.json')
                 if limit is not None else directory/'cases.json').read_text())})
        selected = report['rows'] if limit is None else report['rows'][:limit]
        for row in selected:
            filename = row['case_id']+('' if recovery else '_'+row['variant'])+'.npz'
            trace_path = directory/filename
            if hashlib.sha256(trace_path.read_bytes()).hexdigest() != row['trace_sha256']:
                raise ValueError('Finished physical trace changed')
            directional = len(cases[row['case_id']]['goal']) == 3
            tolerance = contract()['position_tolerance_m'] if directional else .2
            yaw = contract()['yaw_tolerance_rad'] if directional else None
            with np.load(trace_path, allow_pickle=False) as trace:
                old = legacy.measure(trace, row['seconds'], tolerance, yaw_tolerance=yaw)
                new = measure(trace, row['seconds'], tolerance, yaw_tolerance=yaw)
                if old != {key:new[key] for key in old}:
                    raise ValueError('A historical wait metric changed')
                post = None
                if recovery:
                    n = row['frozen_prefix_frames']
                    suffix = {name:trace[name][n:] for name in ['labels','masks','choices','commands']}
                    post = measure(suffix, row['seconds'], tolerance, yaw_tolerance=yaw)
            rows.append(dict(source=folder, case_id=row['case_id'],
                source_report_completed=report['completed'],
                source_report_at_read_sha256=hashlib.sha256(raw).hexdigest(),
                trace_sha256=row['trace_sha256'], safe_success=row['safe_success'],
                legacy_metric_keys_preserved=len(old), metrics=new, post_release_metrics=post))
    if len(rows) != 20:
        raise ValueError('All 18 old tasks, one new full task and one recovery are required')
    result = dict(completed=True, implementation_tests=8,
        unchanged_legacy_metrics_on_physical_traces=len(rows), rows=rows,
        no_runtime_or_training_change=True, no_new_navigation_runs=True,
        exact_parent_physical_recovery_result_still_pending=True,
        full_candidate_gate_still_pending=True,
        goal_achieved=False, deployment_approved=False)
    (OUT/'RESULTS.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(completed=True, checked_physical_traces=len(rows),
        implementation_tests=8, legacy_metrics_preserved=True,
        learned_recovery_post_release=rows[-1]['post_release_metrics'])), flush=True)


if __name__ == '__main__':
    main()
