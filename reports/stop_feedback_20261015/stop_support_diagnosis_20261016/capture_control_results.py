"""Freeze all35 predeclared control rows without aggregating correlated tasks."""
from datetime import datetime
import hashlib
import json
from pathlib import Path
from zoneinfo import ZoneInfo

from neural_navigation.experiment_budget import ExperimentBudget


ROOT=Path(__file__).resolve().parents[3]
OUT=Path(__file__).resolve().parent


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    destination=OUT/'CONTROL35_STATUS.json'
    if destination.exists():
        raise ValueError('Do not replace completed control evidence')
    budget=ExperimentBudget(.05,memory_mib=1024,directory=OUT)
    groups={};inputs={}
    for label,suffix,expected in [('full_reset','navigation',18),('original_prefix','prefixes',2),
                                  ('related_w11_history','w11',3),('extra_condition','fresh',12)]:
        folder=OUT.parent/('w11_control2000_'+suffix+'_20261016')
        p=folder/'RESULTS.json';report=json.loads(p.read_text())
        if not report['completed'] or len(report['rows'])!=expected:
            raise ValueError('Every predeclared control group must be complete')
        for row in report['rows']:
            tail='' if row.get('recovery_test') else '_'+row['variant']
            trace=folder/(row['case_id']+tail+'.npz')
            if digest(trace)!=row['trace_sha256']:
                raise ValueError('Complete physical evidence changed')
            inputs[str(trace.relative_to(ROOT))]=row['trace_sha256']
        inputs[str(p.relative_to(ROOT))]=digest(p)
        groups[label]=dict(rows=expected,safe_successes=sum(r['safe_success'] for r in report['rows']),
            collisions=sum(r['collisions'] for r in report['rows']),
            timeouts=sum(r['timeout'] for r in report['rows']),
            false_arrivals=sum(r['false_arrival'] for r in report['rows']),
            incomplete_case_ids=[r['case_id'] for r in report['rows'] if not r['safe_success']],
            source=str(p.relative_to(ROOT)),source_sha256=digest(p))
        budget.tick()
    preservation=OUT.parent/'control2000_gate_analysis_20261016/RESULTS.json'
    original=json.loads(preservation.read_text())
    if not original['completed']:
        raise ValueError('Original-success preservation analysis required')
    inputs[str(preservation.relative_to(ROOT))]=digest(preservation)
    checkpoint=ROOT/'models/stop_feedback_20261015/w11_control2000_s231015/policy.zip'
    inputs[str(checkpoint.relative_to(ROOT))]=digest(checkpoint)
    budget.tick()
    result=dict(completed=True,at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
        separate_group_results=groups,number_of_completed_rows=35,
        original_success_preservation=original['original14_results'],
        aggregated_independent_success_rate=None,
        limitations=['The35 rows mix full resets,correlated historical prefixes and same-map extra conditions.',
            'Do not turn this mixed total into an independent navigation or unseen-map success rate.',
            'Retained variant is still being physically evaluated; this is not a complete two-model comparison.'],
        additional_environment_steps=0,inputs_sha256=inputs,
        source_sha256=digest(Path(__file__)),resource=budget.statistics(),
        goal_achieved=False,deployment_approved=False)
    destination.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(completed=True,groups=groups,additional_environment_steps=0),indent=2),flush=True)


if __name__=='__main__':
    main()
