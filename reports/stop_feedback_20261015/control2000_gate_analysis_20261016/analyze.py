"""Pair the complete 18-task control gate and inspect actual issued waiting."""
from datetime import datetime
import hashlib
import json
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

from experiments.audit_decision_waits import measure
from experiments.paired_navigation_gate import compare
from neural_navigation.experiment_budget import ExperimentBudget
from neural_navigation.goal_pose import contract


ROOT=Path(__file__).resolve().parents[3]
OUT=Path(__file__).resolve().parent
BASE=ROOT/'reports/stop_feedback_20261015'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    if (OUT/'RESULTS.json').exists():
        raise ValueError('Do not overwrite a completed gate analysis')
    budget=ExperimentBudget(.05,memory_mib=1024,directory=OUT)
    folders=dict(reference=BASE/'w11_recovery_bc800_navigation_20261016',
                 control=BASE/'w11_control2000_navigation_20261016')
    reports={};metrics={};inputs={}
    for variant,folder in folders.items():
        source=folder/'RESULTS.json';report=json.loads(source.read_text())
        if not report['completed'] or len(report['rows'])!=18:
            raise ValueError('Two complete18-task reports are required')
        reports[variant]=report
        inputs[str(source.relative_to(ROOT))]=digest(source)
        cases={c['id']:c for c in json.loads((folder/'cases.json').read_text())}
        rows=[]
        for row in report['rows']:
            case=cases[row['case_id']]
            path=folder/(row['case_id']+'_'+row['variant']+'.npz')
            if digest(path)!=row['trace_sha256']:
                raise ValueError('Complete physical trace changed: '+str(path))
            directional=len(case['goal'])==3
            with np.load(path,allow_pickle=False) as trace:
                metric=measure(trace,row['seconds'],contract()['position_tolerance_m'] if directional else .2,
                    yaw_tolerance=contract()['yaw_tolerance_rad'] if directional else None)
                if not row['safe_success']:
                    available=trace['masks'][:,1:].any(-1)
                    idle=np.max(abs(trace['commands']),axis=-1)<1e-8
                    masks=trace['masks'].astype(bool)
                    selected=(available & idle)
                    patterns={}
                    for mask in masks[selected]:
                        signature=','.join(map(str,np.flatnonzero(mask)))
                        patterns[signature]=patterns.get(signature,0)+1
                    route=trace['route_diagnostics']
                    metric['failed_state_details']=dict(
                        final_legal_actions=np.flatnonzero(masks[-1]).tolist(),
                        legal_mask_patterns_during_issued_idle=patterns,
                        final_route_points=int(route[-1,1]),
                        final_route_endpoint=route[-1,2:4].tolist(),
                        final_route_endpoint_goal_error_m=float(np.linalg.norm(route[-1,2:4]-case['goal'][:2])),
                        last_route_reason=str(trace['route_reasons'][-1]),
                        final_position_error_m=float(trace['labels'][-1,7]))
            rows.append(dict(case_id=row['case_id'],safe_success=row['safe_success'],
                collisions=row['collisions'],timeout=row['timeout'],directional=directional,
                **metric))
            inputs[str(path.relative_to(ROOT))]=row['trace_sha256']
            budget.tick()
        metrics[variant]=rows
    reference=reports['reference']['rows'];control=reports['control']['rows']
    original_ids={row['case_id'] for row in reference if not row['case_id'].startswith('recovery_holdout_')}
    if len(original_ids)!=14:
        raise ValueError('Exactly14 original regression tasks required')
    summary={}
    for variant,rows in metrics.items():
        summary[variant]=dict(tasks=len(rows),safe_successes=sum(r['safe_success'] for r in rows),
            collisions=sum(r['collisions'] for r in rows),timeouts=sum(r['timeout'] for r in rows),
            original14_successes=sum(r['safe_success'] for r in rows if r['case_id'] in original_ids),
            directional_successes=sum(r['safe_success'] for r in rows if r['directional']),
            unfinished_executed_idle_with_motion_s=sum(r['unfinished_task_executed_idle_with_motion_s'] for r in rows),
            only_stop_s=sum(r['only_stop_s'] for r in rows))
    result=dict(completed=True,at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
        all18_results=compare(reference,control),
        original14_results=compare([r for r in reference if r['case_id'] in original_ids],
                                  [r for r in control if r['case_id'] in original_ids]),
        summary=summary,wait_metrics=metrics,inputs_sha256=inputs,
        source_sha256=digest(Path(__file__)),resource=budget.statistics(),
        additional_environment_steps=0,current_policy_or_runtime_changed=False,
        scope='Complete control18-task development gate only; other predeclared35-case outcomes and retention variant remain separate.',
        limitations=['A legal alternative does not imply a useful safe maneuver.',
            'Higher total success cannot compensate for losing an original successful mission.',
            'Aggregate waiting is affected by which missions fail and how long they remain unfinished.',
            'Physical truth and route endpoints are logged metrics only; not new policy inputs.'],
        goal_achieved=False,deployment_approved=False)
    (OUT/'RESULTS.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(completed=True,summary=summary,
        lost_original_successes=result['original14_results']['lost_successes'],
        gained_successes=result['all18_results']['gained_successes'],
        failed_control=[r for r in metrics['control'] if not r['safe_success']],
        resource=result['resource']),indent=2),flush=True)


if __name__=='__main__':
    main()
