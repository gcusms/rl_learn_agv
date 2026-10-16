"""Pair task outcomes before considering speed or distance for promotion."""
import math


def compare(before,after):
    def index(rows):
        result={}
        for row in rows:
            key=row['case_id']
            if key in result:raise ValueError('Duplicate paired task')
            for field in ('seconds','travel_m'):
                if not math.isfinite(row[field]) or row[field]<0:raise ValueError('Finite nonnegative physical metrics required')
            result[key]=row
        return result
    old,new=index(before),index(after)
    if not old or old.keys()!=new.keys():raise ValueError('Exact nonempty paired task identities required')
    common=[key for key in old if old[key]['safe_success'] and new[key]['safe_success']]
    lost=[key for key in old if old[key]['safe_success'] and not new[key]['safe_success']]
    gained=[key for key in old if not old[key]['safe_success'] and new[key]['safe_success']]
    # Fixed development gate, not an assertion of optimality or significance.
    limits=dict(total_travel_increase=.02,total_time_increase=.05,
        per_task_travel_increase=.20,per_task_time_increase=.20)
    def increase(a,b):
        return b/a-1. if a>1e-9 else (0. if b<=1e-9 else None)
    pairs=[dict(case_id=key,
        travel_relative_change=increase(old[key]['travel_m'],new[key]['travel_m']),
        time_relative_change=increase(old[key]['seconds'],new[key]['seconds'])) for key in common]
    travel_before=sum(old[key]['travel_m'] for key in common);travel_after=sum(new[key]['travel_m'] for key in common)
    time_before=sum(old[key]['seconds'] for key in common);time_after=sum(new[key]['seconds'] for key in common)
    travel=increase(travel_before,travel_after);seconds=increase(time_before,time_after)
    within=lambda change,limit:change is not None and change<=limit+1e-9
    efficiency=bool(common and within(travel,limits['total_travel_increase']) and within(seconds,limits['total_time_increase'])
        and all(within(row['travel_relative_change'],limits['per_task_travel_increase'])
            and within(row['time_relative_change'],limits['per_task_time_increase']) for row in pairs))
    safety=not lost and all(new[key]['collisions']<=old[key]['collisions']
        and new[key]['false_arrival']<=old[key]['false_arrival'] for key in old)
    return dict(schema='paired_navigation_development_gate_v1',tasks=len(old),common_successes=len(common),
        lost_successes=lost,gained_successes=gained,safety_nonregression=bool(safety),
        common_success_travel_m=[travel_before,travel_after],common_success_seconds=[time_before,time_after],
        total_travel_relative_change=travel,total_time_relative_change=seconds,pairs=pairs,limits=limits,
        efficiency_nonregression=efficiency,all_regression_checks_passed=bool(safety and efficiency),
        larger_training_gate_passed=bool(safety and efficiency and all(new[k]['safe_success'] for k in new)),
        deployment_approved=False,scope='Reused development tasks only; no automatic promotion')
