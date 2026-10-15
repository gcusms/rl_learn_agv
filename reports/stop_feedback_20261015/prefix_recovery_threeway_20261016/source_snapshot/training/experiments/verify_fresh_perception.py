"""Verify fresh-perception artifacts and report matched development changes."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from experiments.verify_quality_regression import verify


def main():
    parser=argparse.ArgumentParser()
    for name in ('run','bundle','baseline','output'):parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():parser.error('Fresh output required')
    torch.set_num_threads(1)
    result=verify(args.run,args.bundle)
    source=json.loads((args.run/'SOURCE_PROVENANCE.json').read_text())
    if source.get('captured_before_launch') is not True or source.get('runtime_variant')!='per_sensor_frame_fusion':
        raise ValueError('Expected pre-launch fresh-perception provenance')
    result.update(source_captured_before_launch=True,
        limitation='48 reused development cases. Source captured before launch and unchanged; simulated arrival/collision labels come from evaluator. No semantic understanding, real hardware or sealed final acceptance claim.')
    current=json.loads((args.run/'RESULTS.json').read_text())
    baseline=json.loads((args.baseline/'RESULTS.json').read_text())
    if not baseline['completed']:raise ValueError('Completed baseline required')
    old={x['case_id']:x for x in baseline['cases']}
    if set(old)!={x['case_id'] for x in current['cases']}:raise ValueError('Different task sets')
    transitions=[];common=[]
    for x in current['cases']:
        y=old[x['case_id']]
        now_case=json.loads((args.run/(x['case_id']+'.json')).read_text())['case']
        old_case=json.loads((args.baseline/(x['case_id']+'.json')).read_text())['case']
        if now_case!=old_case:raise ValueError('Different task definitions')
        if bool(x['safe_success'])!=bool(y['safe_success']):
            transitions.append(dict(case_id=x['case_id'],old_safe=y['safe_success'],new_safe=x['safe_success']))
        if x['safe_success'] and y['safe_success']:
            common.append(dict(case_id=x['case_id'],seconds_delta=x['seconds']-y['seconds'],travel_delta_m=x['travel_m']-y['travel_m']))
    result['matched_comparison']=dict(transitions=transitions,common_safe_tasks=len(common),
        median_seconds_delta=float(np.median([x['seconds_delta'] for x in common])),
        median_travel_delta_m=float(np.median([x['travel_delta_m'] for x in common])))
    args.output.write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k!='per_case'}),flush=True)


if __name__=='__main__':main()
