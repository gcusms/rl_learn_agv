"""Run an explicit subset of the published fixed development cases."""
import argparse
import json
from pathlib import Path
import torch
from experiments.evaluate_quality_regression import evaluate


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--bundle',type=Path,default=Path('models/quality_recovery_20261014/portable'))
    p.add_argument('--manifest',type=Path,default=Path('models/quality_recovery_20261014/reports/cases.json'))
    p.add_argument('--case',nargs='+',help='Named cases; omitted means all 48')
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    cases=json.loads(args.manifest.read_text())['cases']
    if args.case:
        if len(set(args.case))!=len(args.case): p.error('Duplicate cases')
        unknown=set(args.case)-{c['id'] for c in cases}
        if unknown: p.error(f'Unknown cases: {sorted(unknown)}')
        cases=[c for c in cases if c['id'] in args.case]
    if args.output.exists(): p.error('Fresh output required')
    args.output.mkdir(parents=True)
    torch.set_num_threads(1)
    results=[]
    for case in cases:
        trace=evaluate(case,args.bundle)
        (args.output/(case['id']+'.json')).write_text(json.dumps(trace,allow_nan=False))
        results.append(trace['result'])
        summary=dict(completed_cases=len(results),requested_cases=len(cases),
            safe_success=sum(r['safe_success'] for r in results),collisions=sum(r['collisions'] for r in results),
            timeouts=sum(r['timeout'] for r in results),cases=results,deployment_approved=False)
        (args.output/'summary.json').write_text(json.dumps(summary,indent=2))
        print(json.dumps(trace['result']),flush=True)


if __name__=='__main__': main()
