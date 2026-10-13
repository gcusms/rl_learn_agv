"""A development selection gate, never an automatic deployment approval."""
import argparse
import json
from pathlib import Path
import numpy as np


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--baseline',type=Path,required=True)
    parser.add_argument('--candidate',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():parser.error('Use a fresh report path')
    baseline=json.loads(args.baseline.read_text());candidate=json.loads(args.candidate.read_text())
    if not baseline['completed'] or not candidate['completed']:parser.error('Both evaluations must be complete')
    if baseline['manifest_sha256']!=candidate['manifest_sha256'] or baseline['max_steps']!=candidate['max_steps']:
        parser.error('Evaluation manifests or step limits differ')
    b={r['case_id']:r for r in baseline['cases']};c={r['case_id']:r for r in candidate['cases']}
    if b.keys()!=c.keys():parser.error('Case IDs differ')
    summary={}
    for name,cases in [('baseline',b),('candidate',c)]:
        rows=list(cases.values())
        summary[name]={'tasks':len(rows),'safe_success':sum(r['safe_success'] for r in rows),
                       'collisions':sum(r['collisions'] for r in rows),'guards':sum(r['guards'] for r in rows),
                       'mean_seconds':float(np.mean([r['seconds'] for r in rows])),
                       'mean_travel_m':float(np.mean([r['travel_m'] for r in rows])),
                       'max_localization_error_m':max(r['localization_error_max_m'] for r in rows)}
    old=summary['baseline'];new=summary['candidate']
    time_change=new['mean_seconds']/old['mean_seconds']-1
    path_change=new['mean_travel_m']/old['mean_travel_m']-1
    safe=all(r['safe_success'] for r in c.values())
    efficient=time_change<=.02 and path_change<=.02 and min(time_change,path_change)<=-.01
    report={'summary':summary,'time_change_percent':time_change*100,'path_change_percent':path_change*100,
            'development_gate_passed':bool(safe and efficient),
            'experimental_preference':candidate['policy'] if safe and efficient else baseline['policy'],
            'gate':'All candidate cases safe; mean time/path regress no more than 2%; at least one improves by 1%',
            'case_deltas':[{'case_id':key,'seconds_change':c[key]['seconds']-b[key]['seconds'],
                            'travel_m_change':c[key]['travel_m']-b[key]['travel_m']} for key in sorted(b)],
            'deployment_approved':False,'limitation':baseline['limitation'],
            'note':'This is a small development comparison, not a statistical generalization claim. Combined training changes do not establish single-factor causality.'}
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k!='case_deltas'},indent=2))


if __name__=='__main__':main()
