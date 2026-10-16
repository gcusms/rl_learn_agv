"""Pair the first pending normal task with the frozen four completed tasks."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'training'))
from experiments.paired_navigation_gate import compare


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--normal-result',type=Path,required=True)
    parser.add_argument('--bundle',type=Path,default=ROOT/'training/runs/handoff2_native')
    args=parser.parse_args()
    base=ROOT/'reports/stop_feedback_20261015'
    frozen=json.loads((base/'recovery_followup_validation_20261016/RESULTS.json').read_text())
    direct=json.loads((base/'actual95_broad_regression_20261016/RESULTS.json').read_text())
    report=json.loads(args.normal_result.read_text())
    if not report['completed'] or len(report['rows'])!=1:
        raise ValueError('Exactly one completed pending normal task is required')
    row=report['rows'][0]
    if row['case_id']!='motion_profile_015' or row['variant']!='recovery_followup_bc400':
        raise ValueError('Wrong task or candidate label')
    manifest_path=args.bundle/'manifest.json'
    provenance=json.loads((args.normal_result.parent/'PROVENANCE.json').read_text())
    if provenance['bundle_manifest_sha256']!=hashlib.sha256(manifest_path.read_bytes()).hexdigest():
        raise ValueError('Evaluation used a different bundle')
    expected=json.loads((ROOT/'models/stop_feedback_20261015/recovery_followup_bc400_s231016/bc_portable/manifest.json').read_text())
    actual=json.loads(manifest_path.read_text())
    if actual['required_runtime']!=expected['required_runtime'] or actual['observation_contract']!=expected['observation_contract']:
        raise ValueError('Runtime or observation contract changed')
    for name,checksum in expected['files'].items():
        if Path(name).suffix!='.so' and hashlib.sha256((args.bundle/name).read_bytes()).hexdigest()!=checksum:
            raise ValueError('Candidate component changed: '+name)
    trace=args.normal_result.parent/(row['case_id']+'_'+row['variant']+'.npz')
    if hashlib.sha256(trace.read_bytes()).hexdigest()!=row['trace_sha256']:
        raise ValueError('Completed trace checksum mismatch')
    after=[r for r in frozen['rows'] if r['group']=='normal4']+[row]
    if len(after)!=5:
        raise ValueError('Frozen four normal tasks are missing')
    ids={r['case_id'] for r in after}
    before=[r for r in direct['full18_rows'] if r['case_id'] in ids]
    paired=compare(before,after)
    print(json.dumps(dict(normal5=paired,new_physical_steps=0,deployment_approved=False),indent=2))
    if not paired['all_regression_checks_passed']:
        raise SystemExit('Normal navigation gate failed; preserve results and investigate before expanding.')


if __name__=='__main__':
    main()
