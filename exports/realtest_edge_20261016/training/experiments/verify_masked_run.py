"""Verify completed masked development runs against their actual saved inputs."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from neural_navigation.masked_actions import ACTIONS,masks_from_observation
from experiments.audit_stop_feedback import audit


def verify(root):
    status=json.loads((root/'status.json').read_text())
    if status['status']!='complete':raise ValueError('Completed run required')
    manifest=json.loads((root/'portable/manifest.json').read_text())
    for name,digest in manifest['files'].items():
        if hashlib.sha256((root/'portable'/name).read_bytes()).hexdigest()!=digest:
            raise ValueError('Portable file hash mismatch: '+name)
    if hashlib.sha256((root/'candidate/policy.zip').read_bytes()).hexdigest()!=manifest['policy_sha256']:
        raise ValueError('Final policy hash mismatch')
    torch.set_num_threads(1);actor=torch.jit.load(str(root/'portable/actor.pt'))
    unique={};definitions={};per_case=[];executions=0
    strip=lambda r:{k:v for k,v in r.items() if k!='safety_context_p95_ms'}
    for index in range(len(status['summaries'])):
        report=json.loads((root/f'runtime_{index}.json').read_text())
        if not report['completed'] or report['bundle_manifest']!=manifest:
            raise ValueError('Incomplete or foreign runtime report')
        for result in report['cases']:
            cid=result['case_id'];trace=json.loads((root/f'traces_{index}'/(cid+'.json')).read_text())
            if trace['result']!=result:raise ValueError('Trace/result mismatch')
            feedback=audit(trace)
            if feedback['counts']['missing'] or feedback['counts']['feedback_mismatch']:
                raise ValueError('Missing or inconsistent stop feedback')
            obs=np.asarray([s['policy_observation'] for s in trace['samples']],np.float32)
            proposed=np.asarray([s['proposed_action'] for s in trace['samples']],np.float32)
            with torch.no_grad():actual=actor(torch.from_numpy(obs)).numpy()
            np.testing.assert_allclose(actual,proposed,atol=2e-6,rtol=0)
            distances=np.max(np.abs(proposed[:,None,:]-ACTIONS[None,:,:]),axis=2)
            choices=distances.argmin(axis=1)
            if np.any(distances.min(axis=1)>2e-6):raise ValueError('Unknown categorical command')
            masks=masks_from_observation(obs)
            if not masks[np.arange(len(obs)),choices].all():raise ValueError('Actor selected rejected command')
            voluntary=(choices==0)&masks[:,1:].any(axis=1)
            support=obs[:,281]>.5;wait_risk=obs[:,269]>0
            stats=dict(case_id=cid,feedback_mismatches=0,
                frames=len(obs),voluntary_wait_with_alternative_frames=int(voluntary.sum()),
                voluntary_wait_with_supported_stationary_warning_frames=int((voluntary&support&wait_risk).sum()),
                voluntary_wait_with_unknown_prediction_frames=int((voluntary&~support).sum()),
                only_stop_frames=int((~masks[:,1:].any(axis=1)).sum()))
            if cid in unique:
                if strip(unique[cid])!=strip(result) or definitions[cid]!=trace['case']:
                    raise ValueError('Conflicting repeated task or result')
            else:per_case.append(stats)
            unique[cid]=result;definitions[cid]=trace['case'];executions+=1
    return dict(verified=True,executions=executions,unique_tasks=len(unique),
        safe_success=sum(r['safe_success'] for r in unique.values()),
        collision_cases=sum(r['collisions']>0 for r in unique.values()),
        timeouts=sum(r['timeout'] for r in unique.values()),
        failed_cases=[r for r in unique.values() if not r['safe_success']],
        per_case=per_case,actor_hash=manifest['files']['actor.pt'],
        policy_sha256=manifest['policy_sha256'],deployment_approved=False,
        limitation='Recorded inputs, export parity, masks, stop feedback and task outcomes verified. '
            'Current feasibility is not future safety; frame counts are not independent tasks. '
            'Duplicate wall-clock latency may vary. No semantic understanding or deployment claim.')


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():p.error('Fresh verification output required')
    result=verify(a.run.resolve());a.output.write_text(json.dumps(result,indent=2))
    print(json.dumps({k:result[k] for k in ('verified','executions','unique_tasks','safe_success','collision_cases','timeouts')}))


if __name__=='__main__':main()
