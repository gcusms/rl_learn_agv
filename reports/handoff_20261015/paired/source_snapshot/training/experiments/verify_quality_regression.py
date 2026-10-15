"""Verify actual quality-actor inputs and feedback for all fixed regression tasks."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from neural_navigation.masked_actions import ACTIONS,COMMANDS,masks_from_observation
from experiments.audit_stop_feedback import audit
from experiments.train_quality_navigation import digest


def verify(root,bundle,record_source_changes=False):
    report=json.loads((root/'RESULTS.json').read_text())
    if not report['completed'] or report['tasks']!=48 or len(report['cases'])!=48:
        raise ValueError('Completed 48-task regression required')
    if len({r['case_id'] for r in report['cases']})!=48:raise ValueError('Duplicate task')
    manifest=json.loads((bundle/'manifest.json').read_text())
    if report['bundle_manifest']!=manifest:raise ValueError('Foreign actor bundle')
    for name,expected in manifest['files'].items():
        if digest(bundle/name)!=expected:raise ValueError('Actor/component hash mismatch')
    sources=json.loads((root/'SOURCE_PROVENANCE.json').read_text())
    project=Path(__file__).resolve().parents[2]
    changed_sources=[]
    for relative,expected in sources['source_sha256'].items():
        if digest(root/'source_snapshot'/relative)!=expected:raise ValueError('Captured source archive changed')
        if digest(project/relative)!=expected:changed_sources.append(relative)
    if changed_sources and not record_source_changes:raise ValueError('Source changed after capture: '+', '.join(changed_sources))
    for path,expected in report['reference_case_sha256'].items():
        if digest(Path(path))!=expected:raise ValueError('Fixed reference case changed')
    actor=torch.jit.load(str(bundle/'actor.pt'),map_location='cpu');actor.eval()
    frames=0;per_case=[]
    for summary in report['cases']:
        path=root/(summary['case_id']+'.json')
        if digest(path)!=summary['trace_sha256']:raise ValueError('Trace hash mismatch')
        trace=json.loads(path.read_text());expected={k:v for k,v in summary.items() if k!='trace_sha256'}
        if trace['result']!=expected or trace['case']['id']!=summary['case_id']:raise ValueError('Trace summary mismatch')
        rows=trace['samples'];obs=np.asarray([r['policy_observation'] for r in rows],np.float32)
        if obs.shape!=(len(rows),300) or not np.isfinite(obs).all():raise ValueError('Actual finite quality inputs required')
        masks=masks_from_observation(obs[:,:282]);choices=np.asarray([r['choice'] for r in rows])
        np.testing.assert_array_equal(masks,[r['mask'] for r in rows])
        if not masks[np.arange(len(rows)),choices].all():raise ValueError('Rejected choice')
        with torch.no_grad():np.testing.assert_allclose(actor(torch.from_numpy(obs)).numpy(),ACTIONS[choices],atol=0,rtol=0)
        np.testing.assert_allclose([r['proposed_action'] for r in rows],ACTIONS[choices],atol=0,rtol=0)
        nonzero=(choices!=0)&np.array([r['stop_reason']!='estimated_goal_reached' for r in rows])
        np.testing.assert_allclose(np.asarray([r['command'] for r in rows])[nonzero],COMMANDS[choices[nonzero]],atol=2e-8,rtol=0)
        feedback=audit(dict(case=trace['case'],samples=[dict(r,policy_observation=r['policy_observation'][:282]) for r in rows]))
        if feedback['counts']['missing'] or feedback['counts']['feedback_mismatch']:raise ValueError('Parking feedback mismatch')
        voluntary=int(((choices==0)&masks[:,1:].any(axis=1)).sum());forced=int((~masks[:,1:].any(axis=1)).sum())
        if (voluntary,forced,len(rows))!=(summary['voluntary_wait_frames'],summary['only_stop_frames'],summary['frames']):
            raise ValueError('Waiting/frame counts disagree')
        frames+=len(rows);per_case.append(dict(case_id=summary['case_id'],voluntary_wait_frames=voluntary,only_stop_frames=forced))
    actual=dict(safe_success=sum(r['safe_success'] for r in report['cases']),collisions=sum(r['collisions'] for r in report['cases']),timeouts=sum(r['timeout'] for r in report['cases']))
    if any(actual[k]!=report[k] for k in actual):raise ValueError('Aggregate score mismatch')
    return dict(verified=True,tasks=48,frames=frames,**actual,policy_sha256=manifest['policy_sha256'],per_case=per_case,
        actual_actor_mask_command_and_feedback_parity=True,source_hashes_unchanged_since_capture=not changed_sources,
        captured_source_archive_verified=True,current_source_changes=changed_sources,
        deployment_approved=False,limitation='Simulated collision/arrival labels are reported by evaluator, not independently reconstructed physical outcomes. Source capture occurred after launch. No semantic understanding, real hardware or final acceptance claim.')


def main():
    parser=argparse.ArgumentParser()
    for name in ('run','bundle','output'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--record-source-changes',action='store_true',help='Verify archive and artifacts, reporting current source differences without claiming unchanged source')
    args=parser.parse_args()
    if args.output.exists():parser.error('Fresh output required')
    torch.set_num_threads(1);result=verify(args.run,args.bundle,args.record_source_changes)
    args.output.write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='per_case'}))


if __name__=='__main__':main()
