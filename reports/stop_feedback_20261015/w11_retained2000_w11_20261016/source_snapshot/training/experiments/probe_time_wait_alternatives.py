"""Bounded simulator-only alternatives with the exact time-consistent runtime.

This is a causal diagnostic, never a trained policy or deployment controller.
Every substituted action is checked against the current sensor-derived mask.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
import experiments.probe_wait_alternatives as probe
from experiments.time_consistent_tracking import TimeConsistentFreshRuntime
from experiments.train_quality_navigation import digest


def main():
    parser=argparse.ArgumentParser()
    for name in ('bundle','trace','output'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--intervention-frames',type=int,choices=(5,25),default=5)
    parser.add_argument('--allow-unsupported-prediction',action='store_true',
        help='Diagnose waits without prediction support; native guard remains mandatory')
    args=parser.parse_args()
    if args.output.exists():parser.error('Fresh output required')
    trace=json.loads(args.trace.read_text());streak=0;target=None
    for index,row in enumerate(trace['samples']):
        streak=streak+1 if row['choice']==0 and any(row['mask'][1:]) else 0
        if streak>=10 and (row['policy_observation'][294]>.5 or args.allow_unsupported_prediction):target=index;break
    if target is None:raise ValueError('No sustained wait meeting requested prediction support')
    choices=[int(x) for x in np.flatnonzero(trace['samples'][target]['mask']) if x]
    if len(choices)>3:raise ValueError('At most three diagnostic alternatives')
    original=probe.DiagnosticPolicy
    class BoundedDiagnosticPolicy(original):
        def __init__(self,policy,choice):
            super().__init__(policy,choice);self.frames_left=args.intervention_frames
    probe.FreshPerceptionRuntime=TimeConsistentFreshRuntime
    probe.DiagnosticPolicy=BoundedDiagnosticPolicy
    torch.set_num_threads(1)
    report=dict(completed=False,target_frame=target,target_time=trace['samples'][target]['timestamp'],
        trace_sha256=digest(args.trace),bundle_manifest_sha256=digest(args.bundle/'manifest.json'),
        runtime='TimeConsistentFreshRuntime',intervention_frames=args.intervention_frames,
        prediction_pose_support=float(trace['samples'][target]['policy_observation'][294]),
        unsupported_prediction_diagnostic=args.allow_unsupported_prediction,
        results=[],deployment_approved=False,
        limitation='Manual simulated action branches with per-frame masks and exact pre-branch replay. Not RL success, not training labels, not deployment behavior. Up to 105 branch frames; truth only scores displacement/contact/goal distance.')
    def save():args.output.write_text(json.dumps(report,indent=2))
    save()
    for choice in [None]+choices:
        result=probe.run(trace['case'],trace['samples'],args.bundle,target,choice)
        report['results'].append(result);save()
        print(json.dumps({k:v for k,v in result.items() if k!='rows'}),flush=True)
    report['completed']=True;save()


if __name__=='__main__':main()
