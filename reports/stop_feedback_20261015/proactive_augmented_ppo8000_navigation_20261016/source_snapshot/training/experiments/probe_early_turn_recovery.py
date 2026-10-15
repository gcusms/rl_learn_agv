"""Simulator-only early steering branches before sustained voluntary waiting."""
import argparse
import json
from pathlib import Path
import torch
import experiments.probe_wait_alternatives as probe
from experiments.time_consistent_tracking import TimeConsistentFreshRuntime
from experiments.train_quality_navigation import digest


def main():
    p=argparse.ArgumentParser()
    for name in ('bundle','trace','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--frame',type=int,default=35)
    a=p.parse_args()
    if a.output.exists():p.error('Fresh output required')
    trace=json.loads(a.trace.read_text())
    if not 0<=a.frame<len(trace['samples']):p.error('Valid frame required')
    if not all(trace['samples'][a.frame]['mask'][c] for c in (4,5)):p.error('Both native turn choices must initially pass')
    base=probe.DiagnosticPolicy
    class EarlyTurnPolicy(base):
        def __init__(self,policy,choice):super().__init__(policy,choice);self.frames_left=25
    probe.FreshPerceptionRuntime=TimeConsistentFreshRuntime;probe.DiagnosticPolicy=EarlyTurnPolicy
    torch.set_num_threads(1)
    report=dict(completed=False,target_frame=a.frame,target_time=trace['samples'][a.frame]['timestamp'],
        trace_sha256=digest(a.trace),bundle_manifest_sha256=digest(a.bundle/'manifest.json'),
        results=[],deployment_approved=False,
        limitation='Manual simulator diagnostic, not RL success or training labels. Each turn uses native per-frame mask for up to25frames; actor resumes afterward. Truth only scores outcomes.')
    def save():a.output.write_text(json.dumps(report,indent=2))
    save()
    for choice in (None,4,5):
        result=probe.run(trace['case'],trace['samples'],a.bundle,a.frame,choice)
        report['results'].append(result);save();print(json.dumps({k:v for k,v in result.items() if k!='rows'}),flush=True)
    report['completed']=True;save()


if __name__=='__main__':main()
