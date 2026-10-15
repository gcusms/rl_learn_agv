"""Continue diagnostic early pure-turn branches through a full simulated episode."""
import argparse
import json
from pathlib import Path
import torch
from experiments.probe_fine_yaw_recovery import branch
from experiments.train_quality_navigation import digest


def main():
    p=argparse.ArgumentParser()
    for name in ('bundle','trace','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--frame',type=int,default=35)
    a=p.parse_args()
    if a.output.exists():p.error('Fresh output required')
    trace=json.loads(a.trace.read_text())
    if not 0<=a.frame<len(trace['samples']):p.error('Valid frame required')
    torch.set_num_threads(1)
    report=dict(completed=False,target_frame=a.frame,intervention_frames=25,
        trace_sha256=digest(a.trace),bundle_manifest_sha256=digest(a.bundle/'manifest.json'),
        results=[],deployment_approved=False,
        limitation='Manual early-turn diagnostic for up to25frames, then original actor to140s episode limit. Native guard checked each frame. Baseline entire observation/command/reason replay verified. Not RL policy success or training labels.')
    def save():a.output.write_text(json.dumps(report,indent=2))
    save()
    for yaw in (None,.5,-.5):
        result=branch(trace,a.bundle,a.frame,yaw,25,followup_frames=700-a.frame-25)
        result['safe_arrival_score_only']=bool(result['rows'][-1]['stop_reason']=='estimated_goal_reached'
            and result['goal_distance_score_only_m']<=.2 and not result['collisions'] and not result['physical_guards'])
        report['results'].append(result);save();print(json.dumps({k:v for k,v in result.items() if k!='rows'}),flush=True)
    report['completed']=True;save()


if __name__=='__main__':main()
