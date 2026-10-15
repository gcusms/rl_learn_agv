"""Exact wait replay, then independent continuous-command guard shadows."""
import argparse
import copy
import json
from pathlib import Path
import numpy as np
import torch
from navrl.geometry import ROOT
from neural_navigation.stress_env import StressNavigationEnv
from experiments.time_consistent_tracking import TimeConsistentFreshRuntime
from experiments.train_quality_navigation import digest


def main():
    p=argparse.ArgumentParser()
    for name in ('bundle','trace','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--frame',type=int,default=56)
    a=p.parse_args()
    if a.output.exists() or a.frame<0:p.error('Fresh output and nonnegative frame required')
    torch.set_num_threads(1);trace=json.loads(a.trace.read_text());case=trace['case'];scene=ROOT/case['scene']
    if a.frame>=len(trace['samples']):p.error('Frame outside recorded trace')
    env=StressNavigationEnv(case=case,components=a.bundle,scene_dir=scene,route_mode='shortcut',max_steps=700)
    probes=[]
    try:
        env.reset(seed=case['seed'],options={k:case[k] for k in ('start','goal','initial_prior_offset') if k in case})
        runtime=TimeConsistentFreshRuntime(a.bundle,scene/'decision_map.yaml',
            np.asarray(case['start'])+np.asarray(case.get('initial_prior_offset',[0,0,0])),case['goal'],'shortcut')
        for index,row in enumerate(trace['samples'][:a.frame+1]):
            result=runtime.step(env.scan,env.observed_odometry,env.measured_velocity,env.elapsed,capture_observation=True)
            np.testing.assert_allclose(result['policy_observation'],row['policy_observation'],atol=2e-6,rtol=0)
            np.testing.assert_allclose(result['command'],row['command'],atol=2e-8,rtol=0)
            assert result['stop_reason']==row['stop_reason']
            if index==a.frame:
                for v in (-.02,0.,.005,.01,.02,.05,.1,.2):
                    for w in (-.5,-.25,-.15,-.1,-.05,0.,.05,.1,.15,.25,.5):
                        command=np.array([v,w]);shadow=copy.deepcopy(runtime.guard)
                        actual,reason=shadow.check(runtime.core,command,env.elapsed,env.elapsed)
                        probes.append(dict(command=command.tolist(),accepted=bool(np.allclose(actual,command,atol=1e-9,rtol=0)),reason=reason))
                break
            v,w=result['command'];env.step(np.array([v/(.2 if v>=0 else .08),w/.5]))
            if env.collision_events or env.hard_guard_activations:raise ValueError('Replay physical failure')
        report=dict(frame=a.frame,timestamp=float(env.elapsed),trace_sha256=digest(a.trace),
            bundle_manifest_sha256=digest(a.bundle/'manifest.json'),native_mask=row['mask'],probes=probes,
            actual_observation_command_and_reason_reproduced=True,commands_executed=False,
            limitation='Independent current-state guard shadows, not continuous-motion safety or RL success. Reverse remains subject to native blind-sector checks; no new commands added to policy.')
        a.output.write_text(json.dumps(report,indent=2));print(json.dumps([x for x in probes if x['accepted']]),flush=True)
    finally:env.close()


if __name__=='__main__':main()
