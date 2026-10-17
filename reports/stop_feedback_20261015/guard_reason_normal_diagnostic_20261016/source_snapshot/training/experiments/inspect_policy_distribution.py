"""Replay a development case with unchanged exported commands and inspect PPO.

The separate SB3 network and guard probes only log evidence. The original
TorchScript actor still emits every proposed action; no oracle is used.
"""
import argparse
import json
import math
from pathlib import Path
import time
import numpy as np
import torch
from stable_baselines3 import PPO
import evaluate_sensor_navigation as evaluation
from experiments.safety_context import SafetyContext,PROBES
from neural_navigation.runtime import SensorNavigationRuntime


def interval_probability(mean,std,lo,hi):
    zlo=(lo-mean)/std;zhi=(hi-mean)/std
    if zlo>0:return max(0.,.5*(math.erfc(zlo/math.sqrt(2))-math.erfc(zhi/math.sqrt(2))))
    if zhi<0:return max(0.,.5*(math.erfc(-zhi/math.sqrt(2))-math.erfc(-zlo/math.sqrt(2))))
    return max(0.,.5*(math.erf(zhi/math.sqrt(2))-math.erf(zlo/math.sqrt(2))))


def main():
    p=argparse.ArgumentParser();p.add_argument('--policy',type=Path,required=True);p.add_argument('--bundle',type=Path,required=True)
    p.add_argument('--manifest',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():p.error('Fresh diagnostic output required')
    torch.set_num_threads(1);model=PPO.load(a.policy,device='cpu');samples=[];errors=[]
    class ObservedPolicy:
        def __init__(self,original,runtime):self.original=original;self.runtime=runtime;self.index=0
        def predict(self,observation):
            action=self.original.predict(observation)
            with torch.no_grad():
                dist=model.policy.get_distribution(torch.from_numpy(observation[None].astype(np.float32))).distribution
                mean=dist.mean[0].numpy();std=dist.stddev[0].numpy()
            errors.append(float(np.max(abs(np.clip(mean,-1,1)-action))))
            if self.index in [50,100,200,400,650]:
                start=time.perf_counter();r=self.runtime
                context=r.context.observation(r.core,r.guard,r.core.elapsed)
                elapsed_ms=(time.perf_counter()-start)*1000;probe_probabilities=[]
                for i,command in enumerate(PROBES):
                    if not context[i]:continue
                    target=command/np.array([.2,.5])
                    # Clipping places all tail mass on +/-1. Count that mass
                    # when the requested executed-action interval touches it.
                    probability=float(np.prod([interval_probability(mean[j],std[j],
                                                -math.inf if target[j]-.05<=-1 else target[j]-.05,
                                                math.inf if target[j]+.05>=1 else target[j]+.05) for j in range(2)]))
                    probe_probabilities.append(dict(command=command.tolist(),normalized=target.tolist(),gaussian_box_probability=probability))
                samples.append(dict(index=self.index,timestamp=r.core.elapsed,observation=observation.tolist(),
                                    unbounded_mean=mean.tolist(),std=std.tolist(),exported_action=action.tolist(),
                                    context=context.tolist(),context_ms=elapsed_ms,feasible_probes=probe_probabilities))
            self.index+=1;return action
    class ObservedRuntime(SensorNavigationRuntime):
        def __init__(self,*args,**kwargs):
            super().__init__(*args,**kwargs);self.context=SafetyContext();self.policy=ObservedPolicy(self.policy,self)
        def step(self,*args,**kwargs):
            result=super().step(*args,**kwargs);self.context.record(result);return result
    evaluation.SensorNavigationRuntime=ObservedRuntime
    manifest=json.loads(a.manifest.read_text())
    if len(manifest['cases'])!=1:raise ValueError('Inspect one case at a time')
    result=evaluation.run_case((manifest['cases'][0],a.bundle.resolve(),700,None))
    report=dict(result=result,samples=samples,max_clipped_actor_difference=float(max(errors)),
                limitation='Single reused development case; original exported actor commands retained, independent Gaussian interval probability around listed probes, not success guarantee or measured exploration frequency')
    a.output.write_text(json.dumps(report,indent=2));print(json.dumps(result),flush=True)
    for s in samples:print(s['index'],'mean',s['unbounded_mean'],'probe_prob',[(p['command'],p['gaussian_box_probability']) for p in s['feasible_probes']],flush=True)


if __name__=='__main__':main()
