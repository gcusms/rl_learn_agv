"""Closed-loop ablation: same trained actor, zero only the 32 context inputs.

The real guard stays active. This diagnostic tests dependence on the added
features; it is neither a deployment mode nor an alternate controller.
"""
import argparse
import json
from pathlib import Path
import torch
import evaluate_sensor_navigation as evaluation
from neural_navigation.runtime import SensorNavigationRuntime
from neural_navigation.safety_context import OBSERVATION_DIM


class MaskedPolicy:
    def __init__(self, policy): self.policy=policy
    def predict(self, observation):
        if len(observation)!=OBSERVATION_DIM:raise ValueError('Context actor required')
        ablated=observation.copy();ablated[235:]=0.
        return self.policy.predict(ablated)


class AblatedRuntime(SensorNavigationRuntime):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        if self.context is None:raise ValueError('Context bundle required')
        self.policy=MaskedPolicy(self.policy)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--bundle',type=Path,required=True)
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():p.error('Fresh report required')
    torch.set_num_threads(1)
    evaluation.SensorNavigationRuntime=AblatedRuntime
    results=[]
    for case in json.loads(a.manifest.read_text())['cases']:
        result=evaluation.run_case((case,a.bundle.resolve(),700,None))
        results.append(result);print('CASE',json.dumps(result),flush=True)
    report=dict(completed=True,context_inputs_zeroed=True,guard_unchanged=True,
                bundle=str(a.bundle.resolve()),cases=results,
                safe_success=sum(x['safe_success'] for x in results),tasks=len(results),
                collisions=sum(x['collisions'] for x in results),
                limitation='Zero-input ablation is out of the trained context distribution. '
                'Differences show dependence, not proof of a complete causal explanation.')
    a.output.write_text(json.dumps(report,indent=2))


if __name__=='__main__':main()
