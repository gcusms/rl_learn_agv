"""Read measured current motion as well as requested motion in guard checks.

Isolated diagnostic, not a calibrated braking model or deployment default.
Never creates a steering action, increases a speed, relaxes inflated-map
geometry, or grants reverse permission. Velocity is the existing runtime
sensor input; no new encoder or firmware requirement is introduced.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
import evaluate_sensor_navigation as evaluation
from neural_navigation.runtime import EmergencyStop,SensorNavigationRuntime


class CurrentMotionGuard(EmergencyStop):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        if self.memory is None:
            raise ValueError('Current-motion experiment requires the near-field guard')

    def _trajectory_check(self,core,command,points,margins,age,allow_escape=False):
        current=np.asarray(getattr(core,'measured_velocity',[np.nan,np.nan]),float)
        if current.shape!=(2,) or not np.isfinite(current).all():return 'nonfinite_input'
        command=np.asarray(command,float)
        candidates=[command.copy()]
        # Check a conservative constant-speed continuation in addition to
        # the request. The existing horizon and braking bound are retained.
        reference=command.copy()
        if abs(current[0])>abs(reference[0])+1e-8:reference[0]=current[0]
        if not np.allclose(reference,command,atol=1e-8,rtol=0):candidates.append(reference.copy())
        if abs(current[1]-command[1])>.05:
            continuation=reference.copy();continuation[1]=current[1]
            candidates.append(continuation)
            if current[1]*command[1]<0:
                straight=reference.copy();straight[1]=0.;candidates.append(straight)
        if current[0]*command[0]<0:
            candidates.append(current.copy())
        for candidate in candidates:
            reason=super()._trajectory_check(core,candidate,points,margins,age,allow_escape)
            if reason is not None:return reason
        return None


class MotionRuntime(SensorNavigationRuntime):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        old=self.guard
        self.guard=CurrentMotionGuard(old.half*2,old.max_scan_age,old.braking_acceleration,
            old.memory.config if old.memory else None)


def main():
    p=argparse.ArgumentParser();p.add_argument('--bundle',type=Path,required=True)
    p.add_argument('--manifest',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--traces',type=Path,required=True);a=p.parse_args()
    if a.output.exists():p.error('Fresh diagnostic report required')
    torch.set_num_threads(1);evaluation.SensorNavigationRuntime=MotionRuntime
    results=[]
    for case in json.loads(a.manifest.read_text())['cases']:
        result=evaluation.run_case((case,a.bundle.resolve(),700,a.traces))
        results.append(result);print('CASE',json.dumps(result),flush=True)
    a.output.write_text(json.dumps(dict(completed=True,cases=results,tasks=len(results),
        safe_success=sum(x['safe_success'] for x in results),collisions=sum(x['collisions'] for x in results),
        experimental_guard='current_motion_continuation',deployment_approved=False,
        limitation='Existing sensor velocity and conservative constant continuations, not a calibrated PID/latency model, '
                   'not exhaustive actuator reachable sets or a future dynamic-obstacle guarantee. Same neural actor; no training update.'),indent=2))


if __name__=='__main__':main()
