"""Isolated diagnostic: bounded model-requested creep when a route is incomplete.

Retains the real scan freshness, localization, static-map and body braking
checks. Only the incomplete-route restriction is replaced in this experiment.
This file is not imported by the active training or normal deployed runtime.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
import evaluate_sensor_navigation as evaluation
from neural_navigation.runtime import EmergencyStop,SensorNavigationRuntime


class GeometryView:
    def __init__(self,core):self.core=core
    def __getattr__(self,name):return getattr(self.core,name)
    def diagnostics(self):return {**self.core.diagnostics(),'route_complete':True}


class BoundedRouteRecoveryGuard(EmergencyStop):
    def check(self,core,command,scan_timestamp,now,scan_valid=True,rear_clearance_known=False):
        command=np.asarray(command,float)
        if (np.isfinite(command).all() and command[0]>0 and not core.diagnostics()['route_complete']):
            limited=command.copy();limited[0]=min(limited[0],.05)
            executed,reason=super().check(GeometryView(core),limited,scan_timestamp,now,scan_valid,rear_clearance_known)
            if reason is None and limited[0]!=command[0]:reason='learned_route_incomplete'
            return executed,reason
        return super().check(core,command,scan_timestamp,now,scan_valid,rear_clearance_known)


class RecoveryRuntime(SensorNavigationRuntime):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        g=self.guard
        self.guard=BoundedRouteRecoveryGuard(g.half*2,g.max_scan_age,g.braking_acceleration,
                                            g.memory.config if g.memory else None)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--bundle',type=Path,required=True)
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--traces',type=Path)
    a=p.parse_args()
    if a.output.exists():p.error('Fresh output required')
    torch.set_num_threads(1)
    evaluation.SensorNavigationRuntime=RecoveryRuntime
    results=[]
    for case in json.loads(a.manifest.read_text())['cases']:
        result=evaluation.run_case((case,a.bundle.resolve(),700,a.traces))
        results.append(result);print('CASE',json.dumps(result),flush=True)
    a.output.write_text(json.dumps(dict(completed=True,cases=results,
        safe_success=sum(x['safe_success'] for x in results),tasks=len(results),
        collisions=sum(x['collisions'] for x in results),experimental_guard=True,
        incomplete_route_forward_cap_m_s=.05,deployment_approved=False,
        limitation='Changes one deterministic guard restriction; same actor, simulated sensors and physical checks. '
                   'Not pure learned recovery, not calibrated braking or guaranteed future dynamic safety.'),indent=2))


if __name__=='__main__':main()
