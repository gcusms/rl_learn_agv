"""Expose measured action feasibility without choosing the robot action.

This is an experimental policy input, not an alternate steering controller.
Probes use cloned guards and real sensor-core state, never simulator actors.
"""
import copy
import numpy as np


PROBES=np.array([[.2,0.],[.1,0.],[.05,0.],[0.,.5],[0.,-.5],
                 [.1,.25],[.1,-.25],[.2,.25],[.2,-.25]],dtype=float)
CONTEXT_DIM=13


class SafetyContext:
    def __init__(self):self.last_proposal=np.zeros(2);self.blocked_since=None

    def observation(self,core,guard,timestamp,received_timestamp=None,scan_valid=True,rear_clearance_known=False):
        now=timestamp if received_timestamp is None else received_timestamp
        if not np.isfinite([timestamp,now]).all() or abs(timestamp-core.elapsed)>1e-8:
            raise ValueError('Context must refer to the current finite sensor frame')
        feasible=[]
        for command in PROBES:
            # No probe may alter the live guard's memory, counters or clock.
            executed,_=copy.deepcopy(guard).check(core,command,timestamp,now,scan_valid,rear_clearance_known)
            feasible.append(float(np.allclose(executed,command,atol=1e-9,rtol=0)))
        held=0. if self.blocked_since is None else np.clip((timestamp-self.blocked_since)/5.,0,1)
        return np.r_[feasible,float(rear_clearance_known),self.last_proposal,held].astype(np.float32)

    def record(self,result):
        self.last_proposal=np.clip(np.asarray(result['proposed_action'],float),-1,1)
        if result['safety_override'] and np.linalg.norm(result['command'])<1e-8:
            if self.blocked_since is None:self.blocked_since=float(result['timestamp'])
        else:self.blocked_since=None
