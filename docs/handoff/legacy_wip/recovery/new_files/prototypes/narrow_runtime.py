"""Opt-in oriented-footprint route geometry; unchanged learned actor and guard.

A union of heading-feasible cells supports learned value propagation. It is not
sufficient for a route: segments and corner rotations are separately checked.
No obstacle truth, graph search, lateral actuator, or safety margin reduction.
"""
import json
from pathlib import Path
import numpy as np
from experiments.time_consistent_tracking import TimeConsistentCore, TimeConsistentFreshRuntime
from neural_navigation.static_footprint import static_footprint_clear
from prototypes.narrow_passage import swept_segment_clear,rotation_clear


class NarrowCore(TimeConsistentCore):
    def __init__(self,static_map,lidar,components,route_mode='shortcut'):
        super().__init__(static_map,lidar,components,route_mode)
        hardware=json.loads((Path(components)/'hardware.json').read_text())
        self.footprint=np.asarray(hardware['robot_footprint_m'])
        self.margin=float(hardware['near_field_memory']['static_footprint_margin_m'])
        headings=np.arange(8)*np.pi/4
        self.oriented_free=np.zeros((8,self.grid_h,self.grid_w),bool)
        for r in range(self.grid_h):
            for c in range(self.grid_w):
                p=self.xy((r,c))
                for i,yaw in enumerate(headings):
                    self.oriented_free[i,r,c]=static_footprint_clear(static_map,np.r_[p,yaw],self.footprint,self.margin)
        self.free=self.oriented_free.any(0).astype(np.float32)
        self.static_visibility_cache.clear();self.router_cache.clear()

    def _static_visible(self,a,b):
        a,b=np.asarray(a),np.asarray(b)
        if np.linalg.norm(b-a)<1e-8:return False
        key=(a.astype('float64').tobytes(),b.astype('float64').tobytes())
        if key not in self.static_visibility_cache:
            if len(self.static_visibility_cache)>4096:self.static_visibility_cache.clear()
            self.static_visibility_cache[key]=swept_segment_clear(self.static_map,a,b,self.footprint,self.margin)
        return self.static_visibility_cache[key]

    def _route_usable(self,path):
        if not super()._route_usable(path):return False
        yaw=float(self.estimate[2])
        for a,b in zip(path[:-1],path[1:]):
            d=b-a
            if np.linalg.norm(d)<1e-8:continue
            next_yaw=float(np.arctan2(d[1],d[0]))
            if not rotation_clear(self.static_map,a,yaw,next_yaw,self.footprint,self.margin):return False
            yaw=next_yaw
        return True

    def _neural_route(self):
        super()._neural_route()
        # An orientation-union grid is only a propagation proposal. Never
        # report a complete route when its full motion geometry was rejected.
        if not self._route_usable(self.predicted_path):
            self.predicted_path=np.array([self.estimate[:2].copy()])
            self.current_probability=0.;self.replan_reason='oriented_motion_geometry_rejected'


class NarrowRuntime(TimeConsistentFreshRuntime):
    experiment_variant='experimental_oriented_full_route_geometry'
    def __init__(self,bundle,map_path,initial_pose,goal,route_mode='shortcut'):
        super().__init__(bundle,map_path,initial_pose,goal,route_mode)
        core=NarrowCore(self.core.static_map,self.core.lidar,bundle,route_mode)
        core.reset(initial_pose,goal);self.core=core;self.policy.core=core
