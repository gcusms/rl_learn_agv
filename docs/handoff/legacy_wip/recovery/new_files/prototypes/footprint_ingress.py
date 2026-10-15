"""Opt-in learned-routing start ingress with the real oriented footprint.

Rejected diagnostic prototype: the 2026-10-13 case 023 still timed out
and increased all-stop frames from 5 to 593. Not a production runtime.
The fixed footprint below belongs only to the evaluated hardware bundle.

Only the current local start region can differ from the conservative circular
planning grid. Occupied/unknown map cells, laser inflation, actor commands and
the native emergency guard remain intact. No graph search or oracle input.
"""
import numpy as np
from navrl.map import segment_is_free
from neural_navigation.static_footprint import static_footprint_clear
from experiments.time_consistent_tracking import TimeConsistentCore, TimeConsistentFreshRuntime

FOOTPRINT = np.array([.393924904, .329460022])


def rectangular_segment_clear(static_map, a, b, yaw, footprint=FOOTPRINT, margin=.04):
    a, b = np.asarray(a), np.asarray(b)
    distance = float(np.linalg.norm(b-a))
    if distance > .45:
        return False  # Never reinterpret a long corridor with a fixed yaw.
    count = max(1, int(np.ceil(distance/.02)))
    # Cover all intermediate translations, not just discrete center samples.
    sampling_margin = distance/count/2
    return all(static_footprint_clear(static_map, np.r_[xy, yaw], footprint,
               margin+sampling_margin) for xy in np.linspace(a, b, count+1))


class FootprintIngressCore(TimeConsistentCore):
    def reset(self, *args, **kwargs):
        self.ingress_free = None
        self.ingress_active = False
        return super().reset(*args, **kwargs)

    def _neural_route(self):
        base = self.free
        self.ingress_free = None
        self.ingress_active = False
        pos = self.estimate[:2]
        sr, sc = self.cell(pos)
        if (0 <= sr < self.grid_h and 0 <= sc < self.grid_w
                and not self.static_map.lookup(pos, self.static_map.free, outside=False)
                and static_footprint_clear(self.static_map, self.estimate, FOOTPRINT, .04)):
            # The footprint can fit near a corner where a radius-.3 circle
            # cannot. Only locally validated cells are enabled for propagation.
            local = base.copy()
            for r in range(max(0,sr-2), min(self.grid_h,sr+3)):
                for c in range(max(0,sc-2), min(self.grid_w,sc+3)):
                    point = self.xy((r,c))
                    if (np.linalg.norm(point-pos) <= .35
                            and rectangular_segment_clear(self.static_map,pos,point,self.estimate[2])):
                        local[r,c] = 1.
            local[sr,sc] = 1.  # Exact robot pose was checked, not its cell center.
            self.ingress_free = local
            self.ingress_active = True
            self.free = local
        try:
            return super()._neural_route()
        finally:
            self.free = base  # Global planning geometry remains conservative.

    def _static_visible(self, a, b):
        if self.ingress_active and (not self.static_map.lookup(a,self.static_map.free,outside=False)
                or not self.static_map.lookup(b,self.static_map.free,outside=False)):
            return rectangular_segment_clear(self.static_map,a,b,self.estimate[2])
        return super()._static_visible(a,b)

    def _segment_clear(self, a, b):
        if not self._static_visible(a,b):
            return False
        free = self.ingress_free if self.ingress_active else self.free
        return segment_is_free(a,b,self.static_map.origin,self.grid_res,
                               (free>0)&~self.laser_fusion.hard)


class FootprintIngressRuntime(TimeConsistentFreshRuntime):
    def __init__(self, bundle, map_path, initial_pose, goal, route_mode='shortcut'):
        super().__init__(bundle,map_path,initial_pose,goal,route_mode)
        core = FootprintIngressCore(self.core.static_map,self.core.lidar,bundle,route_mode)
        core.reset(initial_pose,goal)
        self.core=core;self.policy.core=core
