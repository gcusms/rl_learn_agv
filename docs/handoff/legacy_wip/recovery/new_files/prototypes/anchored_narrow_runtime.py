"""Refine feasible points inside existing coarse cells, without opening walls.

Learned propagation stays on its original grid. Decoded positions are offset
within their own cell only when oriented body geometry permits it; all route
sweeps and rotations remain checked. This is an experimental static-map input
adapter, not a controller or an obstacle-truth interface.
"""
import numpy as np
from prototypes.narrow_runtime import NarrowCore,NarrowRuntime
from neural_navigation.static_footprint import static_footprint_clear
from prototypes.narrow_passage import rotation_clear


def feasible_anchor(static_map, nominal, footprint, margin, grid_res, contains):
    offsets=np.array([(x,y) for x in (-grid_res/3,0,grid_res/3)
                           for y in (-grid_res/3,0,grid_res/3)])
    order=np.argsort(np.linalg.norm(offsets,axis=1),kind='stable')
    candidates=[]
    for offset in offsets[order]:
        p=nominal+offset
        if not contains(p):continue
        clear=[static_footprint_clear(static_map,np.r_[p,yaw],footprint,margin)
               for yaw in np.arange(4)*np.pi/4]
        if not any(clear):continue
        # The router has cardinal edges. A diagonal-only nominal placement
        # must not outrank a nearby placement supporting those same edges.
        cardinal=sum(clear[::2])
        reserve=sum(static_footprint_clear(static_map,np.r_[p,yaw],footprint,margin+.01)
                    for yaw in (0.,np.pi/2)) if cardinal else 0
        score=(cardinal,reserve,sum(clear),-float(np.linalg.norm(offset)))
        candidates.append((score,p))
    return max(candidates,key=lambda x:x[0])[1] if candidates else None


class AnchoredNarrowCore(NarrowCore):
    def _decode_motion_route(self,q,occupancy,start,target):
        """Follow learned Q scores, admitting only executable body connections.

        This is bounded greedy decoding, not a shortest-path search. The first
        rotation remains an alignment task, while later corners must fit.
        """
        proposed=[self.estimate[:2].copy()];r,c=start;visited={(r,c)}
        for _ in range(160):
            if not (0<=r<self.grid_h and 0<=c<self.grid_w):break
            if (r,c)==target:
                point=self.goal.copy()
                if np.linalg.norm(point-proposed[-1])>1e-8:proposed.append(point)
                break
            chosen=None
            for direction in np.argsort(-q[:,r,c],kind='stable'):
                dr,dc=self.router.directions[int(direction)];nr,nc=r+dr,c+dc
                if not (0<=nr<self.grid_h and 0<=nc<self.grid_w) or not occupancy[nr,nc] or (nr,nc) in visited:continue
                if not np.isfinite(q[int(direction),r,c]):continue
                chosen=(nr,nc,self.xy((nr,nc)));break
            if chosen is None:break
            r,c,point=chosen;visited.add((r,c));proposed.append(point)
        # Validate longer connections before rejecting a coarse-cell corner.
        # Every admitted chord follows the learned proposal and has a complete
        # swept-body check; the original short invalid edges are never exposed.
        from prototypes.entrance_staging import connect_proposal,entrance_connection
        path,index=connect_proposal(self,proposed)
        if getattr(self,'enable_entrance_staging',False):
            path=entrance_connection(self,proposed,path,index)
        return path

    def xy(self,cell):
        if hasattr(self,'route_anchors'):return self.route_anchors[cell].copy()
        return super().xy(cell)

    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.route_anchors=np.array([[super(AnchoredNarrowCore,self).xy((r,c))
            for c in range(self.grid_w)] for r in range(self.grid_h)])
        self.anchor_refinements=0
        for r in range(self.grid_h):
            for c in range(self.grid_w):
                p=self.route_anchors[r,c]
                # Reject deep occupied regions before testing local offsets.
                if self.oriented_free[:,r,c].all():continue
                if self.static_map.lookup(p,outside=0.) < min(self.footprint)/2+self.margin-self.grid_res/2:continue
                anchor=feasible_anchor(self.static_map,p,self.footprint,self.margin,self.grid_res,
                                       lambda xy,cell=(r,c):self.cell(xy)==cell)
                if anchor is not None:
                    self.route_anchors[r,c]=anchor;self.free[r,c]=1.
                    self.anchor_refinements+=int(not np.allclose(anchor,p))
        self.laser_fusion.points=self.route_anchors.copy()
        self.router_cache.clear();self.static_visibility_cache.clear()


class AnchoredNarrowRuntime(NarrowRuntime):
    experiment_variant='experimental_oriented_full_route_with_subcell_anchors'
    def __init__(self,bundle,map_path,initial_pose,goal,route_mode='shortcut'):
        super().__init__(bundle,map_path,initial_pose,goal,route_mode)
        core=AnchoredNarrowCore(self.core.static_map,self.core.lidar,bundle,route_mode)
        core.reset(initial_pose,goal);self.core=core;self.policy.core=core
