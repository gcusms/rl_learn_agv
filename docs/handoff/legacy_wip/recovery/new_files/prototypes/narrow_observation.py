"""Append-only sensor features/probes for fine RL actions; never issues commands."""
import copy
import numpy as np
from neural_navigation.masked_actions import COMMANDS as OLD_COMMANDS,masks_from_observation

FINE_COMMANDS=np.array([[0,.1],[0,-.1],[0,.2],[0,-.2],[.05,.1],[.05,-.1],[.1,.1],[.1,-.1]])
COMMANDS=np.vstack([OLD_COMMANDS,FINE_COMMANDS])
ACTIONS=(COMMANDS/[.2,.5]).astype(np.float32)
DIM=318  # old 300 + eight probes + six corridor + four route-state fields


def corridor_features(scan,lidar):
    scan=np.asarray(scan);valid=np.isfinite(scan)&(scan>=lidar.range_min)&(scan<min(2.5,lidar.range_max))
    angles=lidar.angles[valid]+lidar.mount[3]
    points=scan[valid,None]*np.c_[np.cos(angles),np.sin(angles)]+lidar.mount[:2]
    walls=[]
    for sign in (1,-1):
        selected=points[(sign*points[:,1]>.08)&(points[:,0]>-.15)&(points[:,0]<1.2)]
        if len(selected)<8:return np.zeros(6,np.float32)
        # At an entrance, perpendicular end-wall returns coexist with the
        # corridor side. Find a supported longitudinal line before TLS;
        # never fit all these different surfaces as one wall.
        best=None
        for i,j in zip(range(len(selected)-3),range(3,len(selected))):
            delta=selected[j]-selected[i]
            if np.linalg.norm(delta)<.06:continue
            heading=np.arctan2(delta[1],delta[0]);heading=(heading+np.pi/2)%np.pi-np.pi/2
            if abs(heading)>.35:continue
            normal=np.array([-np.sin(heading),np.cos(heading)])
            inliers=np.abs((selected-selected[i])@normal)<.015
            support=selected[inliers]
            if len(support)<8 or np.ptp(support[:,0])<.25:continue
            if best is None or len(support)>len(best):best=support
        if best is None:return np.zeros(6,np.float32)
        selected=best
        center=selected.mean(0);values,vectors=np.linalg.eigh(np.cov(selected.T));normal=vectors[:,0]
        if normal[1]<0:normal=-normal
        residual=np.abs((selected-center)@normal)
        heading=float(np.arctan2(-normal[0],normal[1]))
        if values[-1]<30*max(values[0],1e-8) or np.median(residual)>.015 or abs(heading)>.35:return np.zeros(6,np.float32)
        walls.append((float(normal@center),heading,float(np.quantile(residual,.9))))
    left,right=walls
    if left[0]<=0 or right[0]>=0 or abs(left[1]-right[1])>.14:return np.zeros(6,np.float32)
    width=left[0]-right[0];offset=(left[0]+right[0])/2;heading=(left[1]+right[1])/2
    return np.clip([1,width/2,offset/.25,heading/.35,left[0]/1.5,-right[0]/1.5],-1,1).astype(np.float32)


def fine_masks(core,guard,timestamp,scan_valid=True,received_timestamp=None):
    now=timestamp if received_timestamp is None else received_timestamp
    snapshot=copy.deepcopy(guard)
    if snapshot.memory is not None:
        points,uncertainty=snapshot.memory.update(core)
        class Prepared:
            def update(self,unused):return points,uncertainty
            def reset(self):pass
        snapshot.memory=Prepared()
    return np.array([np.allclose(copy.copy(snapshot).check(core,command,timestamp,now,
          scan_valid,False)[0],command,atol=1e-9,rtol=0) for command in FINE_COMMANDS],bool)


def augment(prefix,core,guard,timestamp,**kwargs):
    prefix=np.asarray(prefix,np.float32)
    if prefix.shape!=(300,) or not np.isfinite(prefix).all():raise ValueError('Finite native 300D prefix required')
    fine=fine_masks(core,guard,timestamp,**kwargs)
    corridor=corridor_features(core.scan,core.lidar)
    state=np.array([core.motion_route_valid,core.alignment_required,
                    core.narrow_route_state=='route_geometry_or_scan_blocked',core.odometry_sanity_failure],np.float32)
    return np.r_[prefix,fine,corridor,state].astype(np.float32)


def masks(observation):
    observation=np.asarray(observation)
    if observation.shape[-1:]!=(DIM,) or not np.isfinite(observation).all():raise ValueError('Finite 318D observation required')
    return np.concatenate([masks_from_observation(observation[...,:282]),observation[...,300:308]>.5],axis=-1)


def contract():
    return dict(schema='experimental_narrow_choices_20_v3',dimensions=DIM,prefix_dimensions=300,
        commands=COMMANDS.tolist(),stop_index=0,scan_only=True,guard_unchanged=False,
        guard_schema='experimental_narrow_alignment_reserve_v2',
        fine_mask_requires_corridor_detection=False,residual_learning_in_blocked_states=True,
        short_waypoint_uses_goal_alignment_hint=True,alignment_hint_certifies_route=False,
        corridor_centerline_hint=True,localizer='bounded_observable_map_surface_v1',
        checked_entrance_staging=True,
        retain_alignment_space_before_forward=True,
        probes_do_not_issue_commands=True,default_runtime_compatible=False)
