"""Preserve sensor-checked alignment information without declaring a safe route."""
import numpy as np
from neural_navigation.core import RoomRoutingMixin
from prototypes.anchored_narrow_runtime import AnchoredNarrowCore,AnchoredNarrowRuntime
from prototypes.narrow_passage import rotation_clear
from prototypes.narrow_passage import swept_segment_clear
from navrl.map import segment_is_free
from experiments.time_consistent_tracking import TimeConsistentFreshRuntime


def path_turns_clear(core,path,initial_yaw):
    yaw=initial_yaw
    for a,b in zip(path[:-1],path[1:]):
        d=b-a
        if np.linalg.norm(d)<1e-8:continue
        heading=float(np.arctan2(d[1],d[0]))
        if yaw is not None and not rotation_clear(core.static_map,a,yaw,heading,core.footprint,core.margin):return False
        yaw=heading
    return True


class AlignmentNarrowCore(AnchoredNarrowCore):
    def __init__(self,static_map,lidar,components,route_mode='shortcut'):
        super().__init__(static_map,lidar,components,route_mode)
        from pathlib import Path
        from prototypes.surface_localizer import SurfaceLocalizer
        self.localizer=SurfaceLocalizer(static_map,lidar,Path(components)/'localizer.pt')
        self.enable_entrance_staging=True

    def _local_segment_clear(self,a,b):
        # Short recovery segments use finer coverage rather than lowering the
        # body margin. The 1 cm global sampler otherwise adds up to 5 mm of
        # extra inflation, enough to erase a feasible local braking segment.
        if np.linalg.norm(b-a)>.15:return False
        if not swept_segment_clear(self.static_map,a,b,self.footprint,self.margin,max_spacing=.002):return False
        return segment_is_free(a,b,self.static_map.origin,self.grid_res,(self.free>0)&~self.laser_fusion.hard)

    def _observation(self,scan,initialize=False):
        base=super()._observation(scan,initialize)
        target=base[216:218]*2
        self.policy_target_kind='checked_waypoint'
        if np.linalg.norm(target)<.08:
            # A near/erased waypoint gives no usable alignment direction.
            # This hint never enters predicted_path or certifies movement.
            delta=self.goal-self.estimate[:2];length=np.linalg.norm(delta)
            if length>.12:
                yaw=self.estimate[2];rotation=np.array([[np.cos(yaw),np.sin(yaw)],[-np.sin(yaw),np.cos(yaw)]])
                local=rotation@(delta/length*.15)
                base[216:218]=local/2;angle=np.arctan2(local[1],local[0])
                base[231:233]=[np.sin(angle),np.cos(angle)]
                self.policy_target_kind='goal_alignment_hint'
        # Observe the corridor early, while there is still turning clearance.
        # The hint describes a centerline point, never an executable path or an
        # action override. A turning goal/route must not be replaced by straight
        # corridor guidance. Missing or inconsistent wall fits give no hint.
        if hasattr(self,'lidar'):
            from prototypes.narrow_observation import corridor_features
            feature=corridor_features(scan,self.lidar)
            self.corridor_alignment_feature=feature.copy()
            if feature[0]>.5 and .40<float(feature[1])*2<.80:
                heading=float(feature[3])*.35;offset=float(feature[2])*.25
                direction=np.array([np.cos(heading),np.sin(heading)])
                normal=np.array([-direction[1],direction[0]])
                yaw=self.estimate[2];rotation=np.array([[np.cos(yaw),np.sin(yaw)],[-np.sin(yaw),np.cos(yaw)]])
                goal_local=rotation@(self.goal-self.estimate[:2])
                hint=direction*.35+normal*offset
                route_direction=base[216:218]*2
                if (goal_local@direction>.4 and abs(np.cross(direction,goal_local))<.25
                        and route_direction@direction>=0):
                    base[216:218]=hint/2;angle=np.arctan2(hint[1],hint[0])
                    base[231:233]=[np.sin(angle),np.cos(angle)]
                    self.policy_target_kind='lidar_corridor_alignment_hint'
        stage=getattr(self,'entrance_staging_point',None)
        if stage is not None:
            delta=stage-self.estimate[:2];length=np.linalg.norm(delta)
            yaw=self.estimate[2];rotation=np.array([[np.cos(yaw),np.sin(yaw)],[-np.sin(yaw),np.cos(yaw)]])
            local=rotation@delta
            # Do not skip the checked approach corner merely because a
            # distant centerline or the final goal is visible beyond it.
            if .05<length<.8 and local[0]>0:
                base[216:218]=local/2;angle=np.arctan2(local[1],local[0])
                base[231:233]=[np.sin(angle),np.cos(angle)]
                self.policy_target_kind='checked_entrance_staging'
        self.last_policy_observation=base.copy()
        return base

    def reset(self,*args,**kwargs):
        self.alignment_required=False;self.motion_route_valid=False
        self.narrow_route_state='initial'
        self.entrance_staging_point=None
        self.corridor_alignment_feature=None
        return super().reset(*args,**kwargs)

    def _neural_route(self):
        # Obtain the original learned proposal before NarrowCore discards it.
        RoomRoutingMixin._neural_route(self)
        candidate=self.predicted_path.copy()
        segments=RoomRoutingMixin._route_usable(self,candidate)
        future_turns=segments and path_turns_clear(self,candidate,None)
        self.motion_route_valid=bool(future_turns and path_turns_clear(self,candidate,float(self.estimate[2])))
        self.alignment_required=bool(future_turns and not self.motion_route_valid)
        if self.motion_route_valid:self.narrow_route_state='executable'
        elif self.alignment_required:self.narrow_route_state='initial_alignment_required'
        else:
            self.narrow_route_state='route_geometry_or_scan_blocked'
            # Keep only a freshly checked local direction if one exists.
            usable=[point for point in candidate[1:] if np.linalg.norm(point-self.estimate[:2])>.05
                    and self._segment_clear(self.estimate[:2],point)]
            if not usable:
                # A failed distant connection must not erase a currently clear
                # short forward segment. This supplies local information only;
                # the policy still chooses the command and the guard caps it.
                heading=np.array([np.cos(self.estimate[2]),np.sin(self.estimate[2])])
                goal=getattr(self,'goal',self.estimate[:2])
                if (goal-self.estimate[:2])@heading>.1:
                    for length in (.12,.08):
                        point=self.estimate[:2]+heading*length
                        if self._local_segment_clear(self.estimate[:2],point):usable=[point];break
            self.predicted_path=np.array([self.estimate[:2].copy(),usable[0]]) if usable else np.array([self.estimate[:2].copy()])
        if not self.motion_route_valid:
            self.current_probability=0.;self.replan_reason=self.narrow_route_state

    def diagnostics(self):
        result=super().diagnostics()
        result['route_complete']=bool(result['route_complete'] and self.motion_route_valid)
        result['narrow_route_state']=self.narrow_route_state
        result['alignment_required']=self.alignment_required
        result['policy_target_kind']=getattr(self,'policy_target_kind','initial')
        stage=getattr(self,'entrance_staging_point',None)
        result['entrance_staging_point']=None if stage is None else stage.tolist()
        localizer=getattr(self,'localizer',None)
        if localizer is not None:
            result['localization_surface_residual_m']=getattr(localizer,'surface_residual_m',None)
            result['localization_observable_rank']=getattr(localizer,'observable_rank',None)
            result['localization_surface_support']=getattr(localizer,'surface_support',None)
            result['localization_rejection_reason']=getattr(localizer,'rejection_reason',None)
        return result


class AlignmentNarrowRuntime(AnchoredNarrowRuntime):
    experiment_variant='experimental_checked_alignment_information'
    def __init__(self,bundle,map_path,initial_pose,goal,route_mode='shortcut'):
        # Avoid constructing each intermediate experimental geometry core.
        TimeConsistentFreshRuntime.__init__(self,bundle,map_path,initial_pose,goal,route_mode)
        core=AlignmentNarrowCore(self.core.static_map,self.core.lidar,bundle,route_mode)
        core.reset(initial_pose,goal);self.core=core;self.policy.core=core
        from prototypes.narrow_recovery_guard import replace_guard
        self.guard=replace_guard(self.guard)
