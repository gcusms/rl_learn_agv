"""PPO transitions through the exact deployed sensor core and action transform.

The existing physics environment supplies sensors/labels only; its legacy
navigation observation is discarded. PPO samples a proposed latent action,
the shared deterministic guard executes it, and rewards use that transition.
No alternate steering controller or simulator truth enters the policy core.
"""
import json
import numpy as np
from gymnasium import spaces
from navrl.geometry import wrap
from neural_navigation.curriculum import NavigationCurriculum
from neural_navigation.core import NavigationCore
from neural_navigation.runtime import EmergencyStop,execute_policy_action
from neural_navigation.safety_context import SafetyContext,OBSERVATION_DIM,validate_contract


def footprint_scan_penalty(scan,lidar,footprint,coefficient,target=.16):
    """Measured endpoint distance to the rectangular body, not LiDAR origin."""
    scan=np.asarray(scan);valid=np.isfinite(scan)&(scan>=lidar.range_min)&(scan<lidar.range_max)
    if not np.any(valid):return 0.,None
    angle=lidar.angles[valid]+lidar.mount[3]
    points=scan[valid,None]*np.c_[np.cos(angle),np.sin(angle)]+lidar.mount[:2]
    half=np.asarray(footprint)/2
    gap=float(np.min(np.linalg.norm(points-np.clip(points,-half,half),axis=1)))
    return float(coefficient*max(0.,(target-gap)/target)**2),gap


class GuardedNavigationCurriculum(NavigationCurriculum):
    def __init__(self,components,stage='static',route_mode='shortcut',max_steps=700,projection_cost=.25,motion_profiles=False,footprint_clearance_cost=0.):
        super().__init__(components,stage,route_mode,max_steps,motion_profiles=motion_profiles)
        if not np.isfinite(projection_cost) or projection_cost<0:raise ValueError('Projection cost must be nonnegative')
        self.projection_cost=projection_cost
        if not np.isfinite(footprint_clearance_cost) or footprint_clearance_cost<0:raise ValueError('Footprint clearance cost must be nonnegative')
        self.footprint_clearance_cost=footprint_clearance_cost
        self.capture_decision_feedback=False
        hardware=self.components/'hardware.json'
        self.hardware=json.loads(hardware.read_text()) if hardware.exists() else {}
        self.use_safety_context='policy_observation' in self.hardware
        if self.use_safety_context:
            validate_contract(self.hardware['policy_observation'])
            self.observation_space=spaces.Box(-1,1,(self.hardware['policy_observation']['dimensions'],),np.float32)

    def reset(self,seed=None,options=None):
        _,info=super().reset(seed=seed,options=options);e=self.active
        source=json.loads(self.training_case_json);prior=np.asarray(source['options']['start'])+np.asarray(source['options'].get('initial_prior_offset',[0,0,0]))
        self.core=NavigationCore(e.static_map,e.lidar,self.components,self.route_mode);self.core.reset(prior,e.goal)
        self.guard=EmergencyStop(e.scene['robot_collision_box_m'][:2],memory_config=self.hardware.get('near_field_memory'))
        obs,_=self.core.update(e.scan,e.observed_odometry,e.measured_velocity,e.elapsed)
        self.guard_overrides=0;self.guard_steps=0;self.previous_executed=np.zeros(2)
        if self.use_safety_context:
            self.context=SafetyContext(self.hardware['policy_observation']['schema'])
            self.context.capture_decision_feedback=self.capture_decision_feedback
            obs=self.context.augment(obs,self.core,self.guard,e.elapsed)
        return obs,info

    def step(self,action):
        e=self.active;proposed=np.clip(np.asarray(action,float),-1,1)
        if proposed.shape!=(2,) or not np.isfinite(proposed).all():raise ValueError('Finite two-dimensional action required')
        result=execute_policy_action(self.core,self.guard,proposed,e.elapsed)
        if self.use_safety_context:self.context.record(result)
        command=np.asarray(result['command']);executed=np.array([command[0]/(.2 if command[0]>=0 else .08),command[1]/.5])
        self.guard_steps+=1;self.guard_overrides+=int(result['safety_override'])
        old_collisions=e.collision_events;old_guards=e.hard_guard_activations;old_travel=e.travel
        _,_,_,_,info=e.step(executed)
        obs,diagnostics=self.core.update(e.scan,e.observed_odometry,e.measured_velocity,e.elapsed)
        if self.use_safety_context:obs=self.context.augment(obs,self.core,self.guard,e.elapsed)
        distance=self.field.value(e.truth[:2]);progress=self.previous_distance-distance;self.previous_distance=distance
        finite=e.scan[np.isfinite(e.scan)&(e.scan>=e.lidar.range_min)]
        clearance=float(finite.min()) if len(finite) else e.lidar.range_max
        reward=15*progress-.05-.1*(e.travel-old_travel)-.01*np.sum((executed-self.previous_executed)**2)
        reward-=.05*max(0.,(.45-clearance)/.45)
        reward-=self.projection_cost*np.sum((proposed-executed)**2)
        body_penalty,body_gap=footprint_scan_penalty(e.scan,e.lidar,e.scene['robot_collision_box_m'][:2],self.footprint_clearance_cost)
        reward-=body_penalty
        self.previous_executed=executed.copy()
        failure=e.collision_events>old_collisions or e.hard_guard_activations>old_guards
        done=bool(diagnostics['estimated_goal_pose_reached'])
        verified=bool(done and info['true_goal_distance_m']<=.2 and not failure)
        if self.core.goal_yaw is not None:
            from neural_navigation.goal_pose import contract
            goal_contract=contract();true_yaw_error=float(abs(wrap(self.core.goal_yaw-e.truth[2])))
            true_velocity=np.asarray(info['actual_velocity'])
            verified=bool(verified and info['true_goal_distance_m']<goal_contract['position_tolerance_m']
                and true_yaw_error<goal_contract['yaw_tolerance_rad']
                and abs(true_velocity[0])<goal_contract['stopped_linear_m_s']
                and abs(true_velocity[1])<goal_contract['stopped_angular_rad_s'])
            info.update(true_goal_yaw_error_rad=true_yaw_error)
        truncated=bool(e.steps>=self.max_steps and not done)
        if failure:reward=min(reward,0.)-100.;done=True;truncated=False
        elif done:reward+=50. if verified else -50.
        elif truncated:reward-=5.
        error=self.core.estimate-e.truth;error[2]=wrap(error[2])
        info.update(mission_targets_reached=int(verified),verified_mission_success=verified,safety_termination=bool(failure),
                    estimated_pose=self.core.estimate.copy(),localization_error_m=float(np.linalg.norm(error[:2])),localization_error_yaw=float(abs(error[2])),
                    fixed_geodesic_progress_m=float(progress),training_stage=self.stage,training_scene=str(self.scene),training_case_json=self.training_case_json,
                    training_actions_overridden=bool(result['safety_override']),execution_mode='shared_sensor_core_and_guard',
                    execution_guard_overrides=self.guard_overrides,execution_guard_steps=self.guard_steps,
                    execution_guard_override_fraction=self.guard_overrides/self.guard_steps,
                    proposed_action=proposed.copy(),executed_action=executed.copy(),execution_guard_reason=result['stop_reason'],
                    measured_body_gap_m=body_gap,footprint_clearance_penalty=body_penalty,
                    commanded_velocity=command.copy(),sensor_core_diagnostics=diagnostics)
        return obs,float(reward),bool(done),bool(truncated),info
