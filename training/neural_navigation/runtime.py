"""Pure sensor-stream navigation with an explicitly logged emergency stop.

The guard never creates a route or substitutes a steering controller. It can
block reverse translation while retaining a verified safe, model-requested
rotation. Overrides must not be counted as learned obstacle avoidance.
"""
import json
import time
from pathlib import Path
import numpy as np
from navrl.geometry import integrate
from navrl.map import StaticMap
from neural_navigation.core import NavigationCore,PortablePolicy
from neural_navigation.sensors import LidarSpec
from neural_navigation.safety_memory import NearFieldMemory
from neural_navigation.safety_context import SafetyContext,validate_contract,OBSERVATION_DIM,PREDICTIVE_SCHEMAS
from neural_navigation.static_footprint import static_footprint_clear
from neural_navigation.motion_response import force_response_trajectory


class EmergencyStop:
    def __init__(self,footprint=(.393924904,.329460022),max_scan_age=.4,braking_acceleration=.3,memory_config=None):
        self.half=np.asarray(footprint,float)/2
        self.max_scan_age=max_scan_age;self.braking_acceleration=braking_acceleration
        self.reverse_started=None
        self.memory=NearFieldMemory(memory_config) if memory_config else None
        self.clearance_increasing_rotation=bool((memory_config or {}).get('clearance_increasing_rotation',False))
        config=memory_config or {}
        self.static_footprint_schema=config.get('static_footprint_schema')
        if self.static_footprint_schema not in (None,'oriented_rectangle_v1'):
            raise ValueError('Unsupported static footprint schema')
        self.static_margin=float(config.get('static_footprint_margin_m',.04))
        self.static_pose_sigma_scale=float(config.get('static_footprint_pose_sigma_scale',0.))
        if not np.isfinite([self.static_margin,self.static_pose_sigma_scale]).all() or self.static_margin<.04 or self.static_pose_sigma_scale<0:
            raise ValueError('Static margin >=0.04 and nonnegative pose uncertainty scale required')
        self.motion_response_schema=config.get('motion_response_schema')
        if self.motion_response_schema not in (None,'force_response_v1'):
            raise ValueError('Unsupported motion response schema')
        self.response_gain=float(config.get('response_gain_s_inv',20.))
        self.response_acceleration=np.array([config.get('response_linear_acceleration_m_s2',.5),
            config.get('response_angular_acceleration_rad_s2',1.)],float)
        if not np.isfinite(np.r_[self.response_gain,self.response_acceleration]).all() or self.response_gain<=0 or min(self.response_acceleration)<=0:
            raise ValueError('Positive finite motion response parameters required')
        self.predictive_constraint=None
        if config.get('predictive_action_constraint') is not None:
            from neural_navigation.predictive_constraint import PredictiveConstraint
            self.predictive_constraint=PredictiveConstraint(config['predictive_action_constraint'],self.half*2,self.braking_acceleration)

    def _trajectory(self,core,command,age):
        horizon=abs(command[0])/self.braking_acceleration+max(age,0)+.2
        if self.motion_response_schema is None:
            for t in np.linspace(0,horizon,9):
                yield integrate(np.zeros(3),*command,t),integrate(core.estimate,*command,t),horizon
            return
        velocity=np.asarray(core.measured_velocity,float)
        horizon=max(horizon,abs(velocity[0])/self.braking_acceleration+max(age,0)+.2,
            abs(velocity[1]-command[1])/self.response_acceleration[1]+max(age,0)+.2)
        poses=force_response_trajectory(velocity,command,horizon,
            self.response_gain,self.response_acceleration)
        c,s=np.cos(core.estimate[2]),np.sin(core.estimate[2]);rotation=np.array([[c,s],[-s,c]])
        for local in poses:
            world=np.r_[core.estimate[:2]+local[:2]@rotation,core.estimate[2]+local[2]]
            yield local,world,horizon

    def _static_pose_clear(self,core,world,command,horizon):
        if self.static_footprint_schema is None:
            return bool(core.static_map.lookup(world[:2],core.static_map.free,outside=False))
        radius=np.linalg.norm(self.half+self.static_margin)
        # Nine samples: cover the maximum body displacement to the closest
        # sample throughout each interval, including rotational corner motion.
        speed=np.abs(command)
        if self.motion_response_schema is not None:speed=np.maximum(speed,np.abs(core.measured_velocity))
        sweep_margin=(speed[0]+speed[1]*radius)*horizon/16.
        std=np.asarray(core.localizer.std,float)
        pose_margin=self.static_pose_sigma_scale*(max(std[:2])+radius*std[2])
        return static_footprint_clear(core.static_map,world,self.half*2,
            self.static_margin+sweep_margin+pose_margin)

    def _trajectory_check(self,core,command,points,margins,age,allow_escape=False):
        initial=np.linalg.norm(points-np.clip(points,-self.half,self.half),axis=1)
        rotation_escape=bool(self.clearance_increasing_rotation and abs(command[0])<1e-8 and abs(command[1])>.005 and np.all(initial>0))
        escaping=bool(allow_escape and (command[0]>.005 or rotation_escape) and np.any(initial<margins))
        minimum=np.minimum(initial,margins) if escaping else margins
        for local,world,horizon in self._trajectory(core,command,age):
            if not self._static_pose_clear(core,world,command,horizon):return 'static_braking_footprint'
            c,s=np.cos(local[2]),np.sin(local[2]);relative=(points-local[:2])@np.array([[c,-s],[s,c]])
            gap=np.linalg.norm(relative-np.clip(relative,-self.half,self.half),axis=1)
            if np.any(gap<minimum-1e-4):return 'laser_braking_footprint'
        if escaping and np.any(gap[initial<margins]<initial[initial<margins]+.005):return 'escape_not_increasing_clearance'
        return None
    def check(self,core,command,scan_timestamp,now,scan_valid=True,rear_clearance_known=False):
        command=np.asarray(command,float).copy()
        reason=None;projection_reason=None;age=now-scan_timestamp
        if not np.isfinite(command).all() or not np.isfinite([scan_timestamp,now]).all():reason='nonfinite_input'
        elif self.motion_response_schema is not None and (
            np.asarray(getattr(core,'measured_velocity',[])).shape!=(2,) or
            not np.isfinite(core.measured_velocity).all()):reason='nonfinite_input'
        elif age<-.05 or age>self.max_scan_age:reason='stale_or_future_scan'
        elif not scan_valid or not np.any(np.isfinite(core.scan)&(core.scan>=core.lidar.range_min)&(core.scan<core.lidar.range_max)):reason='scan_unavailable'
        elif core.odometry_sanity_failure:
            reason='odometry_jump'
            if self.memory:self.memory.reset()
        elif not np.isfinite(core.localizer.std).all() or max(core.localizer.std[:2])>.2 or core.localizer.std[2]>.25:reason='localization_uncertain'
        elif not core.diagnostics()['route_complete']:
            if self.memory:command[0]=0.;projection_reason='learned_route_incomplete'
            else:reason='learned_route_incomplete'
        elif command[0]<-.001:
            if not rear_clearance_known:
                command[0]=0.;projection_reason='reverse_without_rear_visibility'
            elif self.reverse_started is None:self.reverse_started=now
            elif now-self.reverse_started>1.5:
                command[0]=0.;projection_reason='reverse_duration_limit'
        else:self.reverse_started=None
        if reason is None and self.predictive_constraint is not None:
            reason=self.predictive_constraint.rejection(core,command,age)
        if reason is None:
            valid=np.isfinite(core.scan)&(core.scan>=core.lidar.range_min)&(core.scan<core.lidar.range_max)
            m=core.lidar.mount;angle=core.lidar.angles[valid]+m[3]
            points=np.c_[core.scan[valid]*np.cos(angle)+m[0],core.scan[valid]*np.sin(angle)+m[1]]
            if self.memory is not None:
                remembered,uncertainty=self.memory.update(core)
                margins=np.r_[np.full(len(points),.04),.04+uncertainty]
                points=np.vstack([points,remembered])
                unsafe=self._trajectory_check(core,command,points,margins,age,allow_escape=True)
                if unsafe:
                    # Preserve only safe components of the model's proposal;
                    # never invent a turn, speed, route or recovery controller.
                    accepted=False
                    for candidate,label in [(np.array([command[0],0.]),'unsafe_turn_blocked'),(np.array([0.,command[1]]),'unsafe_translation_blocked')]:
                        if np.allclose(candidate,command) or np.allclose(candidate,0.):continue
                        if self._trajectory_check(core,candidate,points,margins,age,allow_escape=True) is None:
                            command=candidate;projection_reason=label;accepted=True;break
                    if not accepted:reason=unsafe
                return (np.zeros(2) if reason else command.copy()),reason or projection_reason
            # Braking and one control-period latency, in the robot frame.
            for local,world,horizon in self._trajectory(core,command,age):
                if not self._static_pose_clear(core,world,command,horizon):
                    reason='static_braking_footprint';break
                if len(points):
                    c,s=np.cos(local[2]),np.sin(local[2]);relative=(points-local[:2])@np.array([[c,-s],[s,c]])
                    gap=np.linalg.norm(relative-np.clip(relative,-self.half,self.half),axis=1)
                    if np.min(gap)<.04:reason='laser_braking_footprint';break
        return (np.zeros(2) if reason else command.copy()),reason or projection_reason


def execute_policy_action(core,guard,proposed,timestamp,received_timestamp=None,scan_valid=True,rear_clearance_known=False):
    """Shared deployment/training action transform; records what was executed.

Training must use the proposed latent action's PPO likelihood and the return
from this same deterministic transform, and report its override frequency.
"""
    proposed=np.asarray(proposed,float);command=core.velocity_command(proposed)
    if abs(timestamp-core.elapsed)>1e-8:raise ValueError('Action must refer to the current sensor frame')
    now=timestamp if received_timestamp is None else received_timestamp
    executed,reason=guard.check(core,command,timestamp,now,scan_valid,rear_clearance_known)
    diagnostics=core.diagnostics()
    if diagnostics['estimated_goal_pose_reached'] and reason in (None,'learned_route_incomplete'):
        executed=np.zeros(2);reason='estimated_goal_reached'
    executed_action=np.array([executed[0]/(.2 if executed[0]>=0 else .08),executed[1]/.5])
    core.record_action(executed_action)
    result=dict(timestamp=float(timestamp),proposed_action=proposed.tolist(),model_command=command.tolist(),
                command=executed.tolist(),safety_override=reason not in (None,'estimated_goal_reached'),
                stop_reason=reason,diagnostics=diagnostics,guard_memory_points=len(guard.memory.entries) if guard.memory else 0,
                uncertainty_calibrated=False,braking_model_calibrated=False)
    if getattr(guard,'predictive_constraint',None) is not None:
        result['predictive_constraint']=guard.predictive_constraint.diagnostics(core)
    return result


class SensorNavigationRuntime:
    def __init__(self,bundle,map_path,initial_pose,goal,route_mode='raw'):
        bundle=Path(bundle);hardware=json.loads((bundle/'hardware.json').read_text())
        if len(goal)==3 and not (bundle/'goal_pose_observation.json').exists():
            raise ValueError('Directional goals require a goal-pose trained bundle; legacy actor is position-only')
        static_map=map_path if isinstance(map_path,StaticMap) else StaticMap(map_path,radius=.3)
        self.core=NavigationCore(static_map,LidarSpec(hardware['lidar']),bundle,route_mode=route_mode)
        self.core.reset(initial_pose,goal)
        self.context=None
        if 'policy_observation' in hardware:
            validate_contract(hardware['policy_observation'])
            self.context=SafetyContext(hardware['policy_observation']['schema'])
        dimensions=hardware.get('policy_observation',{}).get('dimensions',235)
        if 'policy_action' in hardware:
            from neural_navigation.masked_actions import validate_hardware_action
            validate_hardware_action(hardware)
            if self.context is None or self.context.schema not in PREDICTIVE_SCHEMAS:
                raise ValueError('Mismatched masked velocity action/observation contract')
        self.policy=PortablePolicy(bundle/'actor.pt',dimensions)
        self.guard=EmergencyStop(hardware['robot_footprint_m'],memory_config=hardware.get('near_field_memory'))
    def step(self,scan,odom,velocity,timestamp,received_timestamp=None,scan_valid=True,rear_clearance_known=False,capture_observation=False):
        observation,diagnostics=self.core.update(scan,odom,velocity,timestamp)
        if self.context is not None:
            context_started=time.perf_counter()
            previous_reason=self.context.last_reason
            previous_executed=self.context.last_executed.tolist()
            observation=self.context.augment(observation,self.core,self.guard,timestamp,
                received_timestamp=received_timestamp,scan_valid=scan_valid,rear_clearance_known=rear_clearance_known)
            context_ms=(time.perf_counter()-context_started)*1000
        result=execute_policy_action(self.core,self.guard,self.policy.predict(observation),timestamp,received_timestamp,scan_valid,rear_clearance_known)
        if self.context is not None:
            self.context.record(result)
            result['policy_observation_dimensions']=len(observation)
            result['action_feasibility']=observation[235:244].tolist()
            result['policy_previous_stop_reason']=previous_reason
            result['policy_previous_executed_action']=previous_executed
            result['policy_wait_fraction']=float(observation[OBSERVATION_DIM-1])
            if self.context.schema in PREDICTIVE_SCHEMAS:
                result['fast_probe_feasibility']=observation[OBSERVATION_DIM:OBSERVATION_DIM+2].tolist()
                result['predictive_risk_features']=observation[OBSERVATION_DIM+2:].tolist()
            result['safety_context_ms']=context_ms
        if capture_observation:
            # Opt-in offline diagnostics: preserve the exact actor input,
            # before action execution updated the previous-action history.
            result['policy_observation']=observation.tolist()
        return result
