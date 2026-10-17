"""Small causal planar wheel/gyro frontend, with explicit continuity failures.

Inputs are measured body-frame velocities in one acquisition clock domain.
No command integration, absolute wheel pose, IMU acceleration or map pose is
fused. Wheel slip and gyro bias remain possible; covariance is provisional.
"""
from collections import deque
import numpy as np
from navrl.geometry import wrap


def contract():
    return dict(schema='measured_wheel_gyro_odometry_v1', experimental=True,
        wheel_max_age_s=.15, imu_max_age_s=.05, output_max_age_s=.06,
        maximum_interval_s=.10, future_tolerance_s=1e-6,
        maximum_linear_m_s=.6, maximum_yaw_rad_s=2.,
        wheel_linear_variance=.0025, wheel_yaw_variance=.04, gyro_variance=.01,
        rest_linear_m_s=.005, rest_yaw_rad_s=.01, rest_gyro_rad_s=.03,
        rest_span_s=2., rest_gyro_std_rad_s=.01,
        bias_time_constant_s=20., maximum_bias_rad_s=.02,
        gap_std_per_sqrt_s=[.10,.10,.10],
        measured_lateral_velocity_retained=True, command_velocity_used=False,
        repeated_pose_and_twist_fused=False, uncertainty_calibrated=False,
        input_frame='base_footprint', timestamps='one acquisition clock domain')


class MeasuredOdometry:
    def __init__(self, config=None):
        self.config=contract() if config is None else config
        if self.config!=contract():raise ValueError('Unsupported measured odometry contract')
        self.reset()

    def reset(self, pose=None):
        self.pose=np.zeros(3) if pose is None else np.asarray(pose,float).copy()
        if self.pose.shape!=(3,) or not np.isfinite(self.pose).all():raise ValueError('Finite planar pose required')
        self.pose[2]=wrap(self.pose[2])
        self.timestamp=None;self.previous_velocity=None;self.velocity=np.zeros(3)
        self.covariance=np.zeros((3,3));self.gyro_bias=0.
        self.rest=deque(maxlen=1000);self.bias_updates=0
        self.continuity_lost=False;self.reason='uninitialized'
        self.last_wheel_stamp=None;self.last_imu_stamp=None

    def _failure(self, timestamp, dt, reason):
        self.timestamp=float(timestamp);self.previous_velocity=None;self.velocity=np.zeros(3)
        self.continuity_lost=True;self.reason=reason;self.rest.clear()
        self.covariance+=np.diag(np.array(self.config['gap_std_per_sqrt_s'])**2*max(dt,0.))
        return self.state(timestamp)

    def update(self, timestamp, wheel_velocity, wheel_timestamp, gyro_z, imu_timestamp,
               wheel_valid=True, imu_valid=True):
        cfg=self.config
        if not np.isfinite(timestamp):raise ValueError('Finite timestamp required')
        dt=0. if self.timestamp is None else timestamp-self.timestamp
        if dt<0:raise ValueError('Out-of-order fusion timestamp')
        if self.timestamp is not None and dt==0:
            return self.state(timestamp)  # Duplicate delivery never integrates twice.
        wheel=np.asarray(wheel_velocity,float)
        if wheel.shape!=(3,):raise ValueError('Measured wheel vx, vy, wz required')
        values=np.r_[wheel,wheel_timestamp,gyro_z,imu_timestamp]
        if not wheel_valid or not imu_valid or not np.isfinite(values).all():
            return self._failure(timestamp,dt,'invalid_motion_measurement')
        if not (-cfg['future_tolerance_s']<=timestamp-wheel_timestamp<=cfg['wheel_max_age_s']):
            return self._failure(timestamp,dt,'stale_or_future_wheel')
        if not (-cfg['future_tolerance_s']<=timestamp-imu_timestamp<=cfg['imu_max_age_s']):
            return self._failure(timestamp,dt,'stale_or_future_imu')
        if ((self.last_wheel_stamp is not None and wheel_timestamp<self.last_wheel_stamp) or
            (self.last_imu_stamp is not None and imu_timestamp<self.last_imu_stamp)):
            return self._failure(timestamp,dt,'out_of_order_sensor')
        if np.linalg.norm(wheel[:2])>cfg['maximum_linear_m_s'] or max(abs(wheel[2]),abs(gyro_z))>cfg['maximum_yaw_rad_s']:
            return self._failure(timestamp,dt,'motion_out_of_range')
        self.last_wheel_stamp=float(wheel_timestamp);self.last_imu_stamp=float(imu_timestamp)
        if dt>cfg['maximum_interval_s']:
            return self._failure(timestamp,dt,'motion_interval_gap')
        resting=(max(abs(wheel[:2]))<cfg['rest_linear_m_s'] and abs(wheel[2])<cfg['rest_yaw_rad_s']
                 and abs(gyro_z)<cfg['rest_gyro_rad_s'])
        if resting:
            self.rest.append((float(timestamp),float(gyro_z)))
            while len(self.rest)>1 and timestamp-self.rest[1][0]>=cfg['rest_span_s']:
                self.rest.popleft()
            if self.rest and timestamp-self.rest[0][0]>=cfg['rest_span_s']:
                sample=np.array([v for _,v in self.rest])
                if sample.std()<=cfg['rest_gyro_std_rad_s']:
                    target=np.clip(np.median(sample),-cfg['maximum_bias_rad_s'],cfg['maximum_bias_rad_s'])
                    self.gyro_bias+=(1-np.exp(-dt/cfg['bias_time_constant_s']))*(target-self.gyro_bias)
                    self.bias_updates+=1
        else:self.rest.clear()
        gyro_weight=cfg['wheel_yaw_variance']/(cfg['wheel_yaw_variance']+cfg['gyro_variance'])
        velocity=wheel.copy()
        velocity[2]=gyro_weight*(gyro_z-self.gyro_bias)+(1-gyro_weight)*wheel[2]
        if self.previous_velocity is not None:
            average=(self.previous_velocity+velocity)/2
            angle=average[2]*dt;heading=self.pose[2]+angle/2
            c,s=np.cos(heading),np.sin(heading);rotation=np.array([[c,-s],[s,c]])
            delta=rotation@average[:2]*dt
            transition=np.eye(3);transition[:2,2]=[-delta[1],delta[0]]
            noise=np.diag([cfg['wheel_linear_variance'],cfg['wheel_linear_variance'],
                1/(1/cfg['wheel_yaw_variance']+1/cfg['gyro_variance'])])*dt**2
            self.covariance=transition@self.covariance@transition.T+noise
            self.pose[:2]+=delta;self.pose[2]=wrap(self.pose[2]+angle)
        self.timestamp=float(timestamp);self.previous_velocity=velocity.copy();self.velocity=velocity
        self.reason='continuity_requires_reset' if self.continuity_lost else 'ok'
        return self.state(timestamp)

    def state(self, timestamp):
        if not np.isfinite(timestamp):raise ValueError('Finite query timestamp required')
        age=None if self.timestamp is None else float(timestamp-self.timestamp)
        fresh=age is not None and -self.config['future_tolerance_s']<=age<=self.config['output_max_age_s']
        valid=bool(fresh and not self.continuity_lost and self.reason=='ok')
        reason=self.reason if fresh or self.timestamp is None else 'stale_or_future_motion_output'
        return dict(pose=self.pose.copy(),velocity=self.velocity.copy(),valid=valid,reason=reason,
            age_s=age,gyro_bias_rad_s=float(self.gyro_bias),bias_updates=self.bias_updates,
            continuity_lost=self.continuity_lost,uncertainty_calibrated=False,
            std=np.sqrt(np.diag(self.covariance)))


class MeasuredNavigationAdapter:
    """Explicit opt-in adapter; the caller still owns publications/watchdogs.

    Deliver wheel/gyro updates causally, then call step at the scan's time.
    Invalid or discontinuous motion input returns stop without invoking the
    policy or advancing map localization. No hidden fallback to commands.
    """
    def __init__(self, runtime, odometry=None):
        self.runtime=runtime
        self.odometry=MeasuredOdometry() if odometry is None else odometry

    def step(self, scan, timestamp, received_timestamp=None, **kwargs):
        state=self.odometry.state(timestamp)
        diagnostic={key:value.tolist() if isinstance(value,np.ndarray) else value for key,value in state.items()}
        if not state['valid']:
            self.runtime.core.record_action(np.zeros(2))
            context=getattr(self.runtime,'context',None)
            if context is not None:
                context.last_executed=np.zeros(2);context.last_reason='motion_input_unavailable'
            return dict(timestamp=float(timestamp),command=[0.,0.],safety_override=True,
                stop_reason='motion_input_unavailable',motion_input=diagnostic,
                actual_publication_owned_by_caller=True)
        result=self.runtime.step(scan,state['pose'],state['velocity'][[0,2]],timestamp,
                                 received_timestamp=received_timestamp,**kwargs)
        result['motion_input']=diagnostic
        return result
