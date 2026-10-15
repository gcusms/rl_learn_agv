"""Sensor-only command-integrated odometry challenge, with no motion truth."""
import numpy as np
from navrl.geometry import integrate


class CommandOdometryInput:
    def __init__(self,translation_scale=1.,yaw_scale=1.):
        self.pose=np.zeros(3);self.velocity=np.zeros(2)
        self.scale=np.array([translation_scale,yaw_scale],float)
        if not np.isfinite(self.scale).all() or min(self.scale)<=0:
            raise ValueError('Positive finite command integration scales required')
    def advance(self,published_command,dt):
        command=np.asarray(published_command,float)
        if command.shape!=(2,) or not np.isfinite(np.r_[command,dt]).all() or dt<=0:raise ValueError('Finite published velocity and positive dt required')
        self.velocity=command*self.scale
        self.pose=integrate(self.pose,*self.velocity,dt)


class CommandImuOdometryInput(CommandOdometryInput):
    """Command translation with a separately measured gyro increment/rate.

    The fusion weight approximates the recorded EKF sensor combination; it
    does not claim to reproduce the entire EKF. No measured translation is
    accepted by this interface, and no encoder is required.
    """
    def __init__(self,gyro_weight=.8,translation_scale=1.):
        super().__init__(translation_scale,1.)
        if not np.isfinite(gyro_weight) or not 0<=gyro_weight<=1:raise ValueError('Gyro weight must be in [0,1]')
        self.gyro_weight=gyro_weight
    def advance(self,published_command,dt,gyro_increment,gyro_rate):
        command=np.asarray(published_command,float)
        if command.shape!=(2,) or not np.isfinite(np.r_[command,dt,gyro_increment,gyro_rate]).all() or dt<=0:
            raise ValueError('Finite command and gyro measurements, positive dt required')
        rate=self.gyro_weight*gyro_rate+(1-self.gyro_weight)*command[1]
        average=self.gyro_weight*gyro_increment/dt+(1-self.gyro_weight)*command[1]
        linear=command[0]*self.scale[0]
        self.pose=integrate(self.pose,linear,average,dt);self.velocity=np.array([linear,rate])


def make_sensor_odometry(case):
    if case.get('input_odometry')=='command_integrated':
        return CommandOdometryInput(case.get('command_translation_scale',1.),case.get('command_yaw_scale',1.))
    if case.get('input_odometry')=='command_linear_imu_yaw':
        return CommandImuOdometryInput(case.get('gyro_fusion_weight',.8),case.get('command_translation_scale',1.))
    return None


def advance_sensor_odometry(observer,command,environment,previous_measured_yaw,case):
    if isinstance(observer,CommandImuOdometryInput):
        from navrl.geometry import wrap
        # Angular feedback is the simulator's existing noisy angular sensor.
        # Its integrated yaw is used only to emulate a gyro, never translation
        # or true pose. Truth remains confined to evaluation labels.
        bias=case.get('gyro_bias_rad_s',0.)
        increment=wrap(environment.observed_odometry[2]-previous_measured_yaw)+bias*environment.dt
        observer.advance(command,environment.dt,increment,environment.measured_velocity[1]+bias)
    else:observer.advance(command,environment.dt)
