"""ROS-free kinematics matching driver/controller/controller/mecanum.py."""
from dataclasses import dataclass
import math

from .wheel_feedback import body_velocity


@dataclass(frozen=True)
class MecanumKinematics:
    wheel_diameter: float
    wheelbase: float
    track_width: float

    def __post_init__(self):
        if not all(math.isfinite(x) and x > 0 for x in
                   (self.wheel_diameter, self.wheelbase, self.track_width)):
            raise ValueError('Wheel diameter, wheelbase and track width must be positive finite meters')

    def to_wheel_rps(self, vx, vy, wz):
        """m/s, m/s, rad/s -> raw rps in LF, LR, RF, RR order."""
        if not all(math.isfinite(x) for x in (vx, vy, wz)):
            raise ValueError('Body velocity must be finite')
        k = (self.wheelbase + self.track_width) / 2
        circumference = math.pi * self.wheel_diameter
        wheels = ((vx - vy - k * wz) / circumference,
                  (vx + vy - k * wz) / circumference,
                  -(vx + vy + k * wz) / circumference,
                  -(vx - vy + k * wz) / circumference)
        if not all(math.isfinite(x) for x in wheels):
            raise ValueError('Wheel velocity conversion overflow')
        return wheels

    def to_body_velocity(self, rps):
        """Measured raw wheel rps -> kinematic vx, vy, wz (does not model slip)."""
        if len(rps) != 4 or not all(math.isfinite(x) for x in rps):
            raise ValueError('Four finite wheel speeds required')
        result = body_velocity(rps, self.wheel_diameter, self.wheelbase,
                               self.track_width, chassis='mecanum')
        if not all(math.isfinite(x) for x in result):
            raise ValueError('Body velocity conversion overflow')
        return result
