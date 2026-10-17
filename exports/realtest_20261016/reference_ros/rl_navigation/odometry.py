"""Continuous measured-wheel odometry without ROS dependencies."""
import math
from .vendor.wheel_feedback import signed_diff
from .vendor.mecanum_kinematics import MecanumKinematics


class WheelOdometry:
    def __init__(self, geometry):
        self.geometry = geometry
        self.pose = [0., 0., 0.]
        self.previous = None
        self.reset_id = None

    def update(self, state):
        """Return velocity only for a valid, consecutive MCU time interval.

        After gaps/reboots/CPR changes preserve pose and rebaseline; never invent
        motion over a missing interval. Count jumps are rejected.
        """
        if not state.get('fresh'):
            self.previous = None
            return None
        current = (state['timestamp_ms'], tuple(state['total']), tuple(state['cpr']))
        reset = state['mcu_resets']
        previous = self.previous
        self.previous = current
        if previous is None or reset != self.reset_id or current[2] != previous[2]:
            self.reset_id = reset
            return None
        dt = signed_diff(current[0], previous[0]) / 1000.
        if dt == 0:
            return None
        if not 0 < dt <= .2:
            return None
        turns = [signed_diff(a, b)/c for a, b, c in zip(current[1], previous[1], current[2])]
        dx, dy, angle = self.geometry.to_body_velocity(turns)
        if max(abs(dx), abs(dy)) > .1 or abs(angle) > .3:
            raise ValueError('Encoder count jump; rebaseline required')
        heading = self.pose[2] + angle/2
        self.pose[0] += dx*math.cos(heading)-dy*math.sin(heading)
        self.pose[1] += dx*math.sin(heading)+dy*math.cos(heading)
        self.pose[2] = (self.pose[2]+angle+math.pi) % (2*math.pi)-math.pi
        return dx/dt, dy/dt, angle/dt


def wheel_targets(geometry, command, maximum):
    if not all(math.isfinite(v) for v in command):
        raise ValueError('Non-finite velocity command')
    result = geometry.to_wheel_rps(*command)
    if max(map(abs, result)) > maximum:
        raise ValueError('Wheel command exceeds max_rps')
    return result


class CommandOdometry:
    """Integrate successfully sent body commands; this is not measured motion."""
    def __init__(self):
        self.pose = [0., 0., 0.]
        self.previous_time = None
        self.previous_velocity = (0., 0., 0.)

    def update(self, now, velocity):
        if not math.isfinite(now) or not all(math.isfinite(v) for v in velocity):
            raise ValueError('Non-finite command odometry input')
        if self.previous_time is not None:
            dt = now-self.previous_time
            if 0 < dt <= .2:
                vx,vy,wz = self.previous_velocity
                yaw = self.pose[2]
                self.pose[0] += (vx*math.cos(yaw)-vy*math.sin(yaw))*dt
                self.pose[1] += (vx*math.sin(yaw)+vy*math.cos(yaw))*dt
                self.pose[2] = (yaw+wz*dt+math.pi) % (2*math.pi)-math.pi
        self.previous_time = now
        self.previous_velocity = tuple(velocity)
        return self.previous_velocity
