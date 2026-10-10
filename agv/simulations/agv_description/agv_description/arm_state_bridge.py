"""Read-only NexArm feedback -> JointState. Never sends motion commands."""
import hashlib
import json
import math
import time
from pathlib import Path

import rclpy
import yaml
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from ros_robot_controller_msgs.msg import ArmFullState
from ros_robot_controller_msgs.srv import GetArmIK
from sensor_msgs.msg import JointState
from std_msgs.msg import String

ARM_JOINTS = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'gripper_base_joint']
WHEEL_JOINTS = ['wheel_rf_joint', 'wheel_lf_joint', 'wheel_rb_joint', 'wheel_lb_joint']
HOME_KEYS = ['x', 'y', 'z', 'pitch', 'roll', 'claw']


def pulse_to_positions(pulses, config):
    if len(pulses) != 6 or not all(math.isfinite(float(p)) and 100 <= p <= 4095 for p in pulses):
        raise ValueError('Expected six valid measured native servo pulses (IDs 1..6)')
    result = [math.radians((float(p) - c) * s * d) + offset for p, c, s, d, offset in zip(
        pulses, config['servo_centers'], config['degrees_per_tick'], config['urdf_signs'], config['urdf_offsets_rad'])]
    # Preserve the URDF gripper mechanism: q=0 is open, q=0.5 closed.
    # Raw servo rotation exceeds this model's linkage range at fully open poses.
    result[5] = max(0.0, min(0.5, (pulses[5] - config['gripper_open_pulse']) / config['gripper_ticks_per_rad']))
    return result


def validate_home_solution(positions, home, config):
    """Check the returned IK pulses against CAD FK; reject stale board replies."""
    q1, q2, q3, q4, q5 = positions[:5]
    b, c = q2 + q3, q2 + q3 + q4
    length = config['tool_length_m']
    radial = .0348*math.cos(q2) + .22535*math.sin(q2) + .027316*math.cos(b) - .14239*math.sin(b) + length*math.cos(c)
    z = .110449 - .0348*math.sin(q2) + .22535*math.cos(q2) - .027316*math.sin(b) - .14239*math.cos(b) - length*math.sin(c)
    predicted = (radial*math.cos(q1), -radial*math.sin(q1), z)
    error = math.sqrt(sum((value-home[key]/1000.0)**2 for value,key in zip(predicted, ('x','y','z'))))
    angle_error = math.atan2(math.sin(-c-math.radians(home['pitch'])), math.cos(-c-math.radians(home['pitch'])))
    if error > .005 or abs(angle_error) > math.radians(2) or abs(q5-math.radians(home['roll'])) > math.radians(2):
        raise ValueError(f'Home IK/CAD FK mismatch: {error*1000:.2f} mm; refusing stale or incompatible IK reply')


def load_home(path):
    data = yaml.safe_load(Path(path).read_text(encoding='utf-8'))
    home = {k: float(data['home_pose'][k]) for k in HOME_KEYS}
    if not all(math.isfinite(v) for v in home.values()):
        raise ValueError('home_pose must contain finite millimeters / degrees')
    return home


class ArmStateBridge(Node):
    def __init__(self):
        super().__init__('agv_arm_state_bridge')
        defaults = {
            'init_pose_file': '/home/ubuntu/ros2_ws/src/driver/ros_arm_controller/config/init_pose.yaml',
            'mapping_file': '', 'controller_prefix': '/arm_controller',
            'cache_file': str(Path.home() / '.ros/agv_chassis_home.json'),
            'follow_feedback': True,
        }
        for key, value in defaults.items():
            self.declare_parameter(key, value)
        self.config = yaml.safe_load(Path(self.get_parameter('mapping_file').value).read_text())
        for key in ('servo_centers', 'degrees_per_tick', 'urdf_signs', 'urdf_offsets_rad'):
            values = self.config[key]
            if len(values) != 6 or not all(math.isfinite(float(v)) for v in values):
                raise ValueError(f'{key} must have six finite values')
        if not all(v > 0 for v in self.config['degrees_per_tick']):
            raise ValueError('degrees_per_tick must be positive')
        self.home = load_home(self.get_parameter('init_pose_file').value)
        self.cache = Path(self.get_parameter('cache_file').value)
        self.key = hashlib.sha256(json.dumps([self.home, self.config], sort_keys=True).encode()).hexdigest()
        self.positions = list(self.config['preview_radians'])
        self.home_positions = None
        self.mode = 'preview_waiting_for_home_ik'
        try:
            saved = json.loads(self.cache.read_text())
            if saved['key'] == self.key:
                cached_positions = pulse_to_positions(saved['servos'], self.config)
                validate_home_solution(cached_positions, self.home, self.config)
                self.positions = cached_positions
                self.home_positions = list(self.positions)
                self.mode = 'cached_home_ik'
        except (OSError, ValueError, KeyError, TypeError):
            pass
        self.last_feedback = None
        self.future = None
        self.future_started = 0.0
        self.ik_done = False
        self.warned_stale = False
        self.last_invalid_warning = -100.0
        self.wheels = dict.fromkeys(WHEEL_JOINTS, 0.0)
        prefix = '/' + self.get_parameter('controller_prefix').value.strip('/')
        self.publisher = self.create_publisher(JointState, '/agv_chassis/joint_states', 10)
        self.status = self.create_publisher(String, '/agv_chassis/arm_sync_status', 10)
        self.create_subscription(ArmFullState, prefix + '/arm/full_state', self.on_feedback, qos_profile_sensor_data)
        self.create_subscription(JointState, '/agv_chassis/wheel_joint_states', self.on_wheels, 10)
        self.client = self.create_client(GetArmIK, prefix + '/arm/get_ik')
        self.create_timer(1.0 / 30.0, self.publish)
        self.create_timer(2.0, self.poll_home)
        self.get_logger().info(f'Home from YAML (mm/deg): {self.home}; source={self.mode}')
        self.get_logger().info(f'Read-only feedback: {prefix}/arm/full_state; no hardware commands')

    def on_feedback(self, msg):
        try:
            positions = pulse_to_positions(msg.servos, self.config)
        except ValueError as exc:
            now = time.monotonic()
            if now - self.last_invalid_warning > 5:
                self.get_logger().warning(str(exc))
                self.last_invalid_warning = now
            return
        if not self.get_parameter('follow_feedback').value:
            return
        if self.last_feedback is None:
            self.get_logger().info('Live measured servo feedback now drives the model')
        self.positions = positions
        self.last_feedback = time.monotonic()
        self.mode = 'live_feedback'
        self.warned_stale = False

    def on_wheels(self, msg):
        if len(msg.name) != len(msg.position) or not all(math.isfinite(p) for p in msg.position):
            return
        for name, position in zip(msg.name, msg.position):
            if name in self.wheels:
                self.wheels[name] = float(position)

    def poll_home(self):
        if self.ik_done:
            return
        if self.future is not None:
            if not self.future.done():
                if time.monotonic() - self.future_started > 5:
                    self.client.remove_pending_request(self.future)
                    self.future.cancel()
                    self.future = None
                return
            try:
                result = self.future.result()
                if result is None or not result.success:
                    raise ValueError('Board home IK unavailable; retrying')
                home_positions = pulse_to_positions(result.servos, self.config)
                validate_home_solution(home_positions, self.home, self.config)
                self.home_positions = home_positions
                if self.last_feedback is None:
                    self.positions = home_positions
                    self.mode = 'board_home_ik'
                self.ik_done = True
                self.get_logger().info(f'Home IK pulses={list(result.servos)}, URDF radians={home_positions}')
                try:
                    self.cache.parent.mkdir(parents=True, exist_ok=True)
                    temporary = self.cache.with_suffix('.tmp')
                    temporary.write_text(json.dumps({'key': self.key, 'home': self.home, 'servos': list(result.servos)}, indent=2))
                    temporary.replace(self.cache)
                except OSError as exc:
                    self.get_logger().warning(f'Cannot cache home: {exc}')
            except Exception as exc:
                self.get_logger().warning(f'Home IK read failed: {exc}')
            finally:
                self.future = None
            return
        if self.client.service_is_ready():
            request = GetArmIK.Request()
            for key, value in self.home.items():
                setattr(request, key, value)
            self.future = self.client.call_async(request)
            self.future_started = time.monotonic()

    def publish(self):
        if self.last_feedback is not None and time.monotonic() - self.last_feedback > 2.0:
            self.mode = 'feedback_stale_holding_last_pose'
            if not self.warned_stale:
                self.get_logger().warning('Arm feedback stopped; holding last measured pose')
                self.warned_stale = True
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = WHEEL_JOINTS + ARM_JOINTS
        msg.position = [self.wheels[n] for n in WHEEL_JOINTS] + self.positions
        self.publisher.publish(msg)
        self.status.publish(String(data=self.mode))


def main(args=None):
    rclpy.init(args=args)
    node = ArmStateBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
