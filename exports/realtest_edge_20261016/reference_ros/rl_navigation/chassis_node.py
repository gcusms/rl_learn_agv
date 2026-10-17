"""Single serial owner: raw IMU, encoder feedback, measured odometry, guarded commands."""
import json
import math
import time

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu
from std_msgs.msg import String

from .vendor.ros_robot_controller_sdk import Board
from .vendor.agv_wheels import EncoderReader, ZERO
from .vendor.mecanum_kinematics import MecanumKinematics
from .odometry import WheelOdometry, CommandOdometry, wheel_targets


class ChassisNode(Node):
    def __init__(self):
        super().__init__('rl_chassis')
        defaults = dict(device='/dev/rrc', baudrate=1000000, wheel_diameter=.1,
                        wheelbase=.2, track_width=.22, max_rps=1.0, cmd_timeout=.2,
                        enable_control=False, odom_source="command", gyro_variance=.01,
                        wheel_linear_variance=.0025, wheel_angular_variance=.04)
        self.settings = {k:self.declare_parameter(k,v).value for k,v in defaults.items()}
        s = self.settings
        if s["odom_source"] not in ("command", "encoder"):
            raise ValueError("odom_source must be command or encoder")
        for key in ('max_rps','cmd_timeout','gyro_variance','wheel_linear_variance','wheel_angular_variance'):
            if not math.isfinite(s[key]) or s[key] <= 0:
                raise ValueError(key+' must be finite and positive')
        self.geometry = MecanumKinematics(s['wheel_diameter'], s['wheelbase'], s['track_width'])
        self.odometry = WheelOdometry(self.geometry)
        self.command_odometry = CommandOdometry()
        self.executed_velocity = (0., 0., 0.)
        self.targets = ZERO
        self.command_at = None
        self.command_reset_id = None
        self.fault = None
        self.next_send = self.next_status = 0.
        self.odom_pub = self.create_publisher(Odometry, '/wheel/odom', 10)
        self.command_odom_pub = self.create_publisher(Odometry, '/command/odom', 10)
        self.selected_odom_pub = self.create_publisher(Odometry, '/rl_navigation/odom_raw', 10)
        self.imu_pub = self.create_publisher(Imu, '/ros_robot_controller/imu_raw', 10)
        self.feedback_pub = self.create_publisher(String, '/wheel/feedback', 10)
        self.status_pub = self.create_publisher(String, '/rl_navigation/chassis_status', 10)
        self.board = Board(s['device'], s['baudrate'], timeout=.05, exclusive=True)
        self.board.enable_reception(True)
        self.reader = EncoderReader(self.board, self.geometry)
        self.create_subscription(Twist, '/rl_navigation/cmd_vel', self.on_command, 10)
        self.create_timer(.01, self.tick)
        self.get_logger().info('Serial connected; enable_control='+str(s['enable_control']))

    def on_command(self, msg):
        state = self.reader.snapshot()
        try:
            if not self.settings['enable_control']:
                return
            if self.fault or not state.get('fresh') or not state.get('measurement_valid') or not state.get('flags',0)&1:
                raise ValueError('Encoder feedback/control gate unavailable')
            rps = wheel_targets(self.geometry, (msg.linear.x,msg.linear.y,msg.angular.z),self.settings['max_rps'])
            self.targets = list(enumerate(rps,1))
            self.command_at = time.monotonic()
            self.command_reset_id = state['mcu_resets']
        except ValueError as exc:
            self.targets = ZERO
            self.command_at = None
            self.get_logger().warning(str(exc))

    def tick(self):
        now = time.monotonic()
        try:
            self.reader.poll()
            state = self.reader.snapshot()
            stamp = self.get_clock().now().to_msg()
            velocity = self.odometry.update(state)
            if velocity is not None:
                self.publish_odometry(self.odometry.pose, velocity, self.odom_pub, 'encoder', stamp)
            imu = self.board.get_imu_sample()
            if imu is not None:
                values, received_ns = imu
                if len(values)==6 and all(math.isfinite(v) for v in values) and 0 <= (time.monotonic_ns()-received_ns)/1e9 < .2:
                    msg = Imu()
                    # Host reception timestamp (firmware IMU packet has no sample clock).
                    age = time.monotonic_ns()-received_ns
                    msg.header.stamp = rclpy.time.Time(nanoseconds=self.get_clock().now().nanoseconds-age).to_msg()
                    msg.header.frame_id = 'imu_link'
                    msg.orientation_covariance[0] = -1.
                    msg.linear_acceleration.x,msg.linear_acceleration.y,msg.linear_acceleration.z = [v*9.80665 for v in values[:3]]
                    msg.angular_velocity.x,msg.angular_velocity.y,msg.angular_velocity.z = [math.radians(v) for v in values[3:]]
                    for i in (0,4,8):
                        msg.angular_velocity_covariance[i] = self.settings['gyro_variance']
                        msg.linear_acceleration_covariance[i] = .01
                    self.imu_pub.publish(msg)
            if self.settings['enable_control'] and now >= self.next_send:
                good = (state.get('fresh') and state.get('measurement_valid') and state.get('flags',0)&1
                        and self.command_at is not None and now-self.command_at <= self.settings['cmd_timeout']
                        and state.get('mcu_resets') == self.command_reset_id and not self.fault)
                if not good:
                    self.targets = ZERO
                    self.command_at = None
                self.board.set_motor_speed(self.targets)
                self.executed_velocity = self.geometry.to_body_velocity([v for _,v in self.targets])
                self.next_send = now+.05
            command_velocity = self.command_odometry.update(now, self.executed_velocity)
            self.publish_odometry(self.command_odometry.pose, command_velocity,
                                  self.command_odom_pub, 'command', stamp)
            if now >= self.next_status:
                self.feedback_pub.publish(String(data=json.dumps(state,allow_nan=False)))
                self.status_pub.publish(String(data=json.dumps(dict(fresh=state.get('fresh'),
                    enable_control=self.settings['enable_control'],odom_source=self.settings['odom_source'],fault=self.fault,
                    encoder_rest=state.get('encoder_rest'),mcu_resets=state.get('mcu_resets')))))
                self.next_status = now+.1
        except Exception as exc:
            self.targets = ZERO
            self.command_at = None
            if self.fault != str(exc):
                self.get_logger().error('Chassis fault: '+str(exc))
            self.fault = str(exc)
            self.executed_velocity = (0., 0., 0.)
            self.command_odometry.update(now, self.executed_velocity)
            if self.settings['enable_control']:
                try:
                    self.board.set_motor_speed(ZERO)
                except Exception:
                    pass
            self.status_pub.publish(String(data=json.dumps({'fault':self.fault,'fresh':False})))

    def publish_odometry(self, pose, velocity, publisher, source, stamp):
        msg = Odometry()
        msg.header.stamp = stamp
        msg.header.frame_id = 'odom'
        msg.child_frame_id = 'base_footprint'
        x,y,yaw = pose
        msg.pose.pose.position.x, msg.pose.pose.position.y = x,y
        msg.pose.pose.orientation.z, msg.pose.pose.orientation.w = math.sin(yaw/2),math.cos(yaw/2)
        msg.twist.twist.linear.x, msg.twist.twist.linear.y, msg.twist.twist.angular.z = velocity
        for i in (0,7,14,21,28,35):
            msg.pose.covariance[i] = 1e3
            msg.twist.covariance[i] = 1e6
        msg.twist.covariance[0] = msg.twist.covariance[7] = self.settings['wheel_linear_variance']
        msg.twist.covariance[35] = self.settings['wheel_angular_variance']
        publisher.publish(msg)
        if source == self.settings['odom_source']:
            self.selected_odom_pub.publish(msg)

    def close(self):
        try:
            if self.settings['enable_control']:
                for _ in range(3):
                    self.board.set_motor_speed(ZERO)
        finally:
            self.board.close()


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = ChassisNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            try:
                node.close()
            finally:
                node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
