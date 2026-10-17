"""Own rosbag and analysis logs without publishing any motion command."""
import json
import math
import os
import shutil
import signal
import subprocess
import threading
import time
from pathlib import Path

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from rcl_interfaces.srv import ListParameters, GetParameters
from rosidl_runtime_py.utilities import get_message
from rosidl_runtime_py.convert import message_to_ordereddict
import yaml

from .recording import TOPICS, LATCHED, RecordingStore, write_json


def parameter_value(value):
    fields = {1:'bool_value', 2:'integer_value', 3:'double_value', 4:'string_value',
              5:'byte_array_value', 6:'bool_array_value', 7:'integer_array_value',
              8:'double_array_value', 9:'string_array_value'}
    if value.type not in fields: return None
    result = getattr(value, fields[value.type])
    if value.type == 5: return [v[0] if isinstance(v, bytes) else int(v) for v in result]
    return list(result) if value.type >= 6 else result


def safe_json(value):
    # ROS messages may contain NaN covariance/laser values. Preserve their meaning.
    if isinstance(value, float) and not math.isfinite(value):
        return 'nan' if math.isnan(value) else 'inf' if value > 0 else '-inf'
    if isinstance(value, dict): return {k:safe_json(v) for k,v in value.items()}
    if isinstance(value, (list, tuple)): return [safe_json(v) for v in value]
    return value


class RecordNode(Node):
    def __init__(self):
        super().__init__('rl_recorder')
        defaults = dict(output_root='/home/ubuntu/rl_ws/recordings', session_name='',
                        duration=0., min_free_gb=2., split_duration=300, split_size_mb=512)
        self.options = {k:self.declare_parameter(k,v).value for k,v in defaults.items()}
        for key in ('duration','min_free_gb','split_duration','split_size_mb'):
            value = self.options[key]
            if not math.isfinite(value) or value < 0: raise ValueError('Invalid '+key)
        root = Path(self.options['output_root']).expanduser()
        root.mkdir(parents=True, exist_ok=True)
        if shutil.disk_usage(root).free < self.options['min_free_gb']*1024**3:
            raise RuntimeError('Not enough free disk space to start recording')
        self.store = RecordingStore(root, self.options['session_name'])
        self.started = time.monotonic(); self.should_stop = None; self.closed = False
        self.bag = None; self.bag_thread = None; self.bag_ready = threading.Event()
        self.ready_announced = False; self.received = {}; self.last_chassis = None
        self.param_pending = {}; self.param_retry_at = {}; self.param_clients = {}
        self.next_snapshot = 0.; self.model_warning_sent = False
        self.store.session.update(recorder_options=self.options,
            environment={k:os.environ.get(k) for k in ('ROS_DOMAIN_ID','ROS_DISTRO','RMW_IMPLEMENTATION','ROS_LOCALHOST_ONLY')})
        self.store.submit('recorder_start', self.options, self.get_clock().now().nanoseconds)
        try:
            self._snapshot_package()
            self.create_record_subscriptions()
            qos = {topic:dict(history='keep_last', depth=100,
                        reliability='reliable' if topic in LATCHED or topic == '/rl_navigation/decision' else 'best_effort',
                        durability='transient_local' if topic in LATCHED else 'volatile') for topic in TOPICS}
            qos_path = self.store.path/'recording_qos.yaml'
            qos_path.write_text(yaml.safe_dump(qos))
            command = ['ros2','bag','record','-s','sqlite3','-o',str(self.store.path/'rosbag'),
                '--include-unpublished-topics','--max-cache-size','33554432',
                '--qos-profile-overrides-path',str(qos_path),
                '--max-bag-duration',str(int(self.options['split_duration'])),
                '--max-bag-size',str(int(self.options['split_size_mb'])*1024**2), *TOPICS]
            self.store.session['bag_command'] = command
            self.bag = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, text=True, bufsize=1, start_new_session=True,
                env=dict(os.environ, PYTHONUNBUFFERED='1', RCUTILS_COLORIZED_OUTPUT='0'))
            self.bag_thread = threading.Thread(target=self._read_bag, name='rosbag_output', daemon=True)
            self.bag_thread.start()
            self.create_timer(.5, self.monitor)
            self.get_logger().info('Recording directory: '+str(self.store.path))
        except Exception:
            self.should_stop = 'startup_error'; self.close(); raise

    def _snapshot_package(self):
        from ament_index_python.packages import get_package_share_directory
        share = Path(get_package_share_directory('rl_navigation'))
        target = self.store.path/'snapshots/package'
        for name in ('launch','config','rviz'):
            shutil.copytree(share/name, target/name, ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
        shutil.copytree(Path(__file__).parent, target/'rl_navigation',
                        ignore=shutil.ignore_patterns('__pycache__','*.pyc'))

    def create_record_subscriptions(self):
        for topic in ('/rl_navigation/model_status','/rl_navigation/decision','/rl_navigation/session_metadata',
                      '/rl_navigation/chassis_status','/rl_navigation/cmd_vel','/initialpose','/goal_pose'):
            qos = QoSProfile(depth=100,
                reliability=ReliabilityPolicy.RELIABLE,
                durability=DurabilityPolicy.TRANSIENT_LOCAL if topic in LATCHED else DurabilityPolicy.VOLATILE)
            self.create_subscription(get_message(TOPICS[topic]), topic,
                lambda message, topic=topic:self.on_message(topic,message), qos)

    def on_message(self, topic, message):
        now = self.get_clock().now().nanoseconds
        self.received[topic] = self.received.get(topic, 0)+1
        try:
            if topic == '/rl_navigation/cmd_vel':
                self.store.submit('command', safe_json(dict(velocity=[message.linear.x,message.linear.y,message.angular.z],
                    semantics='published Twist, not MCU acknowledgement')), now)
            elif topic in ('/initialpose','/goal_pose'):
                self.store.submit(topic.strip('/'), safe_json(message_to_ordereddict(message)), now)
            else:
                data = json.loads(message.data)
                kind = {'/rl_navigation/model_status':'status','/rl_navigation/decision':'decision',
                        '/rl_navigation/session_metadata':'metadata'}.get(topic,'chassis_status')
                if kind == 'chassis_status':
                    signature = json.dumps(data,sort_keys=True)
                    if signature == self.last_chassis: return
                    self.last_chassis = signature
                self.store.submit(kind,safe_json(data),now)
        except (ValueError,TypeError) as exc:
            self.store.submit('invalid_message',dict(topic=topic,error=str(exc)),now)

    def _read_bag(self):
        with (self.store.path/'rosbag.log').open('w',buffering=1) as stream:
            for line in self.bag.stdout:
                stream.write(line)
                if 'Recording...' in line: self.bag_ready.set()

    def monitor(self):
        elapsed = time.monotonic()-self.started
        if self.bag.poll() is not None:
            self.should_stop = 'rosbag_exited'; return
        if self.store.error:
            self.get_logger().error(self.store.error)
            self.should_stop = 'log_writer_error'; return
        if shutil.disk_usage(self.store.path).free < self.options['min_free_gb']*1024**3:
            self.get_logger().error('Low disk space; stopping this recording')
            self.should_stop = 'low_disk_space'; return
        if self.options['duration'] and elapsed >= self.options['duration']:
            self.should_stop = 'duration_reached'; return
        if self.bag_ready.is_set() and not self.ready_announced:
            self.ready_announced = True
            self.store.session['bag_ready_after_s'] = elapsed
            self.store.submit('bag_ready',dict(elapsed_s=elapsed),self.get_clock().now().nanoseconds)
            print('RL_RECORDING_READY '+str(self.store.path),flush=True)
        elif not self.bag_ready.is_set() and elapsed > 20:
            self.should_stop = 'rosbag_start_timeout'; return
        if elapsed > 12 and not self.received.get('/rl_navigation/session_metadata') and not self.model_warning_sent:
            self.model_warning_sent = True
            self.get_logger().warning('No model metadata yet; start/restart the updated navigation to capture model snapshots')
        self._poll_parameters()

    def _poll_parameters(self):
        now = time.monotonic()
        for node, pending in list(self.param_pending.items()):
            future, stage, names, started = pending
            if not future.done():
                if now-started > 5:
                    future.cancel(); del self.param_pending[node]; self.param_retry_at[node] = now+10
                continue
            del self.param_pending[node]
            try:
                response = future.result()
                if stage == 'list':
                    names = response.result.names
                    request = GetParameters.Request(names=names)
                    future = self.param_clients[node][1].call_async(request)
                    self.param_pending[node] = (future,'get',names,now)
                else:
                    self.store.submit('parameters',dict(node=node,
                        values={key:safe_json(parameter_value(v)) for key,v in zip(names,response.values)}),
                        self.get_clock().now().nanoseconds)
                    self.param_retry_at[node] = now+30
            except Exception as exc:
                self.store.submit('parameter_snapshot_error',dict(node=node,error=str(exc)))
                self.param_retry_at[node] = now+10
        if now < self.next_snapshot: return
        self.next_snapshot = now+2
        targets = {'/rl_model','/rl_chassis','/ekf_filter_node','/map_server','/imu_calib','/imu_filter','/robot_state_publisher'}
        for topic in ('/scan','/scan_raw','/imu_corrected','/imu'):
            for endpoint in self.get_publishers_info_by_topic(topic):
                targets.add(endpoint.node_namespace.rstrip('/')+'/'+endpoint.node_name)
        for node in targets:
            if node in self.param_pending or now < self.param_retry_at.get(node,0): continue
            if node not in self.param_clients:
                self.param_clients[node] = (self.create_client(ListParameters,node+'/list_parameters'),
                                           self.create_client(GetParameters,node+'/get_parameters'))
            listing, getting = self.param_clients[node]
            if listing.service_is_ready() and getting.service_is_ready():
                self.param_pending[node] = (listing.call_async(ListParameters.Request(depth=0)),'list',[],now)

    def close(self):
        if self.closed: return
        self.closed = True
        stop_reason = self.should_stop or 'user_interrupt'
        forced = False
        if self.bag and self.bag.poll() is None:
            os.killpg(self.bag.pid,signal.SIGINT)
            try: self.bag.wait(timeout=15)
            except subprocess.TimeoutExpired:
                forced = True; os.killpg(self.bag.pid,signal.SIGTERM)
                try: self.bag.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    os.killpg(self.bag.pid,signal.SIGKILL); self.bag.wait(timeout=3)
        if self.bag_thread: self.bag_thread.join(timeout=2)
        self.store.submit('recorder_stop',dict(reason=stop_reason,forced=forced))
        self.store.close(dict(state='closed', stop_reason=stop_reason,forced_bag_shutdown=forced,
            bag_exit_code=self.bag.returncode if self.bag else None,
            analysis_received_counts=self.received,
            parameter_nodes_missing=[n for n in self.param_clients if n not in self.store.parameter_hashes]))
        print('RL_RECORDING_SAVED '+str(self.store.path),flush=True)


def main(args=None):
    from rclpy.signals import SignalHandlerOptions
    from rclpy.executors import ExternalShutdownException
    rclpy.init(args=args,signal_handler_options=SignalHandlerOptions.NO)
    node = None
    try:
        node = RecordNode()
        while rclpy.ok() and not node.should_stop:
            rclpy.spin_once(node,timeout_sec=.1)
    except (KeyboardInterrupt,ExternalShutdownException): pass
    finally:
        if node:
            signal.signal(signal.SIGINT,signal.SIG_IGN)
            node.close(); node.destroy_node()
        if rclpy.ok(): rclpy.shutdown()


if __name__ == '__main__': main()
