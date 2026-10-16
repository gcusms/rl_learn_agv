"""Real Nav2 AMCL backend: consumes scans/odometry, never simulator ground truth."""
import math
import os
from pathlib import Path
import subprocess
import time
import numpy as np
import yaml
import rclpy
from rclpy.context import Context
from rclpy.executors import SingleThreadedExecutor
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
from geometry_msgs.msg import PoseWithCovarianceStamped, TransformStamped
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from tf2_msgs.msg import TFMessage
from tf2_ros import TransformBroadcaster, StaticTransformBroadcaster
from lifecycle_msgs.srv import ChangeState
from std_srvs.srv import Empty
from .geometry import ROOT, wrap


class RosAmclLocalizer:
    def __init__(self, static_map, lidar, global_tf=False):
        self.lidar=lidar; self.map=static_map
        self.namespace=f'rl_amcl_{os.getpid()}_{id(self)%100000}'
        self.context=Context(); rclpy.init(context=self.context)
        self.node=rclpy.create_node('sensor_bridge',namespace=self.namespace,context=self.context)
        self.executor=SingleThreadedExecutor(context=self.context); self.executor.add_node(self.node)
        self.pose=None; self.std=np.ones(3); self.correction=None; self.latest_pose=None
        self.correction_time=0.; self.pose_count=0; self.processes=[]; self.logs=[]
        self.log_dir=ROOT/'runs'/self.namespace; self.log_dir.mkdir(parents=True,exist_ok=True)
        self.scan_pub=self.node.create_publisher(LaserScan,'scan',10)
        self.odom_pub=self.node.create_publisher(Odometry,'odom',10)
        self.initial_pub=self.node.create_publisher(PoseWithCovarianceStamped,'initialpose',10)
        # Keep all transforms local to this environment namespace.
        self.tf_pub=self.node.create_publisher(TFMessage,'/tf' if global_tf else 'tf',50)
        qos=QoSProfile(depth=1,durability=DurabilityPolicy.TRANSIENT_LOCAL,reliability=ReliabilityPolicy.RELIABLE)
        self.static_pub=self.node.create_publisher(TFMessage,'/tf_static' if global_tf else 'tf_static',qos)
        self.node.create_subscription(TFMessage,'/tf' if global_tf else 'tf',self._tf,50)
        self.node.create_subscription(PoseWithCovarianceStamped,'amcl_pose',self._pose,10)
        self.nomotion=self.node.create_client(Empty,'request_nomotion_update')
        cfg={'/**':{'ros__parameters':{
            'use_sim_time':False,'global_frame_id':'map','odom_frame_id':'odom','base_frame_id':'base_footprint',
            'scan_topic':'scan','min_particles':300,'max_particles':1000,'max_beams':60,
            'robot_model_type':'nav2_amcl::DifferentialMotionModel','laser_model_type':'likelihood_field_prob',
            'do_beamskip':True,'beam_skip_distance':.5,'beam_skip_threshold':.3,'beam_skip_error_threshold':.9,
            'alpha1':.1,'alpha2':.1,'alpha3':.1,'alpha4':.1,'alpha5':.1,
            'update_min_d':.02,'update_min_a':.02,'resample_interval':1,
            'laser_min_range':lidar.range_min,'laser_max_range':lidar.range_max,'sigma_hit':.12,
            'z_hit':.9,'z_rand':.1,'laser_likelihood_max_dist':2.,'transform_tolerance':.2,
            'set_initial_pose':False,'tf_broadcast':True,
            'yaml_filename':str(static_map.path)}}}
        self.params=self.log_dir/'amcl.yaml'; self.params.write_text(yaml.safe_dump(cfg))
        try:
            for package,name in [('nav2_map_server','map_server'),('nav2_amcl','amcl')]:
                log=(self.log_dir/f'{name}.log').open('w'); self.logs.append(log)
                command=[f'/opt/ros/humble/lib/{package}/{name}','--ros-args','--params-file',str(self.params),
                         '-r',f'__ns:=/{self.namespace}','-r','/tf:=tf','-r','/tf_static:=tf_static']
                if global_tf:command=command[:-4]
                self.processes.append(subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT))
            self._publish_mount()
            for name in ['map_server','amcl']:
                client=self.node.create_client(ChangeState,name+'/change_state')
                deadline=time.monotonic()+15
                while not client.wait_for_service(timeout_sec=.2):
                    self._alive()
                    if time.monotonic()>deadline: raise RuntimeError(f'{name} startup timeout: {self.log_dir}')
                for transition in [1,3]:
                    req=ChangeState.Request(); req.transition.id=transition
                    future=client.call_async(req)
                    self.executor.spin_until_future_complete(future,timeout_sec=10)
                    if not future.done() or not future.result().success:
                        raise RuntimeError(f'{name} lifecycle transition failed: {self.log_dir}')
                self.node.destroy_client(client)
            deadline=time.monotonic()+5
            while self.scan_pub.get_subscription_count()==0 or self.initial_pub.get_subscription_count()==0:
                self.executor.spin_once(timeout_sec=.05)
                if time.monotonic()>deadline: raise RuntimeError('AMCL subscriptions unavailable')
        except BaseException:
            self.close(); raise

    def _alive(self):
        if any(p.poll() is not None for p in self.processes):
            raise RuntimeError(f'AMCL/map_server exited; inspect {self.log_dir}')

    def _tf(self,message):
        for tf in message.transforms:
            if tf.header.frame_id=='map' and tf.child_frame_id=='odom':
                q=tf.transform.rotation
                yaw=math.atan2(2*(q.w*q.z+q.x*q.y),1-2*(q.y*q.y+q.z*q.z))
                self.correction=np.array([tf.transform.translation.x,tf.transform.translation.y,yaw])
                self.correction_time=time.monotonic()

    def _pose(self,message):
        stamp=message.header.stamp.sec*1000000000+message.header.stamp.nanosec
        if stamp<getattr(self,'reset_stamp',0): return
        self.latest_pose=message; self.pose_count+=1
        covariance=np.array(message.pose.covariance).reshape(6,6)
        self.std=np.sqrt(np.maximum(0,np.diag(covariance)[[0,1,5]]))

    def _publish_mount(self):
        mount=self.lidar.mount
        tf=TransformStamped(); tf.header.stamp=self.node.get_clock().now().to_msg()
        tf.header.frame_id='base_footprint'; tf.child_frame_id='lidar_frame'
        tf.transform.translation.x=float(mount[0]); tf.transform.translation.y=float(mount[1]); tf.transform.translation.z=float(mount[2])
        tf.transform.rotation.z=math.sin(mount[3]/2); tf.transform.rotation.w=math.cos(mount[3]/2)
        self.static_pub.publish(TFMessage(transforms=[tf]))

    def reset(self,nominal,rng):
        # The nominal launch/start location is a prior; actual spawn is separately jittered.
        for _ in range(10): self.executor.spin_once(timeout_sec=0)
        self.pose=None; self.correction=None; self.std=np.array([.12,.12,.08])
        self.nominal=np.array(nominal,float); self.pending_initial=True
        self.latest_pose=None; self.reset_stamp=self.node.get_clock().now().nanoseconds

    def _publish(self,odom,scan):
        stamp=self.node.get_clock().now().to_msg()
        tf=TransformStamped(); tf.header.stamp=stamp; tf.header.frame_id='odom'; tf.child_frame_id='base_footprint'
        tf.transform.translation.x=float(odom[0]); tf.transform.translation.y=float(odom[1])
        tf.transform.rotation.z=math.sin(odom[2]/2); tf.transform.rotation.w=math.cos(odom[2]/2)
        self.tf_pub.publish(TFMessage(transforms=[tf]))
        msg=Odometry(); msg.header.stamp=stamp; msg.header.frame_id='odom'; msg.child_frame_id='base_footprint'
        msg.pose.pose.position.x=float(odom[0]); msg.pose.pose.position.y=float(odom[1]); msg.pose.pose.orientation=tf.transform.rotation
        velocity=getattr(self,'odometry_velocity',np.zeros(2))
        msg.twist.twist.linear.x=float(velocity[0]);msg.twist.twist.angular.z=float(velocity[1])
        self.odom_pub.publish(msg)
        if self.pending_initial:
            initial=PoseWithCovarianceStamped(); initial.header.stamp=stamp; initial.header.frame_id='map'
            initial.pose.pose.position.x=float(self.nominal[0]); initial.pose.pose.position.y=float(self.nominal[1])
            initial.pose.pose.orientation.z=math.sin(self.nominal[2]/2); initial.pose.pose.orientation.w=math.cos(self.nominal[2]/2)
            initial.pose.covariance[0]=.12**2; initial.pose.covariance[7]=.12**2; initial.pose.covariance[35]=.08**2
            self.initial_pub.publish(initial); self.pending_initial=False
        laser=LaserScan(); laser.header.stamp=stamp; laser.header.frame_id='lidar_frame'
        laser.angle_min=float(self.lidar.angle_min); laser.angle_max=float(self.lidar.angle_max)
        laser.angle_increment=float(self.lidar.angles[1]-self.lidar.angles[0])
        laser.range_min=self.lidar.range_min; laser.range_max=self.lidar.range_max; laser.scan_time=.2
        laser.ranges=np.asarray(scan,dtype=float).tolist(); self.scan_pub.publish(laser)

    def update(self,odom,scan):
        self._alive(); initializing=self.correction is None
        # Force a measurement update even for waiting actions; dynamic beams are skipped by AMCL.
        if self.nomotion.service_is_ready():
            future=self.nomotion.call_async(Empty.Request())
            self.executor.spin_until_future_complete(future,timeout_sec=.1)
        self._publish(odom,scan)
        deadline=time.monotonic()+(5. if initializing else .05)
        republish=time.monotonic()+.05
        retry_initial=time.monotonic()+.25
        while time.monotonic()<deadline:
            self.executor.spin_once(timeout_sec=.005)
            if initializing and self.correction is not None and self.latest_pose is not None: break
            if initializing and time.monotonic()>republish:
                if time.monotonic()>retry_initial:
                    self.pending_initial=True; retry_initial=float('inf')
                self._publish(odom,scan); republish=time.monotonic()+.05
        if self.correction is None or (initializing and self.latest_pose is None):
            raise RuntimeError(f'No AMCL laser localization received; inspect {self.log_dir}')
        if time.monotonic()-self.correction_time>2.:
            raise RuntimeError('AMCL transform stale; stop episode rather than use ground truth')
        x,y,yaw=self.correction
        self.pose=np.array([x+np.cos(yaw)*odom[0]-np.sin(yaw)*odom[1],
                            y+np.sin(yaw)*odom[0]+np.cos(yaw)*odom[1],wrap(yaw+odom[2])])
        return self.pose.copy()

    def close(self):
        for p in getattr(self,'processes',[]):
            if p.poll() is None: p.terminate()
        for p in getattr(self,'processes',[]):
            try: p.wait(timeout=3)
            except subprocess.TimeoutExpired: p.kill(); p.wait(timeout=3)
        for log in getattr(self,'logs',[]): log.close()
        if hasattr(self,'executor'):
            self.executor.shutdown(timeout_sec=1)
            self.node.destroy_node()
        if hasattr(self,'context') and self.context.ok(): self.context.shutdown()
