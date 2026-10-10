"""ROS inference adapter. Output /rl/cmd_vel is for an explicit velocity arbiter.
Localization is provided by the robot's existing AMCL/TF stack.
"""
import argparse
from collections import deque
import json
import math
from pathlib import Path
import time
import numpy as np
import torch
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from geometry_msgs.msg import Twist, PoseWithCovarianceStamped
from nav_msgs.msg import Odometry, Path as RosPath
from sensor_msgs.msg import LaserScan
from tf2_ros import Buffer, TransformListener, TransformException
from stable_baselines3 import PPO
from navrl.geometry import ROOT,LidarProfile,wrap,integrate
from navrl.map import StaticMap,path_features


class PolicyNode(Node):
    def __init__(self,args):
        super().__init__('rl_navigation_policy')
        self.args=args; self.lidar=LidarProfile(); self.map=StaticMap(args.map,radius=.3)
        config=json.loads((args.policy.parent/'config.json').read_text())
        if config['lidar']!=self.lidar.signature: raise ValueError('Policy and real LiDAR profile mismatch')
        self.policy=PPO.load(args.policy,device='cpu'); self.history=deque(maxlen=3)
        self.scan=None; self.latest_binned=None; self.scan_time=0.; self.velocity=np.zeros(2); self.odom_time=0.; self.path=None
        self.uncertainty=np.ones(3); self.pose_time=0.; self.last_action=np.zeros(2)
        self.buffer=Buffer(); self.listener=TransformListener(self.buffer,self)
        self.publisher=self.create_publisher(Twist,args.output_topic,10)
        self.create_subscription(LaserScan,args.scan_topic,self.on_scan,qos_profile_sensor_data)
        self.create_subscription(Odometry,args.odom_topic,self.on_odom,10)
        self.create_subscription(RosPath,args.path_topic,self.on_path,10)
        self.create_subscription(PoseWithCovarianceStamped,args.pose_topic,self.on_pose,10)
        self.create_timer(.2,self.tick)
        self.get_logger().info(f'Localized policy output: {args.output_topic}; scanner {math.degrees(self.lidar.angle_max-self.lidar.angle_min):.2f} degrees')

    def on_scan(self,message):
        if message.header.frame_id!=self.args.lidar_frame: return
        if len(message.ranges)<2 or message.angle_increment==0: return
        angle=wrap(message.angle_min+np.arange(len(message.ranges))*message.angle_increment)
        ranges=np.asarray(message.ranges,float)
        # Circular matching handles the real retained sector crossing 0 degrees.
        differences=np.abs(wrap(angle[:,None]-self.lidar.angles[None,:]))
        nearest=np.argmin(differences,axis=0)
        sampled=ranges[nearest].copy()
        unsupported=differences[nearest,np.arange(self.lidar.beams)]>max(abs(message.angle_increment)*1.5,.025)
        sampled[unsupported|(sampled<max(message.range_min,self.lidar.range_min))]=np.inf
        if np.mean(unsupported | np.isnan(sampled))>.1:
            return
        self.scan=sampled
        cleaned=np.nan_to_num(sampled,nan=self.lidar.range_max,posinf=self.lidar.range_max,neginf=self.lidar.range_min)
        binned=np.array([np.min(part) for part in np.array_split(cleaned,72)])/self.lidar.range_max
        if not self.history:
            self.history.extend([binned.copy() for _ in range(3)])
        self.latest_binned=binned
        self.scan_time=time.monotonic()

    def on_odom(self,message):
        self.velocity=np.array([message.twist.twist.linear.x,message.twist.twist.angular.z])
        self.odom_time=time.monotonic()

    def on_path(self,message):
        if message.header.frame_id!=self.args.map_frame or len(message.poses)<2:
            self.path=None; return
        raw=np.array([[p.pose.position.x,p.pose.position.y] for p in message.poses])
        if not np.isfinite(raw).all(): self.path=None; return
        sampled=[]
        for a,b in zip(raw[:-1],raw[1:]):
            sampled.extend(np.linspace(a,b,max(2,int(np.linalg.norm(b-a)/.1)+1))[:-1])
        sampled.append(raw[-1]); self.path=np.array(sampled)

    def on_pose(self,message):
        if message.header.frame_id!=self.args.map_frame: return
        self.uncertainty=np.sqrt(np.maximum(0,np.diag(np.asarray(message.pose.covariance).reshape(6,6))[[0,1,5]]))
        self.pose_time=time.monotonic()

    def stop(self):
        self.publisher.publish(Twist()); self.last_action=np.zeros(2)

    def tick(self):
        now=time.monotonic()
        if self.scan is None or self.path is None or now-self.scan_time>.6 or now-self.odom_time>.6 or now-self.pose_time>5:
            self.stop(); return
        if np.max(self.uncertainty[:2])>.3 or self.uncertainty[2]>.35:
            self.stop(); return
        try:
            tf=self.buffer.lookup_transform(self.args.map_frame,self.args.base_frame,rclpy.time.Time())
        except TransformException:
            self.stop(); return
        stamp=tf.header.stamp.sec*1e9+tf.header.stamp.nanosec
        if abs(self.get_clock().now().nanoseconds-stamp)>1e9:
            self.stop(); return
        self.history.append(self.latest_binned.copy())
        q=tf.transform.rotation
        yaw=math.atan2(2*(q.w*q.z+q.x*q.y),1-2*(q.y*q.y+q.z*q.z))
        pose=np.array([tf.transform.translation.x,tf.transform.translation.y,yaw]); goal=self.path[-1]
        if np.linalg.norm(goal-pose[:2])<.25: self.stop(); return
        target,_,deviation=path_features(self.path,pose[:2],self.map)
        rotation=np.array([[np.cos(yaw),np.sin(yaw)],[-np.sin(yaw),np.cos(yaw)]])
        features=np.r_[rotation@(target-pose[:2])/2.,rotation@(goal-pose[:2])/12.,
                       self.velocity/[.2,.5],self.last_action,self.uncertainty/[.5,.5,.5],deviation/2.]
        obs=np.clip(np.r_[np.concatenate(self.history),features],-1,1).astype(np.float32)
        action=np.clip(self.policy.predict(obs,deterministic=True)[0],-1,1)
        v=max(0,float(action[0]))*.2; w=float(action[1])*.5
        # Check swept centres against the static footprint margin and measured obstacles.
        good=np.isfinite(self.scan)&(self.scan>=self.lidar.range_min)&(self.scan<self.lidar.range_max)
        m=self.lidar.mount; angles=self.lidar.angles[good]+m[3]
        points=np.column_stack([self.scan[good]*np.cos(angles)+m[0],self.scan[good]*np.sin(angles)+m[1]])
        for t in np.linspace(0,.6,7):
            future=integrate(pose,v,w,t)
            local=integrate(np.zeros(3),v,w,t)
            if self.map.lookup(future[:2])<.32 or (len(points) and np.min(np.linalg.norm(points-local[:2],axis=1))<.30):
                v=0.; w=0.; break
        command=Twist(); command.linear.x=v; command.angular.z=w
        self.publisher.publish(command); self.last_action=action


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--policy',type=Path,required=True)
    parser.add_argument('--map',type=Path,default=ROOT/'assets/decision_map.yaml')
    parser.add_argument('--scan-topic',default='/scan'); parser.add_argument('--odom-topic',default='/odom')
    parser.add_argument('--path-topic',default='/plan'); parser.add_argument('--pose-topic',default='/amcl_pose')
    parser.add_argument('--output-topic',default='/rl/cmd_vel')
    parser.add_argument('--map-frame',default='map'); parser.add_argument('--base-frame',default='base_footprint')
    parser.add_argument('--lidar-frame',default='lidar_frame')
    args,ros_args=parser.parse_known_args(); torch.set_num_threads(4); rclpy.init(args=ros_args)
    node=None
    try:
        node=PolicyNode(args); rclpy.spin(node)
    finally:
        if node is not None: node.stop(); node.destroy_node()
        rclpy.shutdown()


if __name__=='__main__': main()
