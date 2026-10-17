"""Recovery model inference on ROS sensor streams; shadow mode by default."""
import concurrent.futures
import multiprocessing
import json
import math
import sys
import tempfile
import time
import uuid
import hashlib
from pathlib import Path

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data, QoSProfile, DurabilityPolicy
from geometry_msgs.msg import Twist, PoseStamped, PoseWithCovarianceStamped, TransformStamped, PolygonStamped, Point32
from nav_msgs.msg import Odometry, Path as RosPath
from sensor_msgs.msg import LaserScan, Imu
from std_msgs.msg import String
from std_srvs.srv import Trigger
from tf2_ros import Buffer, TransformListener, TransformBroadcaster

from .model_bundle import inspect_bundle
from .model_adapter import prepare_map, resolve_map, limit_command
from .inference_worker import initialize_worker, worker_ready, infer_frame
from .vendor.mecanum_kinematics import MecanumKinematics


def yaw(q):
    values=(q.x,q.y,q.z,q.w)
    if not all(math.isfinite(v) for v in values) or not math.isclose(sum(v*v for v in values),1.,abs_tol=.02):
        raise ValueError('Invalid orientation quaternion')
    return math.atan2(2*(q.w*q.z+q.x*q.y),1-2*(q.y*q.y+q.z*q.z))


def stamp_seconds(stamp):
    return stamp.sec+stamp.nanosec*1e-9


class ModelNode(Node):
    def __init__(self):
        super().__init__('rl_model')
        from ament_index_python.packages import get_package_share_directory
        defaults=dict(model_config=str(Path(get_package_share_directory('rl_navigation'))/'config/model.json'),
                      model_source='/home/ubuntu/rl_learn_sms/rl_learn_agv',
                      map='/home/ubuntu/ros2_ws/src/slam/maps/map_01.yaml',mode='shadow',
                      auto_initial_pose=True,initial_x=0.,initial_y=0.,initial_yaw=0.,
                      max_rps=1.0,max_missing_scan_fraction=.08,wheel_diameter=.1,wheelbase=.2,track_width=.22)
        self.options={k:self.declare_parameter(k,v).value for k,v in defaults.items()}
        if self.options['mode'] not in ('shadow','execute'):
            raise ValueError('mode must be shadow or execute')
        missing_limit=self.options['max_missing_scan_fraction']
        if not math.isfinite(missing_limit) or not 0<=missing_limit<1:
            raise ValueError('max_missing_scan_fraction must be finite and in [0, 1)')
        if not all(math.isfinite(self.options[k]) for k in ('initial_x','initial_y','initial_yaw')):
            raise ValueError('Initial pose must be finite')
        # Bundle contracts import the explicitly selected model source too.
        sys.dont_write_bytecode=True
        sys.path[:0]=[self.options['model_source'],str(Path(self.options['model_source'])/'training')]
        config=inspect_bundle(self.options['model_config'])
        if (config['sensor_core']=='odom-goal-pose' and self.options['mode']=='execute'
            and not config['deployment_approved']):
            raise ValueError('Goal-pose candidate is not deployment-approved; use shadow validation')
        self.bundle=config['bundle'];self.route_mode=config['route_mode']
        self.model_identity={k:config[k] for k in ('branch','branch_commit','sensor_core','actor_sha256','required_runtime')}
        self.model_identity['bundle']=self.bundle
        self.geometry=MecanumKinematics(*[self.options[k] for k in ('wheel_diameter','wheelbase','track_width')])
        limit_command((0,0),self.geometry,self.options['max_rps'])
        # Read-only imports from the user-selected training source, never train.
        sys.dont_write_bytecode=True
        sys.path.insert(0,str(Path(self.options['model_source'])/'training'))
        sys.path.insert(0,self.options['model_source'])
        from neural_navigation.sensors import LidarSpec, OdometryBuffer
        hardware=json.loads((Path(self.bundle)/'hardware.json').read_text())
        self.lidar=LidarSpec(hardware['lidar'])
        self.buffer=OdometryBuffer(400)
        self.temp_map=tempfile.TemporaryDirectory(prefix='rl_navigation_map_')
        self.map_path=prepare_map(self.options['map'],self.temp_map.name)
        self.run_id=uuid.uuid4().hex
        self.inference_sequence=0
        self.loaded_files={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in
            (resolve_map(self.options['map']),self.map_path,Path(self.options['model_config']),Path(self.bundle)/'manifest.json')}
        # Snapshot the exact images loaded at startup, including padded maps.
        import yaml
        for map_path in (resolve_map(self.options['map']),self.map_path):
            image_path=(map_path.parent/yaml.safe_load(map_path.read_text())['image']).resolve()
            self.loaded_files[str(image_path)]=hashlib.sha256(image_path.read_bytes()).hexdigest()
        worker_settings=dict(model_source=self.options['model_source'],required_runtime=config['required_runtime'],
            bundle=self.bundle,map_path=str(self.map_path),route_mode=self.route_mode,
            geometry=[self.options[k] for k in ('wheel_diameter','wheelbase','track_width')],
            max_rps=self.options['max_rps'])
        # Spawn (never fork a live ROS context). Only this parent publishes
        # commands; worker results still pass the acquisition-time deadline.
        self.pool=concurrent.futures.ProcessPoolExecutor(max_workers=1,
            mp_context=multiprocessing.get_context('spawn'),
            initializer=initialize_worker,initargs=(worker_settings,))
        self.warmup=self.pool.submit(worker_ready)
        self.future=None;self.generation=0;self.localization_generation=0;self.last_result=None;self.last_frame=None
        self.initial=None;self.goal=None;self.active=False;self.fault=None;self.terminal=None
        self.published_command=(0.,0.);self.published_reason='waiting_inference'
        self.map_alignment=None
        if self.options['auto_initial_pose']:
            self.initial=tuple(self.options[k] for k in ('initial_x','initial_y','initial_yaw'))
        self.scan=None;self.scan_received=self.odom_received=self.raw_received=self.imu_received=0.
        self.scan_cache=None;self.scan_quality={};self.visualized_result=None
        self.origin=self.get_clock().now().nanoseconds/1e9
        self.tf_buffer=Buffer();self.tf_listener=TransformListener(self.tf_buffer,self)
        self.tf_pub=TransformBroadcaster(self)
        self.candidate_pub=self.create_publisher(Twist,'/rl_navigation/cmd_vel_candidate',10)
        self.command_pub=self.create_publisher(Twist,'/rl_navigation/cmd_vel',10)
        self.status_pub=self.create_publisher(String,'/rl_navigation/model_status',10)
        self.decision_pub=self.create_publisher(String,'/rl_navigation/decision',10)
        self.metadata_pub=self.create_publisher(String,'/rl_navigation/session_metadata',
            QoSProfile(depth=1,durability=DurabilityPolicy.TRANSIENT_LOCAL))
        self.pose_pub=self.create_publisher(PoseStamped,'/rl_navigation/model_pose',10)
        self.path_pub=self.create_publisher(RosPath,'/rl_navigation/path',10)
        self.footprint_pub=self.create_publisher(PolygonStamped,'/rl_navigation/footprint',10)
        self.create_subscription(LaserScan,'/scan',self.on_scan,qos_profile_sensor_data)
        self.create_subscription(Odometry,'/odom',self.on_odom,10)
        self.create_subscription(Odometry,'/rl_navigation/odom_raw',self.on_raw,10)
        self.create_subscription(Imu,'/imu',self.on_imu,10)
        self.create_subscription(PoseWithCovarianceStamped,'/initialpose',self.on_initial,10)
        self.create_subscription(PoseStamped,'/goal_pose',self.on_goal,10)
        self.create_service(Trigger,'/rl_navigation/cancel',self.cancel)
        self.create_timer(.02,self.tick)
        self.publish_metadata()
        self.get_logger().info('Recovery model interfaces loaded; inference=spawn_process; mode='+self.options['mode']+
                               '; initial_pose='+str(self.initial)+'; model='+json.dumps(self.model_identity)+
                               '; waiting for sensors and goal')

    def status(self,state,**details):
        self.status_pub.publish(String(data=json.dumps(dict(state=state,mode=self.options['mode'],
            run_id=self.run_id,mission_id=self.generation,ros_time_ns=self.get_clock().now().nanoseconds,
            model=self.model_identity,**details),allow_nan=False)))

    def publish_metadata(self):
        self.metadata_pub.publish(String(data=json.dumps(dict(schema='rl_navigation_session_v1',
            run_id=self.run_id,mission_id=self.generation,ros_time_ns=self.get_clock().now().nanoseconds,
            model=self.model_identity,options=self.options,initial=self.initial,goal=self.goal,
            time_origin=self.origin,prepared_map=str(self.map_path),loaded_files=self.loaded_files),allow_nan=False)))

    def record_decision(self,completed,eligibility):
        generation,stamp,result,command,scale,path,odom=completed
        observation=result.pop('policy_observation',None)
        inputs=result.pop('recording_input',None)
        if not self.decision_pub.get_subscription_count():return
        payload=dict(schema='rl_navigation_decision_v1',run_id=self.run_id,
            mission_id=generation,inference_id=self.future_sequence,
            scan_stamp_ns=getattr(self,'future_scan_stamp_ns',int(round((self.origin+stamp)*1e9))),
            decision_stamp_ns=self.get_clock().now().nanoseconds,
            initial=self.future_initial,goal=self.future_goal,
            model=self.model_identity,mode=self.options['mode'],eligibility=eligibility,
            observation_available=observation is not None,
            observation_unavailable_reason=None if observation is not None else 'subscriber_joined_after_inference_submission',
            observation=observation,inputs=inputs,scan_quality=self.future_scan_quality,
            limited_command=list(command),limit_scale=scale,path=path.tolist(),result=result,
            command_semantics='candidate_after_limits; actual publications are recorded separately')
        self.decision_pub.publish(String(data=json.dumps(payload,allow_nan=False)))

    def stop(self):
        self.published_command=(0.,0.);self.published_reason='external_stop'
        self.candidate_pub.publish(Twist())
        if self.options['mode']=='execute':self.command_pub.publish(Twist())

    def new_mission(self):
        self.generation+=1;self.last_result=None;self.last_frame=None;self.fault=None;self.terminal=None
        self.active=self.initial is not None and self.goal is not None
        self.publish_metadata()
        self.stop()

    def on_initial(self,msg):
        try:
            if msg.header.frame_id!='map':raise ValueError('Initial pose must use map frame')
            p=msg.pose.pose
            pose=(p.position.x,p.position.y,yaw(p.orientation))
            if not all(math.isfinite(v) for v in pose):raise ValueError('Nonfinite pose')
            self.initial=pose
            self.localization_generation+=1
            self.map_alignment=None
            if self.buffer.rows:
                self.set_map_alignment(pose,self.buffer.rows[-1][1])
            self.new_mission()
        except ValueError as exc:self.status('INVALID_INITIAL_POSE',reason=str(exc))

    def on_goal(self,msg):
        try:
            if msg.header.frame_id!='map':raise ValueError('Goal must use map frame')
            goal=(msg.pose.position.x,msg.pose.position.y)
            if self.model_identity['sensor_core'] in ('odom-goal-pose','learned-terminal-goal-pose'):
                goal=(*goal,yaw(msg.pose.orientation))
            if not all(math.isfinite(v) for v in goal):raise ValueError('Nonfinite goal')
            # A new goal must not reset localization to the startup pose after
            # the robot has moved. Re-anchor at the current map/odom alignment.
            if self.map_alignment is not None and self.buffer.rows:
                ox,oy,heading=self.buffer.rows[-1][1]
                x,y,angle=self.map_alignment;c,s=math.cos(angle),math.sin(angle)
                self.initial=(x+c*ox-s*oy,y+s*ox+c*oy,angle+heading)
            self.goal=goal;self.new_mission()
        except ValueError as exc:self.status('INVALID_GOAL',reason=str(exc))

    def cancel(self,request,response):
        self.active=False;self.goal=None;self.new_mission();self.terminal="CANCELLED"
        response.success=True;response.message='Cancelled; zero command sent'
        return response

    def fresh_stamp(self,stamp,age=.3):
        now=self.get_clock().now().nanoseconds/1e9
        return -.05 <= now-stamp_seconds(stamp) <= age

    def on_raw(self,msg):
        if self.fresh_stamp(msg.header.stamp):self.raw_received=time.monotonic()

    def on_imu(self,msg):
        if self.fresh_stamp(msg.header.stamp):self.imu_received=time.monotonic()

    def on_odom(self,msg):
        try:
            if msg.header.frame_id!='odom' or msg.child_frame_id!='base_footprint' or not self.fresh_stamp(msg.header.stamp):return
            p=msg.pose.pose
            self.buffer.append(stamp_seconds(msg.header.stamp)-self.origin,
                [p.position.x,p.position.y,yaw(p.orientation)],
                [msg.twist.twist.linear.x,msg.twist.twist.angular.z])
            self.odom_received=time.monotonic()
            if self.initial is not None and self.map_alignment is None:
                self.set_map_alignment(self.initial,[p.position.x,p.position.y,yaw(p.orientation)])
        except ValueError as exc:self.status('INVALID_ODOM',reason=str(exc))

    def on_scan(self,msg):
        if self.fresh_stamp(msg.header.stamp):
            self.scan=msg;self.scan_received=time.monotonic()
            self.scan_cache=None

    def inputs(self):
        now=time.monotonic()
        if not self.active:return None,self.terminal or 'WAITING_INITIAL_POSE_OR_GOAL'
        if self.fault:return None,'FAULT'
        if self.scan is None or any(now-t>.3 for t in (self.scan_received,self.odom_received,self.raw_received,self.imu_received)):
            return None,'WAITING_FRESH_SENSORS'
        scan=self.scan
        if not self.fresh_stamp(scan.header.stamp):return None,'STALE_SCAN'
        tf=self.tf_buffer.lookup_transform('base_footprint',scan.header.frame_id,rclpy.time.Time())
        mounting_yaw=yaw(tf.transform.rotation)
        mount=tf.transform.translation
        if abs(mount.x-self.lidar.mount[0])>.03 or abs(mount.y-self.lidar.mount[1])>.03 or abs(mounting_yaw-self.lidar.mount[3])>.05:
            return None,'LIDAR_MOUNT_CONTRACT_MISMATCH'
        stamp=stamp_seconds(scan.header.stamp)-self.origin
        aligned=self.buffer.sample(stamp,max_gap=.1,max_extrapolation=.05)
        if aligned is None:return None,'WAITING_ALIGNED_ODOMETRY'
        if self.scan_cache is None:
            self.scan_cache=self.lidar.resample(scan.ranges,scan.angle_min,scan.angle_increment,scan.range_min,scan.range_max)
        resampled,valid=self.scan_cache
        missing=np.isnan(resampled)
        self.scan_quality=dict(angular_coverage_valid=bool(valid),missing_beams=int(missing.sum()),
            total_beams=len(resampled),missing_fraction=float(missing.mean()),
            missing_fraction_limit=self.options['max_missing_scan_fraction'],
            missing_angles_deg=np.rad2deg(self.lidar.angles[missing]).tolist())
        # Unknown samples remain NaN; reject substantial missing coverage.
        if not valid or missing.mean()>self.options['max_missing_scan_fraction']:return None,'INVALID_SCAN_COVERAGE'
        return (resampled,*aligned,stamp,valid),'READY'

    def tick(self):
        try:
            self.publish_map_tf()
            frame,state=self.inputs()
            if self.future and self.future.done():
                pending=self.future;self.future=None
                if self.future_generation==self.generation:
                    completed=pending.result()
                    completed[2]['adapter_round_trip_ms']=(time.monotonic()-self.submitted_at)*1000
                    if hasattr(self,'decision_pub'):
                        age=self.get_clock().now().nanoseconds/1e9-self.origin-completed[1]
                        try:
                            self.record_decision(completed,'inputs_unavailable' if frame is None else
                                'eligible' if 0<=age<=.25 else 'expired')
                        except Exception as exc:
                            # Optional recording diagnostics must not change control decisions.
                            completed[2].pop('policy_observation',None);completed[2].pop('recording_input',None)
                            self.get_logger().warning('Decision recording unavailable: '+str(exc),throttle_duration_sec=5.)
                    self.last_result=completed
                else:
                    try:
                        completed=pending.result()
                        if hasattr(self,'decision_pub'):self.record_decision(completed,'mission_changed')
                    except Exception:pass
            if frame is None:
                self.last_result=None;self.stop();self.status(state,reason=self.fault,
                    scan_quality=getattr(self,'scan_quality',{}) if state=='INVALID_SCAN_COVERAGE' else None)
                return
            # No queued frames and no artificial 5 Hz throttle: consume the
            # newest available scan as soon as the previous inference finishes.
            if self.future is None and (self.last_frame is None or frame[3]>self.last_frame):
                self.last_frame=frame[3]
                self.future_generation=self.generation
                self.submitted_at=time.monotonic()
                if hasattr(self,'decision_pub'):
                    self.inference_sequence+=1;self.future_sequence=self.inference_sequence
                    self.future_initial=self.initial;self.future_goal=self.goal
                    self.future_scan_quality=dict(self.scan_quality)
                    self.future_scan_stamp_ns=self.scan.header.stamp.sec*1_000_000_000+self.scan.header.stamp.nanosec
                self.future=self.pool.submit(infer_frame,frame,self.generation,self.initial,self.goal,
                    self.get_clock().now().nanoseconds/1e9-self.origin,
                    bool(getattr(self,'decision_pub',None) and self.decision_pub.get_subscription_count()),
                    self.published_command,self.published_reason,self.localization_generation)
            now=self.get_clock().now().nanoseconds/1e9-self.origin
            if self.last_result is None or not 0<=now-self.last_result[1]<=.25:
                self.stop();self.status('WAITING_INFERENCE',
                    result_age_s=None if self.last_result is None else now-self.last_result[1],
                    inference_ms=None if self.last_result is None else self.last_result[2].get('adapter_inference_ms'));return
            generation,stamp,result,command,scale,path,odom=self.last_result
            if getattr(self,'visualized_result',None) is not self.last_result:
                self.publish_visuals(result,path,odom,stamp)
                self.visualized_result=self.last_result
            msg=Twist();msg.linear.x,msg.angular.z=command
            self.candidate_pub.publish(msg)
            if self.options['mode']=='execute':
                self.command_pub.publish(msg)
                self.published_command=tuple(command);self.published_reason=result['stop_reason'] or 'none'
            reached=result['stop_reason']=='estimated_goal_reached'
            self.status('ESTIMATED_GOAL_REACHED' if reached else 'RUNNING',
                result=result,limit_scale=scale,limited_command=list(command),result_age_s=now-stamp)
            if reached:self.active=False;self.terminal="ESTIMATED_GOAL_REACHED";self.stop()
        except Exception as exc:
            self.stop()
            # TF not yet available is recoverable; model failures require new mission.
            from tf2_ros import TransformException
            if isinstance(exc,TransformException):self.status('WAITING_TF',reason=str(exc))
            else:
                self.fault=str(exc)
                if self.future and self.future.done():self.future=None
                self.status('FAULT',reason=self.fault)

    def publish_visuals(self,result,path,odom,stamp):
        estimate=result['diagnostics']['estimated_pose']
        header_stamp=rclpy.time.Time(nanoseconds=int((stamp+self.origin)*1e9)).to_msg()
        pose=PoseStamped();pose.header.frame_id='map';pose.header.stamp=header_stamp
        pose.pose.position.x,pose.pose.position.y=map(float,estimate[:2])
        pose.pose.orientation.z=math.sin(estimate[2]/2);pose.pose.orientation.w=math.cos(estimate[2]/2)
        self.pose_pub.publish(pose)
        # Show the trained rectangular body and fixed 4 cm safety margin.
        # The additional speed-dependent swept braking area is not this polygon.
        polygon=PolygonStamped();polygon.header=pose.header
        c,s=math.cos(estimate[2]),math.sin(estimate[2])
        hx,hy=.393924904/2+.04,.329460022/2+.04
        for x,y in ((hx,hy),(-hx,hy),(-hx,-hy),(hx,-hy)):
            point=Point32();point.x=float(estimate[0]+c*x-s*y)
            point.y=float(estimate[1]+s*x+c*y);polygon.polygon.points.append(point)
        self.footprint_pub.publish(polygon)
        route=RosPath();route.header=pose.header
        for xy in path:
            point=PoseStamped();point.header=route.header
            point.pose.position.x,point.pose.position.y=map(float,xy)
            point.pose.orientation.w=1.;route.poses.append(point)
        self.path_pub.publish(route)
        self.set_map_alignment(estimate,odom)
        self.publish_map_tf()

    def set_map_alignment(self,estimate,odom):
        """Initial pose or learned pose anchors continuous odometry to the map."""
        angle=float(estimate[2]-odom[2]);c,s=math.cos(angle),math.sin(angle)
        self.map_alignment=(float(estimate[0]-c*odom[0]+s*odom[1]),
                            float(estimate[1]-s*odom[0]-c*odom[1]),angle)

    def publish_map_tf(self):
        # No invented map origin before the user supplies an initial pose.
        # Hold the latest correction between updates; EKF carries odom motion.
        if self.map_alignment is None:
            return
        x,y,angle=self.map_alignment
        tf=TransformStamped()
        tf.header.stamp=self.get_clock().now().to_msg()
        tf.header.frame_id='map';tf.child_frame_id='odom'
        tf.transform.translation.x=x;tf.transform.translation.y=y
        tf.transform.rotation.z=math.sin(angle/2);tf.transform.rotation.w=math.cos(angle/2)
        self.tf_pub.sendTransform(tf)

    def close(self):
        try:
            if self.context.ok():self.stop()
        finally:
            try:self.pool.shutdown(wait=True,cancel_futures=True)
            finally:self.temp_map.cleanup()


def main(args=None):
    from rclpy.signals import SignalHandlerOptions
    from rclpy.executors import ExternalShutdownException
    # Keep publishers alive until close() has sent the final zero command.
    rclpy.init(args=args,signal_handler_options=SignalHandlerOptions.NO);node=None
    try:
        node=ModelNode();rclpy.spin(node)
    except (KeyboardInterrupt,ExternalShutdownException):pass
    finally:
        if node:
            node.close();node.destroy_node()
        if rclpy.ok():rclpy.shutdown()
