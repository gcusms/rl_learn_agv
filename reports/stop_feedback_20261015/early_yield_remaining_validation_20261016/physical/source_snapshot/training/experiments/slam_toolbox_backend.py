"""Isolated, simulation-only SLAM Toolbox bridge; publishes no motor commands.

Callers supply measured scans and odometry, never the simulator map or pose.
The ROS domain is private to this experiment. The navigation actor stays in
the portable runtime and receives the observed map and estimated pose.
"""
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import time
import numpy as np
import yaml


class SlamBackend:
    def __init__(self,lidar,output,domain=197,rf2o_executable=None,laser_increments=False,distance_prior_variance=None):
        if rf2o_executable is not None and laser_increments:raise ValueError('Select one incremental odometry backend')
        if distance_prior_variance is not None and (not np.isfinite(distance_prior_variance) or distance_prior_variance<=0):
            raise ValueError('Positive finite SLAM distance prior variance required')
        self.distance_prior_variance=distance_prior_variance
        self.increment_matcher=None
        if laser_increments:
            from neural_navigation.laser_increment_odometry import LaserIncrementOdometry
            self.increment_matcher=LaserIncrementOdometry(lidar)
        import rclpy
        from rclpy.context import Context
        from rclpy.node import Node
        from rclpy.executors import SingleThreadedExecutor
        from rclpy.parameter import Parameter
        from rclpy.qos import qos_profile_sensor_data,QoSProfile,DurabilityPolicy
        from sensor_msgs.msg import LaserScan
        from nav_msgs.msg import OccupancyGrid,Odometry
        from geometry_msgs.msg import PoseWithCovarianceStamped,TransformStamped
        from rosgraph_msgs.msg import Clock
        from tf2_ros import TransformBroadcaster,StaticTransformBroadcaster,Buffer,TransformListener
        self.rclpy=rclpy;self.lidar=lidar;self.output=Path(output);self.output.mkdir(parents=True,exist_ok=True)
        self.context=Context();rclpy.init(context=self.context,domain_id=domain)
        self.node=Node('rl_slam_sensor_bridge',context=self.context,
            parameter_overrides=[Parameter('use_sim_time',value=True)])
        self.executor=SingleThreadedExecutor(context=self.context)
        self.scan_pub=self.node.create_publisher(LaserScan,'/scan',qos_profile_sensor_data)
        self.clock_pub=self.node.create_publisher(Clock,'/clock',10)
        self.tf_pub=TransformBroadcaster(self.node);self.static_pub=StaticTransformBroadcaster(self.node)
        self.tf_buffer=Buffer(node=self.node);self.tf_listener=TransformListener(self.tf_buffer,self.node)
        self.map_msg=None;self.map_version=0;self.last_pose_msg=None;self.pose_messages=0
        self.transient_returns={};self.filtered_returns=0
        self.rf2o_process=None;self.rf2o_log=None;self.rf2o_msg=None;self.rf2o_used=0;self.rf2o_fallback=0
        self.rf2o_previous=None;self.raw_previous=None;self.last_odometry=np.zeros(3)
        self.map_sub=self.node.create_subscription(OccupancyGrid,'/map',self._map,
            QoSProfile(depth=1,durability=DurabilityPolicy.TRANSIENT_LOCAL))
        self.pose_sub=self.node.create_subscription(PoseWithCovarianceStamped,'/pose',self._pose,10)
        static=TransformStamped();static.header.frame_id='base_footprint';static.child_frame_id='lidar_frame'
        static.transform.translation.x=float(lidar.mount[0]);static.transform.translation.y=float(lidar.mount[1])
        static.transform.translation.z=float(lidar.mount[2]);static.transform.rotation.z=math.sin(lidar.mount[3]/2)
        static.transform.rotation.w=math.cos(lidar.mount[3]/2);self.static_pub.sendTransform(static)
        config=yaml.safe_load(Path('/opt/ros/humble/share/slam_toolbox/config/mapper_params_online_async.yaml').read_text())
        config['slam_toolbox']['ros__parameters'].update(
            use_sim_time=True,enable_interactive_mode=False,ceres_loss_function='HuberLoss',
            # Humble's shouldProcessScan gates on translation alone. Zero
            # permits pure-turn observations; the 0.15s time gate bounds rate.
            map_update_interval=.4,minimum_time_interval=.15,minimum_travel_distance=0.,
            minimum_travel_heading=.05,min_laser_range=float(lidar.range_min),max_laser_range=float(lidar.range_max),
            transform_timeout=.1,scan_buffer_size=20,loop_match_minimum_chain_size=6)
        if distance_prior_variance is not None:
            config['slam_toolbox']['ros__parameters']['distance_variance_penalty']=float(distance_prior_variance)
        config_path=self.output/'slam_parameters.yaml';config_path.write_text(yaml.safe_dump(config))
        env=os.environ.copy();env['ROS_DOMAIN_ID']=str(domain)
        self.log=(self.output/'slam_backend.log').open('w')
        # The installed Humble Ceres plugin hard-codes up to 50 threads. Limit
        # its whole process tree to one logical CPU without editing ROS itself.
        backend_cpu=max(os.sched_getaffinity(0))
        self.process=subprocess.Popen(['taskset','-c',str(backend_cpu),'ros2','launch','slam_toolbox','online_async_launch.py',
            'slam_params_file:='+str(config_path.resolve()),'use_sim_time:=true'],env=env,
            stdout=self.log,stderr=subprocess.STDOUT,start_new_session=True)
        if rf2o_executable is not None:
            executable=Path(rf2o_executable).resolve()
            if not executable.is_file():self.close();raise ValueError('RF2O executable missing')
            self.rf2o_scan_pub=self.node.create_publisher(LaserScan,'/rf2o_scan',qos_profile_sensor_data)
            self.rf2o_sub=self.node.create_subscription(Odometry,'/odom_rf2o',self._rf2o,10)
            # RF2O assumes a scan symmetric about zero. Rotate only its virtual
            # scan frame by the actual angle midpoint; SLAM keeps real angles.
            midpoint=(lidar.angle_min+lidar.angle_max)/2
            rf_static=TransformStamped();rf_static.header.frame_id='base_footprint';rf_static.child_frame_id='rf2o_lidar_frame'
            rf_static.transform.translation=static.transform.translation
            rf_static.transform.rotation.z=math.sin((lidar.mount[3]+midpoint)/2)
            rf_static.transform.rotation.w=math.cos((lidar.mount[3]+midpoint)/2)
            self.static_pub.sendTransform([static,rf_static])
            self.rf2o_log=(self.output/'rf2o_backend.log').open('w')
            self.rf2o_process=subprocess.Popen(['taskset','-c',str(backend_cpu),str(executable),'--ros-args',
                '-p','use_sim_time:=true','-p','laser_scan_topic:=/rf2o_scan','-p','odom_topic:=/odom_rf2o',
                '-p','publish_tf:=false','-p','base_frame_id:=base_footprint','-p','odom_frame_id:=odom',
                '-p','init_pose_from_topic:=""','-p','freq:=20.0'],env=env,stdout=self.rf2o_log,
                stderr=subprocess.STDOUT,start_new_session=True)
        self.started=time.monotonic();self.alignment=None;self.last_alignment_stamp=None
        try:
            self.spin_for(.3)
            # Discovery includes backend initialization.
            limit=time.monotonic()+12.
            while (self.scan_pub.get_subscription_count()==0 or
                self.rf2o_process is not None and self.rf2o_scan_pub.get_subscription_count()==0) and time.monotonic()<limit:
                if self.process.poll() is not None:raise RuntimeError('SLAM Toolbox exited; see '+str(self.output/'slam_backend.log'))
                if self.rf2o_process is not None and self.rf2o_process.poll() is not None:raise RuntimeError('RF2O exited; see '+str(self.output/'rf2o_backend.log'))
                self.spin_for(.1)
            if self.scan_pub.get_subscription_count()==0:raise RuntimeError('SLAM scan subscription not ready')
            if self.rf2o_process is not None and self.rf2o_scan_pub.get_subscription_count()==0:raise RuntimeError('RF2O scan subscription not ready')
        except Exception:
            self.close();raise

    def _map(self,msg):self.map_msg=msg;self.map_version+=1
    def _pose(self,msg):self.last_pose_msg=msg;self.pose_messages+=1
    def _rf2o(self,msg):self.rf2o_msg=msg
    def spin_for(self,seconds):
        until=time.monotonic()+seconds
        while time.monotonic()<until:
            self.rclpy.spin_once(self.node,executor=self.executor,timeout_sec=min(.02,max(0.,until-time.monotonic())))

    def push(self,scan,odom,timestamp,wait=.06):
        from builtin_interfaces.msg import Time
        from rosgraph_msgs.msg import Clock
        from sensor_msgs.msg import LaserScan
        from geometry_msgs.msg import TransformStamped
        total_ns=int(round((100.+float(timestamp))*1e9))
        stamp=Time(sec=total_ns//1_000_000_000,nanosec=total_ns%1_000_000_000)
        clock=Clock();clock.clock=stamp;self.clock_pub.publish(clock)
        if self.rf2o_process is not None:odom=self._laser_odometry(scan,odom,timestamp,stamp)
        elif self.increment_matcher is not None:
            odom=self.increment_matcher.update(scan,odom,timestamp);self.last_odometry=odom.copy()
        else:self.last_odometry=np.asarray(odom,float).copy()
        transform=TransformStamped();transform.header.stamp=stamp
        transform.header.frame_id='odom';transform.child_frame_id='base_footprint'
        transform.transform.translation.x=float(odom[0]);transform.transform.translation.y=float(odom[1])
        transform.transform.rotation.z=math.sin(odom[2]/2);transform.transform.rotation.w=math.cos(odom[2]/2)
        self.tf_pub.sendTransform(transform)
        self.spin_for(.012)
        mapping_scan=self._mapping_scan(scan,odom,timestamp)
        message=LaserScan();message.header.stamp=stamp;message.header.frame_id='lidar_frame'
        message.angle_min=float(self.lidar.angle_min);message.angle_max=float(self.lidar.angle_max)
        message.angle_increment=float((self.lidar.angle_max-self.lidar.angle_min)/(len(scan)-1))
        message.range_min=float(self.lidar.range_min);message.range_max=float(self.lidar.range_max)
        message.scan_time=.2;message.time_increment=0.;message.ranges=np.asarray(mapping_scan,np.float32).tolist()
        self.scan_pub.publish(message);self.spin_for(wait)
        return self.estimated_pose(odom)

    def _laser_odometry(self,scan,raw_odom,timestamp,stamp):
        from sensor_msgs.msg import LaserScan
        from navrl.geometry import wrap
        if self.rf2o_process.poll() is not None:raise RuntimeError('RF2O stopped; motion experiment aborted')
        message=LaserScan();message.header.stamp=stamp;message.header.frame_id='rf2o_lidar_frame'
        half=(self.lidar.angle_max-self.lidar.angle_min)/2
        message.angle_min=float(-half);message.angle_max=float(half);message.angle_increment=float(2*half/(len(scan)-1))
        message.range_min=float(self.lidar.range_min);message.range_max=float(self.lidar.range_max)
        message.scan_time=.2;message.ranges=np.asarray(scan,np.float32).tolist();self.rf2o_scan_pub.publish(message)
        limit=time.monotonic()+.18
        while time.monotonic()<limit:
            self.spin_for(.01)
            if self.rf2o_msg is not None and abs(self.rf2o_msg.header.stamp.sec+self.rf2o_msg.header.stamp.nanosec/1e9-100.-timestamp)<.005:break
        raw_odom=np.asarray(raw_odom,float)
        def increment(current,previous):
            delta=current[:2]-previous[:2];a=previous[2];c,s=np.cos(a),np.sin(a)
            return np.r_[c*delta[0]+s*delta[1],-s*delta[0]+c*delta[1],wrap(current[2]-previous[2])]
        delta=np.zeros(3) if self.raw_previous is None else increment(raw_odom,self.raw_previous)
        self.raw_previous=raw_odom.copy();accepted=False
        if self.rf2o_msg is not None:
            p=self.rf2o_msg.pose.pose;q=p.orientation
            pose=np.array([p.position.x,p.position.y,math.atan2(2*(q.w*q.z+q.x*q.y),1-2*(q.y*q.y+q.z*q.z))])
            matched_time=self.rf2o_msg.header.stamp.sec+self.rf2o_msg.header.stamp.nanosec/1e9-100.
            fresh=abs(timestamp-matched_time)<.005 and np.isfinite(pose).all() and np.isfinite(scan).sum()>=12
            if fresh and self.rf2o_previous is not None:
                previous_pose,previous_time=self.rf2o_previous;dt=matched_time-previous_time
                proposed=increment(pose,previous_pose)
                if .01<dt<=.4 and np.linalg.norm(proposed[:2])<=.35*dt+.015 and abs(proposed[2])<=.75*dt+.03:
                    delta=proposed;accepted=True
            if fresh:self.rf2o_previous=(pose,matched_time)
            else:self.rf2o_previous=None
        a=self.last_odometry[2];c,s=np.cos(a),np.sin(a)
        self.last_odometry[:2]+=[c*delta[0]-s*delta[1],s*delta[0]+c*delta[1]]
        self.last_odometry[2]=wrap(a+delta[2])
        if accepted:self.rf2o_used+=1
        else:self.rf2o_fallback+=1
        return self.last_odometry.copy()

    def _mapping_scan(self,scan,odom,timestamp):
        """Keep transient returns in avoidance; withhold them from static SLAM.

        New structures in unknown space are retained. A return inside previously
        observed free space, far from static occupancy, must persist at the
        same 10cm world cell for 3s before it enters the static mapper. This
        provisional filter has no access to obstacle IDs or simulator truth.
        """
        m=self.current_map();pose=self.estimated_pose(odom)
        if m is None or pose is None:return scan
        valid=np.flatnonzero(np.isfinite(scan)&(scan>=self.lidar.range_min)&(scan<self.lidar.range_max))
        mount=self.lidar.mount;a=pose[2];c,s=np.cos(a),np.sin(a)
        origin=pose[:2]+[c*mount[0]-s*mount[1],s*mount[0]+c*mount[1]]
        angles=a+mount[3]+self.lidar.angles[valid]
        ends=origin+np.asarray(scan)[valid,None]*np.c_[np.cos(angles),np.sin(angles)]
        transient=m.lookup(ends,~m.blocked,outside=False)&(m.lookup(ends,m.distance,outside=0.)>.20)
        seen={}
        for index,point in zip(valid[transient],ends[transient]):
            key=tuple(np.floor(point/.1).astype(int));previous=self.transient_returns.get(key)
            beginning=timestamp if previous is None or timestamp-previous[1]>.4 else previous[0]
            seen[key]=(beginning,timestamp)
        output=np.asarray(scan,float).copy()
        for index,point in zip(valid[transient],ends[transient]):
            if timestamp-seen[tuple(np.floor(point/.1).astype(int))][0]<3.:
                output[index]=np.nan;self.filtered_returns+=1
        self.transient_returns=seen
        return output

    def estimated_pose(self,odom):
        from rclpy.time import Time
        try:
            msg=self.tf_buffer.lookup_transform('map','odom',Time())
            q=msg.transform.rotation
            angle=math.atan2(2*(q.w*q.z+q.x*q.y),1-2*(q.y*q.y+q.z*q.z))
            self.alignment=np.array([msg.transform.translation.x,msg.transform.translation.y,angle])
            self.last_alignment_stamp=msg.header.stamp.sec+msg.header.stamp.nanosec/1e9-100.
        except Exception:pass
        if self.alignment is None:return None
        x,y,a=self.alignment;c,s=math.cos(a),math.sin(a)
        return np.array([x+c*odom[0]-s*odom[1],y+s*odom[0]+c*odom[1],(a+odom[2]+np.pi)%(2*np.pi)-np.pi])

    def current_map(self):
        if self.map_msg is None:return None
        from navrl.map import StaticMap
        message=self.map_msg;p=message.info.origin
        yaw=math.atan2(2*(p.orientation.w*p.orientation.z+p.orientation.x*p.orientation.y),
            1-2*(p.orientation.y*p.orientation.y+p.orientation.z*p.orientation.z))
        return StaticMap.from_occupancy_grid(message.data,message.info.width,message.info.height,
            message.info.resolution,[p.position.x,p.position.y,yaw])

    def pose_metadata(self):
        """Age is tied to a scan-matched pose, not a periodically rebroadcast TF."""
        if self.last_pose_msg is None:return None,np.array([.20,.20,.15])
        msg=self.last_pose_msg
        stamp=msg.header.stamp.sec+msg.header.stamp.nanosec/1e9-100.
        covariance=np.asarray(msg.pose.covariance).reshape(6,6)
        # Floors are provisional, not a calibrated SLAM accuracy guarantee.
        variance=np.diag(covariance)[[0,1,5]]
        if not np.isfinite(variance).all() or np.any(variance<0):return stamp,np.array([.25,.25,.30])
        std=np.maximum(np.sqrt(np.maximum(variance,0)),[.04,.04,.03])
        return stamp,std

    def save(self):
        from slam_toolbox.srv import SerializePoseGraph
        client=self.node.create_client(SerializePoseGraph,'/slam_toolbox/serialize_map')
        if not client.wait_for_service(timeout_sec=2.):raise RuntimeError('Pose-graph serialization unavailable')
        request=SerializePoseGraph.Request();request.filename=str((self.output/'persistent_map').resolve())
        future=client.call_async(request);limit=time.monotonic()+10.
        while not future.done() and time.monotonic()<limit:self.spin_for(.05)
        if not future.done() or future.result().result!=0:raise RuntimeError('Pose-graph serialization failed')
        if self.map_msg is not None:
            m=self.current_map();np.savez_compressed(self.output/'occupancy_map.npz',
                data=np.asarray(self.map_msg.data,dtype=np.int16),width=self.map_msg.info.width,
                height=self.map_msg.info.height,resolution=m.res,origin=m.origin)
            from PIL import Image
            # 205 would be interpreted as FREE with free_thresh=0.25. Use a
            # mid-gray value that remains unknown on an exact YAML round trip.
            image=np.full(m.blocked.shape,127,np.uint8);image[~m.blocked]=254;image[m.occupied]=0
            Image.fromarray(image).save(self.output/'map.pgm')
            (self.output/'map.yaml').write_text(yaml.safe_dump(dict(image='map.pgm',resolution=m.res,
                origin=m.origin.tolist(),negate=0,occupied_thresh=.65,free_thresh=.25)))
        return dict(map_messages=self.map_version,pose_messages=self.pose_messages,
            experimental_distance_prior_variance=self.distance_prior_variance,
            transient_mapping_returns_withheld=self.filtered_returns,
            rf2o_accepted_frames=self.rf2o_used,rf2o_fallback_frames=self.rf2o_fallback,
            laser_increment_accepted_frames=self.increment_matcher.accepted if self.increment_matcher else 0,
            laser_increment_fallback_frames=self.increment_matcher.fallback if self.increment_matcher else 0,
            last_alignment_stamp_s=self.last_alignment_stamp,persistent_files=[p.name for p in self.output.glob('persistent_map*')])

    def load_graph(self,filename):
        """Resume at the first saved node; response alone does not prove success.

        Humble's response is empty even on read errors. Require a restored map
        publication before accepting the request as a successful reload.
        """
        from slam_toolbox.srv import DeserializePoseGraph
        filename=Path(filename).resolve()
        if not all(Path(str(filename)+extension).is_file() for extension in ('.posegraph','.data')):
            raise ValueError('Both saved pose graph and scan data are required')
        client=self.node.create_client(DeserializePoseGraph,'/slam_toolbox/deserialize_map')
        if not client.wait_for_service(timeout_sec=3.):raise RuntimeError('Pose-graph reload unavailable')
        request=DeserializePoseGraph.Request();request.filename=str(filename)
        request.match_type=DeserializePoseGraph.Request.START_AT_FIRST_NODE
        previous=self.map_version;future=client.call_async(request);limit=time.monotonic()+20.
        while not future.done() and time.monotonic()<limit:self.spin_for(.05)
        if not future.done() or future.exception() is not None:raise RuntimeError('Pose-graph reload request failed')
        while self.map_version==previous and time.monotonic()<limit:self.spin_for(.05)
        if self.map_version==previous or self.current_map() is None:raise RuntimeError('No map restored after pose-graph reload')
        return dict(graph=str(filename),map_restored=True,map_version=self.map_version,
            match_type='START_AT_FIRST_NODE',arbitrary_start_relocalization_tested=False)

    def close(self):
        if self.rf2o_process is not None:
            if self.rf2o_process.poll() is None:
                os.killpg(self.rf2o_process.pid,signal.SIGINT)
                try:self.rf2o_process.wait(timeout=3.)
                except subprocess.TimeoutExpired:
                    os.killpg(self.rf2o_process.pid,signal.SIGTERM);self.rf2o_process.wait(timeout=3.)
            self.rf2o_log.close()
        if self.process.poll() is None:
            os.killpg(self.process.pid,signal.SIGINT)
            try:self.process.wait(timeout=5.)
            except subprocess.TimeoutExpired:
                os.killpg(self.process.pid,signal.SIGTERM);self.process.wait(timeout=5.)
        self.log.close();self.executor.shutdown();self.node.destroy_node();self.rclpy.shutdown(context=self.context)
