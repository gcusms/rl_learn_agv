"""Genuine Nav2 NavFn/TEB action servers, fed real laser-localized state."""
import math,time,subprocess,json
import numpy as np
import yaml
from geometry_msgs.msg import Twist,PoseStamped
from nav2_msgs.action import ComputePathToPose,FollowPath
from nav_msgs.msg import Path as RosPath
from tf2_msgs.msg import TFMessage
from costmap_converter_msgs.msg import ObstacleArrayMsg,ObstacleMsg
from geometry_msgs.msg import Point32
from rclpy.action import ActionClient
from lifecycle_msgs.srv import ChangeState
from navrl.ros_amcl import RosAmclLocalizer

class NativeTebController:
    def __init__(self,env):
        self.env=env;old=env.localizer;env.localizer=RosAmclLocalizer(env.static_map,env.lidar,global_tf=True)
        self.l=env.localizer;self.node=self.l.node;self.exec=self.l.executor;self.processes=[];self.logs=[];self.goal=None;self.last_plan=-100.;self.command=np.zeros(2);self.command_stamp=0.;self.follow_handle=None
        self.node.create_subscription(Twist,'cmd_vel',self._cmd,10)
        self.obstacle_pub=self.node.create_publisher(ObstacleArrayMsg,'obstacles',10)
        self.local_plan_pose=None;self.tf_debug={}
        self.node.create_subscription(TFMessage,"/tf",self._debug_tf,100)
        self.node.create_subscription(RosPath,'local_plan',self._local_plan,10)
        footprint='[[0.197,0.165],[0.197,-0.165],[-0.197,-0.165],[-0.197,0.165]]'
        common=dict(use_sim_time=False,global_frame='map',robot_base_frame='base_footprint',update_frequency=10.,publish_frequency=5.,resolution=.05,transform_tolerance=.4,footprint=footprint,footprint_padding=.025,
            plugins=['static_layer','inflation_layer'],
            static_layer=dict(plugin='nav2_costmap_2d::StaticLayer',map_subscribe_transient_local=True,map_topic=f'/{self.l.namespace}/map'),
            obstacle_layer=dict(plugin='nav2_costmap_2d::ObstacleLayer',observation_sources='scan',scan=dict(topic=f'/{self.l.namespace}/scan',max_obstacle_height=2.,clearing=True,marking=True,data_type='LaserScan',raytrace_max_range=12.,raytrace_min_range=.1,obstacle_max_range=11.9,obstacle_min_range=.1,inf_is_valid=False)),
            inflation_layer=dict(plugin='nav2_costmap_2d::InflationLayer',cost_scaling_factor=4.,inflation_radius=.35))
        local={**common,'rolling_window':True,'width':4,'height':4}
        teb=dict(plugin='teb_local_planner::TebLocalPlannerROS',odom_topic=f'/{self.l.namespace}/odom',max_vel_x=.2,max_vel_x_backwards=.08,max_vel_theta=.5,acc_lim_x=.5,acc_lim_theta=1.,dt_ref=.2,dt_hysteresis=.05,global_plan_overwrite_orientation=True,max_global_plan_lookahead_dist=2.5,global_plan_viapoint_sep=.2,min_obstacle_dist=.06,inflation_dist=.2,include_costmap_obstacles=True,footprint_model={'type':'circular','radius':.26},include_dynamic_obstacles=False,costmap_converter_plugin='costmap_converter::CostmapToPolygonsDBSMCCH',costmap_converter_spin_thread=True,costmap_converter_rate=10,enable_homotopy_class_planning=True,enable_multithreading=False,no_inner_iterations=5,no_outer_iterations=4,weight_obstacle=100.,weight_optimaltime=1.,weight_kinematics_nh=1000.,weight_kinematics_forward_drive=2.,free_goal_vel=False)
        demo=getattr(env,'demo_configuration',None)
        if demo:
            teb.update(max_vel_x=demo['forward_limit'],max_vel_theta=demo['yaw_limit'])
        params={'/**/planner_server':{'ros__parameters':dict(use_sim_time=False,planner_plugins=['GridBased'],GridBased=dict(plugin='nav2_navfn_planner/NavfnPlanner',tolerance=.15,use_astar=True,allow_unknown=False),expected_planner_frequency=2.)},
                '/**/controller_server':{'ros__parameters':dict(use_sim_time=False,controller_frequency=10.,odom_topic=f'/{self.l.namespace}/odom',controller_plugins=['FollowPath'],FollowPath=teb,progress_checker_plugin='progress_checker',progress_checker=dict(plugin='nav2_controller::SimpleProgressChecker',required_movement_radius=.1,movement_time_allowance=30.),goal_checker_plugins=['goal_checker'],goal_checker=dict(plugin='nav2_controller::SimpleGoalChecker',xy_goal_tolerance=.10,yaw_goal_tolerance=.2,stateful=True),failure_tolerance=3.)},
                '/**/global_costmap':{'ros__parameters':common},'/**/local_costmap':{'ros__parameters':local}}
        config=self.l.log_dir/'nav2_teb.yaml';config.write_text(yaml.safe_dump(json.loads(json.dumps(params))));self.params=params
        for package,name in [('nav2_planner','planner_server'),('nav2_controller','controller_server')]:
            log=(self.l.log_dir/f'{name}.log').open('w');self.logs.append(log)
            command=[f'/opt/ros/humble/lib/{package}/{name}','--ros-args','--params-file',str(config),'-r',f'__ns:=/{self.l.namespace}','-r',f'/tf:=/{self.l.namespace}/tf','-r',f'/tf_static:=/{self.l.namespace}/tf_static']
            command=command[:-4]  # isolated DDS domain: use native global TF topics
            self.processes.append(subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT))
        self.configure=False
        self.planner=ActionClient(self.node,ComputePathToPose,'compute_path_to_pose');self.follower=ActionClient(self.node,FollowPath,'follow_path')
    def _debug_tf(self,msg):
        for t in msg.transforms:
            q=t.transform.rotation
            self.tf_debug[t.child_frame_id]=[t.transform.translation.x,t.transform.translation.y,math.atan2(2*(q.w*q.z+q.x*q.y),1-2*(q.y*q.y+q.z*q.z)),t.header.stamp.sec+t.header.stamp.nanosec/1e9]
    def _local_plan(self,msg):
        if msg.poses:
            p=msg.poses[0].pose;q=p.orientation
            self.local_plan_pose=[p.position.x,p.position.y,math.atan2(2*(q.w*q.z+q.x*q.y),1-2*(q.y*q.y+q.z*q.z))]
    def _cmd(self,msg):self.command=np.array([msg.linear.x,msg.angular.z]);self.command_stamp=time.monotonic()
    def _wait(self,future,timeout=8.):
        deadline=time.monotonic()+timeout;published=-100.
        while not future.done() and time.monotonic()<deadline:
            if time.monotonic()-published>.05:
                self.l._publish(self.env.odom,self.env.scan);published=time.monotonic()
            self.exec.spin_once(timeout_sec=.01)
        if not future.done():raise RuntimeError(f'Nav2 request timeout; logs {self.l.log_dir}')
        return future.result()
    def _activate(self):
        for name in ['planner_server','controller_server']:
            client=self.node.create_client(ChangeState,name+'/change_state')
            if not client.wait_for_service(timeout_sec=15):raise RuntimeError(f'{name} missing: {self.l.log_dir}')
            for transition in [1,3]:
                request=ChangeState.Request();request.transition.id=transition
                if not self._wait(client.call_async(request),20).success:raise RuntimeError(f'{name} transition failed: {self.l.log_dir}')
            self.node.destroy_client(client)
        for client in [self.planner,self.follower]:
            if not client.wait_for_server(timeout_sec=10):raise RuntimeError('Nav2 action server missing')
        self.configure=True
        for name,ns in self.node.get_node_names_and_namespaces():
            if self.l.namespace in ns and name in ['controller_server','local_costmap','global_costmap']:
                print('NATIVE TF SUBSCRIPTIONS',name,ns,[x for x in self.node.get_subscriber_names_and_types_by_node(name,ns) if 'tf' in x[0]],flush=True)
    def _publish_obstacles(self,e):
        # Real partial laser returns become map-frame TEB obstacles; no actor poses/truth.
        msg=ObstacleArrayMsg();msg.header.frame_id='map';msg.header.stamp=self.node.get_clock().now().to_msg()
        yaw=e.estimate[2];mount=e.lidar.mount
        origin=e.estimate[:2]+np.array([[math.cos(yaw),-math.sin(yaw)],[math.sin(yaw),math.cos(yaw)]])@np.asarray(mount[:2])
        points=[]
        for angle,distance in zip(e.lidar.angles,e.scan):
            if not np.isfinite(distance) or distance>3.:continue
            a=angle+yaw+mount[3];xy=origin+distance*np.array([math.cos(a),math.sin(a)])
            if e.static_map.lookup(np.asarray([xy]),e.static_map.distance,outside=0.)[0]<=.22:continue
            if points and min(np.linalg.norm(xy-p) for p in points)<.12:continue
            points.append(xy);ob=ObstacleMsg();ob.id=len(points);p=Point32();p.x=float(xy[0]);p.y=float(xy[1]);ob.polygon.points=[p];ob.radius=.02;msg.obstacles.append(ob)
        self.obstacle_pub.publish(msg)
    def action(self,e):
        self._publish_obstacles(e)
        self.l.odometry_velocity=e.measured_velocity.copy()
        if not self.configure:self._activate()
        changed=self.goal is None or not np.allclose(self.goal,e.goal)
        if changed or e.elapsed-self.last_plan>=.8:
            request=ComputePathToPose.Goal();request.planner_id='GridBased';request.use_start=True
            request.start=PoseStamped();request.start.header.frame_id='map';request.start.header.stamp=self.node.get_clock().now().to_msg();request.start.pose.position.x=float(e.estimate[0]);request.start.pose.position.y=float(e.estimate[1]);request.start.pose.orientation.z=math.sin(e.estimate[2]/2);request.start.pose.orientation.w=math.cos(e.estimate[2]/2)
            request.goal=PoseStamped();request.goal.header.frame_id='map';request.goal.header.stamp=self.node.get_clock().now().to_msg();request.goal.pose.position.x=float(e.goal[0]);request.goal.pose.position.y=float(e.goal[1]);request.goal.pose.orientation.z=math.sin(e.estimate[2]/2);request.goal.pose.orientation.w=math.cos(e.estimate[2]/2)
            handle=self._wait(self.planner.send_goal_async(request));result=self._wait(handle.get_result_async())
            if result.result.path.poses:
                e.path=np.array([[p.pose.position.x,p.pose.position.y] for p in result.result.path.poses]);e.predicted_path=e.path.copy()
                follow=FollowPath.Goal();follow.path=result.result.path;follow.controller_id='FollowPath';follow.goal_checker_id='goal_checker'
                self.follow_handle=self._wait(self.follower.send_goal_async(follow))
                if not self.follow_handle.accepted:raise RuntimeError('TEB rejected path')
            self.goal=e.goal.copy();self.last_plan=e.elapsed
        previous=self.command_stamp;deadline=time.monotonic()+1.
        while self.command_stamp<=previous and time.monotonic()<deadline:self.exec.spin_once(timeout_sec=.01)
        if e.steps%100==0:print('TEB POSE',self.local_plan_pose,'AMCL',e.estimate.tolist(),flush=True)
        command=self.command if time.monotonic()-self.command_stamp<.5 else np.zeros(2)
        limits=getattr(e,'demo_configuration',{})
        return np.clip([command[0]/(limits.get('forward_limit',.2) if command[0]>=0 else limits.get('reverse_limit',.08)),command[1]/limits.get('yaw_limit',.5)],-1.,1.)
    def close(self):
        for p in self.processes:
            if p.poll() is None:p.terminate()
        for p in self.processes:
            try:p.wait(timeout=3)
            except subprocess.TimeoutExpired:p.kill();p.wait(timeout=3)
        for log in self.logs:log.close()
