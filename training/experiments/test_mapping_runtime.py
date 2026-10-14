"""Coordinate, freshness and resident-state regressions for map navigation."""
import ast
import math
import subprocess
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
import numpy as np
from navrl.map import StaticMap
from neural_navigation.online_mapping_runtime import select_observed_subgoal,SlamPoseLocalizer


class MappingTests(unittest.TestCase):
    def test_partial_map_dynamic_layer_keeps_free_space_obstacles_and_raw_scan(self):
        from neural_navigation.core import NavigationCore
        from neural_navigation.laser_fusion import LaserMapFusion
        grid=np.zeros((64,100),np.int8);grid[:,60:]=-1
        m=StaticMap.from_occupancy_grid(grid.ravel(),100,64,.05,[0,0,0])
        core=NavigationCore.__new__(NavigationCore);core.static_map=m
        core.grid_res=.2;core.free=m.free[2::4,2::4].astype(np.float32);core.grid_h,core.grid_w=core.free.shape
        core.elapsed=1.;core.estimate=np.array([1.,1.,0.]);core.scan=np.array([1.,3.])
        core.lidar=NS(range_min=.1,range_max=10.,mount=np.zeros(4),angles=np.zeros(2),angle_min=0.,angle_max=0.)
        core.dynamic_memory_seconds=1.
        core.expiry=np.zeros((core.grid_h,core.grid_w))
        core.laser_fusion=LaserMapFusion(core)
        core._routing_dynamic_layer();self.assertEqual(core.unmatched_returns,2)
        core.expiry.fill(0);core.laser_fusion.reset();core.routing_dynamic_known_free_only=True
        layer=core._routing_dynamic_layer();self.assertEqual(core.unmatched_returns,1)
        self.assertGreater(layer[core.cell([2.,1.])],0)
        self.assertEqual(layer[core.cell([4.,1.])],0)
        np.testing.assert_array_equal(core.scan,[1.,3.]);self.assertTrue(m.unknown[:,60:].all())

    def test_command_odometry_integrates_published_twist_without_motion_feedback(self):
        from experiments.command_odometry_input import CommandOdometryInput
        from navrl.geometry import wrap
        odom=CommandOdometryInput()
        for _ in range(200):odom.advance([.1,.2],.2)
        # Forty seconds of a circular command; no measured or true pose is
        # supplied. This intentionally continues to integrate if a car stalls.
        np.testing.assert_allclose(odom.pose,[.5*np.sin(8.),.5*(1-np.cos(8.)),wrap(8.)],atol=1e-10)
        np.testing.assert_allclose(odom.velocity,[.1,.2])
        pose=odom.pose.copy();odom.advance([0.,0.],.2)
        np.testing.assert_array_equal(odom.pose,pose)
        np.testing.assert_array_equal(odom.velocity,[0.,0.])

    def test_command_odometry_rejects_nonphysical_input(self):
        from experiments.command_odometry_input import CommandOdometryInput
        for scale in [0.,-1.,np.inf,np.nan]:
            with self.assertRaises(ValueError):CommandOdometryInput(scale,1.)
        odom=CommandOdometryInput()
        for command,dt in [([np.nan,0.],.2),([.1],.2),([.1,0.],0.),([.1,0.],np.inf)]:
            with self.assertRaises(ValueError):odom.advance(command,dt)

    def test_gyro_odom_uses_sensor_turn_but_keeps_command_translation(self):
        from experiments.command_odometry_input import CommandImuOdometryInput
        gyro=CommandImuOdometryInput(gyro_weight=1.)
        gyro.advance([.2,.5],1.,0.,0.)
        np.testing.assert_allclose(gyro.pose,[.2,0.,0.]);np.testing.assert_allclose(gyro.velocity,[.2,0.])
        gyro.advance([0.,0.],1.,.1,.12)
        np.testing.assert_allclose(gyro.pose,[.2,0.,.1]);np.testing.assert_allclose(gyro.velocity,[0.,.12])
        # The sensor is allowed to measure a turn after the command has stopped.
        # No wheel translation or reference pose can enter this interface.
        for weight in [-.1,1.1,np.nan]:
            with self.assertRaises(ValueError):CommandImuOdometryInput(weight)

    def test_bottom_up_grid_padding_keeps_world_coordinates_and_unknown(self):
        grid=np.zeros((5,7),np.int8);grid[0,0]=100;grid[3,4]=-1
        m=StaticMap.from_occupancy_grid(grid.ravel(),7,5,.05,[-2.,-3.,0.])
        self.assertEqual(m.free.shape,(8,8))
        self.assertTrue(m.lookup([-1.975,-2.975],m.occupied,outside=False))
        self.assertTrue(m.lookup([-1.775,-2.825],m.unknown,outside=False))
        self.assertTrue(m.unknown[:3].all());self.assertTrue(m.unknown[:,-1].all())
        np.testing.assert_allclose(m.xy(m.cells([-1.775,-2.825])),[-1.775,-2.825])
        with self.assertRaises(ValueError):StaticMap.from_occupancy_grid(grid.ravel(),7,5,np.nan,[0,0,0])

    def test_frontier_does_not_select_disconnected_free_component(self):
        grid=np.full((64,100),-1,np.int8);grid[5:59,5:40]=0;grid[5:59,60:95]=0
        m=StaticMap.from_occupancy_grid(grid.ravel(),100,64,.05,[0,0,0])
        subgoal,final=select_observed_subgoal(m,np.array([1.,1.5,0.]),[6.,1.5,0.])
        self.assertFalse(final);self.assertLess(subgoal[0],2.)

    def test_frontier_does_not_cross_a_break_in_actual_neural_grid(self):
        m=StaticMap.from_occupancy_grid(np.zeros(64*100),100,64,.05,[0,0,0])
        m.free[:]=False;m.free[5:59,5:40]=True;m.free[5:59,60:95]=True
        m.free[29,40:60]=True  # Fine-grid bridge missed by the router's samples.
        pose=np.array([1.,1.5,0.]);goal=np.array([4.,1.5,0.])
        _,fine_final=select_observed_subgoal(m,pose,goal)
        subgoal,coarse_final=select_observed_subgoal(m,pose,goal,routing_grid=True)
        self.assertTrue(fine_final);self.assertFalse(coarse_final);self.assertLess(subgoal[0],2.)

    def test_saved_unknown_occupancy_stays_unknown_on_yaml_round_trip(self):
        import tempfile,yaml
        from PIL import Image
        grid=np.zeros((8,8),np.int8);grid[2,2]=-1;grid[4,4]=100
        m=StaticMap.from_occupancy_grid(grid.ravel(),8,8,.05,[-2.,-3.,0.])
        image=np.full(m.blocked.shape,127,np.uint8);image[~m.blocked]=254;image[m.occupied]=0
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder);Image.fromarray(image).save(p/'map.pgm')
            (p/'map.yaml').write_text(yaml.safe_dump(dict(image='map.pgm',resolution=m.res,
                origin=m.origin.tolist(),negate=0,occupied_thresh=.65,free_thresh=.25)))
            restored=StaticMap(p/'map.yaml',radius=.30)
        np.testing.assert_array_equal(restored.unknown,m.unknown)
        np.testing.assert_array_equal(restored.occupied,m.occupied)
        np.testing.assert_allclose(restored.origin,m.origin)

    def test_ray_free_layer_never_clears_occupied_or_occluded_unknown(self):
        from neural_navigation.online_mapping_runtime import clear_observed_ray_unknown
        grid=np.full((40,50),-1,np.int8);grid[20,20]=100
        m=StaticMap.from_occupancy_grid(grid.ravel(),50,40,.05,[0,0,0])
        spec=NS(range_min=.1,range_max=10.,mount=np.zeros(4),angles=np.array([0.]))
        pose=np.array([.525,1.025,0.]);new=clear_observed_ray_unknown(m,pose,np.array([1.5]),spec)
        self.assertFalse(new.lookup([.775,1.025],new.unknown,outside=False))
        self.assertTrue(new.lookup([1.025,1.025],new.occupied,outside=False))
        self.assertTrue(new.lookup([1.275,1.025],new.unknown,outside=False))
        self.assertIs(clear_observed_ray_unknown(m,pose,np.array([np.inf]),spec),m)
        self.assertTrue(m.lookup([.775,1.025],m.unknown,outside=False))

    def test_mapping_pause_updates_sensors_and_records_executed_stop(self):
        from neural_navigation.online_mapping_runtime import OnlineMappingRuntime
        from unittest.mock import Mock,patch
        owner=OnlineMappingRuntime.__new__(OnlineMappingRuntime)
        owner.core=NS(localizer=NS(supply=Mock()),static_map=None,update=Mock(),record_action=Mock())
        from neural_navigation.safety_context import SafetyContext
        owner.runtime=NS(context=SafetyContext());owner.requested_goal=np.array([1.,2.,0.])
        with patch('neural_navigation.online_mapping_runtime.select_observed_subgoal',return_value=(None,False)):
            result=owner.step(np.ones(220),np.zeros(3),np.ones(2),2.,None,np.zeros(3),np.ones(3)*.05)
        owner.core.update.assert_called_once();owner.core.record_action.assert_called_once_with([0.,0.])
        np.testing.assert_array_equal(owner.runtime.context.last_executed,[0.,0.])
        self.assertEqual(owner.runtime.context.last_reason,'other')
        self.assertEqual(result['command'],[0.,0.]);self.assertFalse(result['task_goal_achieved'])

    def test_stale_pose_uses_odometry_and_large_correction_is_explicit(self):
        template=NS(pose=np.zeros(3),std=np.array([.04,.04,.03]),last_odom=None,missing_scan_updates=0,
            map=None,lidar=None)
        localizer=SlamPoseLocalizer(template)
        from unittest.mock import patch
        with patch('neural_navigation.online_mapping_runtime.scan_map_agreement',return_value=.9):
            localizer.supply([0,0,0],0.,[.04,.04,.03]);localizer.update(np.zeros(3),np.ones(220),0.)
            localizer.update(np.array([.1,0,0]),np.ones(220),2.)
            self.assertFalse(localizer.scan_update_accepted)
            np.testing.assert_allclose(localizer.pose,[.1,0,0])
            localizer.supply([.5,0,0],2.2,[.04,.04,.03]);localizer.update(np.array([.12,0,0]),np.ones(220),2.2)
            self.assertTrue(localizer.large_correction)

    def test_body_layer_clears_only_unknown_under_actual_rectangle(self):
        from neural_navigation.online_mapping_runtime import clear_body_unknown
        grid=np.zeros((40,40),np.int8);grid[20,20]=-1;grid[20,21]=100;grid[30,30]=-1
        m=StaticMap.from_occupancy_grid(grid.ravel(),40,40,.05,[0,0,0])
        new=clear_body_unknown(m,[1.025,1.025,0.])
        self.assertTrue(m.lookup([1.025,1.025],m.unknown,outside=False))
        self.assertFalse(new.lookup([1.025,1.025],new.unknown,outside=False))
        self.assertTrue(new.lookup([1.075,1.025],new.occupied,outside=False))
        self.assertTrue(new.lookup([1.525,1.525],new.unknown,outside=False))

    def test_heading_noise_scales_increments_across_pi_without_fake_jumps(self):
        from experiments.probe_online_slam import OdometryProfile
        from navrl.geometry import wrap
        profile=OdometryProfile(np.zeros(3),yaw_scale=1.025)
        values=[profile.update(np.array([0,0,wrap(a)]),a)[2] for a in np.linspace(0,7.,71)]
        np.testing.assert_allclose(wrap(np.diff(values)),.1025,atol=1e-12)

    def test_stress_wrapper_heading_scaling_preserves_wrap_and_reset(self):
        from neural_navigation.stress_env import CorruptedOdometry
        from navrl.geometry import wrap
        class Environment:
            case={'yaw_odom_scale':1.025};elapsed=0.;measured_velocity=np.zeros(2)
        env=Environment();base=NS(update=lambda odom,scan:odom.copy(),reset=lambda:None)
        wrapper=CorruptedOdometry(base,env)
        values=[wrapper.update(np.array([0,0,wrap(a)]),np.ones(220))[2] for a in np.linspace(0,7.,71)]
        np.testing.assert_allclose(wrap(np.diff(values)),.1025,atol=1e-12)
        wrapper.reset();np.testing.assert_array_equal(wrapper.update(np.zeros(3),np.ones(220)),np.zeros(3))

    def test_live_map_resize_preserves_world_memory_localization_and_networks(self):
        from neural_navigation.core import NavigationCore
        from neural_navigation.laser_fusion import LaserMapFusion
        core=NavigationCore.__new__(NavigationCore)
        old=StaticMap.from_occupancy_grid(np.zeros(64*64),64,64,.05,[0,0,0])
        core.static_map=old;core.free=old.free[2::4,2::4].astype(np.float32)
        core.grid_h,core.grid_w=core.free.shape;core.grid_res=.2;core.elapsed=2.
        core.expiry=np.zeros_like(core.free);core.router_cache={'old':'entry'};core.static_visibility_cache={'old':True}
        pose=np.array([1.,1.,.5]);network=object();core.localizer=NS(map=old,pose=pose,network=network)
        core.laser_fusion=LaserMapFusion(core);core.laser_fusion.hard_expiry[4,5]=8.
        remembered=core.laser_fusion.points[4,5].copy()
        expanded=StaticMap.from_occupancy_grid(np.zeros(80*88),88,80,.05,[-.2,-.4,0])
        core.replace_map(expanded)
        self.assertIs(core.localizer.pose,pose);self.assertIs(core.localizer.network,network)
        self.assertFalse(core.router_cache);self.assertFalse(core.static_visibility_cache)
        r,c=expanded.cells(remembered)
        self.assertEqual(core.laser_fusion.hard_expiry[r//4,c//4],8.)

    def test_static_mapper_withholds_new_free_space_return_not_known_wall(self):
        from experiments.slam_toolbox_backend import SlamBackend
        grid=np.zeros((64,80),np.int8);grid[:,60]=100
        m=StaticMap.from_occupancy_grid(grid.ravel(),80,64,.05,[0,0,0])
        backend=SlamBackend.__new__(SlamBackend);backend.lidar=NS(range_min=.1,range_max=12.,mount=np.zeros(4),angles=np.array([0.,0.]))
        backend.transient_returns={};backend.filtered_returns=0
        backend.current_map=lambda:m;backend.estimated_pose=lambda odom:np.array([1.,1.,0.])
        scan=np.array([.7,2.02]);filtered=backend._mapping_scan(scan,np.zeros(3),0.)
        self.assertTrue(np.isnan(filtered[0]));self.assertEqual(filtered[1],scan[1]);np.testing.assert_array_equal(scan,[.7,2.02])
        for timestamp in np.arange(.2,3.4,.2):filtered=backend._mapping_scan(scan,np.zeros(3),float(timestamp))
        self.assertEqual(filtered[0],scan[0])

    def test_portable_module_avoids_training_and_physics_imports(self):
        code="import sys; from neural_navigation.goal_pose_runtime import DockingRuntime; assert not any(x in sys.modules for x in ('gymnasium','stable_baselines3','sb3_contrib','mujoco','rclpy'))"
        subprocess.run([sys.executable,'-c',code],check=True,capture_output=True)

    def test_terminal_ros_goal_preserves_quaternion_heading(self):
        source=Path(__file__).resolve().parents[2]/'deployment/ros2_recording_adapter/model_node.py'
        tree=ast.parse(source.read_text());method=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='on_goal')
        namespace=dict(math=math,yaw=lambda q:2*math.atan2(q.z,q.w))
        exec(compile(ast.Module(body=[method],type_ignores=[]),str(source),'exec'),namespace)
        owner=NS(model_identity={'sensor_core':'learned-terminal-goal-pose'},map_alignment=None,buffer=NS(rows=[]),new_mission=lambda:None)
        msg=NS(header=NS(frame_id='map'),pose=NS(position=NS(x=1.,y=2.),orientation=NS(z=math.sin(.6),w=math.cos(.6))))
        namespace['on_goal'](owner,msg);np.testing.assert_allclose(owner.goal,[1.,2.,1.2])

    def test_goal_change_reuses_runtime_but_initial_pose_reset_recreates_it(self):
        source=Path(__file__).resolve().parents[2]/'deployment/ros2_recording_adapter/inference_worker.py'
        tree=ast.parse(source.read_text());cls=next(n for n in tree.body if isinstance(n,ast.ClassDef))
        namespace=dict(np=np,time=__import__('time'),static_footprint_clear=lambda *args:True,
            limit_command=lambda command,*args:(command,1.))
        exec(compile(ast.Module(body=[cls],type_ignores=[]),str(source),'exec'),namespace)
        worker=namespace['InferenceWorker'].__new__(namespace['InferenceWorker'])
        worker.settings=dict(bundle='bundle',map_path='map',route_mode='shortcut',max_rps=1.)
        worker.geometry=None;worker.runtime=None;worker.generation=None;worker.localization_generation=None;created=[]
        def factory(*args):
            core=NS(set_goal=lambda g:setattr(core,'goal',g),record_action=lambda a:None,
                static_map=None,estimate=np.zeros(3),path=np.zeros((2,2)))
            runtime=NS(core=core,guard=NS(half=np.ones(2),static_margin=.04),context=None,
                step=lambda *a,**k:dict(command=[0.,0.]))
            created.append(runtime);return runtime
        worker.factory=factory;frame=(np.ones(220),np.zeros(3),np.zeros(2),0.,True)
        worker.infer(frame,1,[0,0,0],[1,1],0.);first=worker.runtime
        worker.infer(frame,2,[9,9,9],[2,2,1.],0.)
        self.assertIs(worker.runtime,first);self.assertEqual(worker.runtime.core.goal,[2,2,1.]);self.assertEqual(len(created),1)
        worker.infer(frame,3,[0,0,0],[2,2,1.],0.,localization_generation=1)
        self.assertIsNot(worker.runtime,first);self.assertEqual(len(created),2)


if __name__=='__main__':unittest.main()
