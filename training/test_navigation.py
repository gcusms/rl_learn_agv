"""Behavior tests for limited FOV, collision, localized observations and episode lifecycle."""
import unittest
import numpy as np
from navrl.env import NavigationEnv
from navrl.geometry import LidarProfile


class NavigationTests(unittest.TestCase):
    def setUp(self):
        self.env=NavigationEnv(localizer='particle',dynamic=False,max_steps=3)
        self.env.reset(seed=12,options={'task':0})
        self.env.lidar.noise=0.; self.env.lidar.dropout=0.
    def tearDown(self): self.env.close()

    def test_source_fov_and_urdf_mount(self):
        lidar=self.env.lidar
        self.assertAlmostEqual(np.degrees(lidar.angle_max-lidar.angle_min),219.4428355351053,places=6)
        np.testing.assert_allclose(lidar.mount,[.118389,0,.116682537,0])
        self.assertTrue(np.all(np.abs(lidar.angles)<np.pi))
        self.assertAlmostEqual(self.env.model.site_pos[0,2],-.039910)

    def test_obstacle_in_rear_blind_sector_is_unobservable(self):
        before=self.env._scan()
        mid=self.env.obstacle_mids[0]
        self.env.data.mocap_pos[mid]=[self.env.truth[0]-.65,self.env.truth[1],.5]
        import mujoco
        mujoco.mj_forward(self.env.model,self.env.data)
        np.testing.assert_array_equal(before,self.env._scan())

    def test_front_obstacle_is_observed(self):
        before=self.env._scan()
        mid=self.env.obstacle_mids[0]
        yaw=self.env.truth[2]
        self.env.data.mocap_pos[mid]=[* (self.env.truth[:2]+np.array([np.cos(yaw),np.sin(yaw)])*.9),.5]
        import mujoco
        mujoco.mj_forward(self.env.model,self.env.data)
        after=self.env._scan(); center=np.abs(self.env.lidar.angles)<.05
        self.assertLess(np.min(after[center]),.65)
        self.assertGreater(np.min(before[center]),1.)

    def test_collision_cannot_pass_through_wall(self):
        self.env.truth[:2]=[4.,1.]
        self.env._sync(); self.assertTrue(self.env._collision())
        self.env.truth[:2]=[4.,3.5]
        self.env._sync(); self.assertFalse(self.env._collision())

    def test_collision_is_recorded_but_does_not_terminate_or_cross_wall(self):
        self.env.max_steps=100
        self.env.truth[:2]=[3.68,1.]
        self.env.truth[2]=0.; self.env._sync()
        for _ in range(10):
            _,_,terminated,truncated,info=self.env.step([1,0])
            self.assertFalse(terminated)
            self.assertFalse(truncated)
        self.assertGreater(info['collision_events'],0)
        self.assertLess(self.env.truth[0],3.71)

    def test_ground_truth_is_not_a_policy_feature(self):
        scan=self.env._scan()
        before=self.env._observation(scan,initialize=True)
        self.env.truth+=np.array([5.,2.,1.])
        after=self.env._observation(scan,initialize=True)
        np.testing.assert_array_equal(before,after)
        self.assertEqual(before.shape,(228,))

    def test_global_path_respects_inflated_map(self):
        for task in self.env.scene['tasks']:
            path=self.env.map.plan(task['start'][:2],task['goal'])
            self.assertTrue(all(self.env.map.visible(a,b) for a,b in zip(path[:-1],path[1:])))

    def test_backward_action_does_not_enter_blind_rear(self):
        before=self.env.truth.copy()
        self.env.step([-1,0])
        np.testing.assert_allclose(self.env.truth,before,atol=1e-9)

    def test_timeout_and_repeated_reset(self):
        for _ in range(2):
            self.env.reset(seed=12,options={'task':0})
            for i in range(3):
                obs,reward,terminated,truncated,info=self.env.step([0,0])
                self.assertTrue(self.env.observation_space.contains(obs))
                self.assertFalse(terminated)
                self.assertEqual(truncated,i==2)


class RosAdapterTests(unittest.TestCase):
    def test_wraparound_real_scan_is_resampled_only_in_retained_sector(self):
        from types import SimpleNamespace
        from collections import deque
        from ros_policy_node import PolicyNode
        from sensor_msgs.msg import LaserScan
        node=object.__new__(PolicyNode)
        node.args=SimpleNamespace(lidar_frame='lidar_frame')
        node.lidar=LidarProfile(); node.history=deque(maxlen=3); node.scan_time=0.
        scan=LaserScan(); scan.header.frame_id='lidar_frame'
        scan.angle_min=0.; scan.angle_increment=float(np.pi/360); scan.angle_max=float(2*np.pi)
        scan.range_min=.1; scan.range_max=12.
        angle=np.arange(721)*scan.angle_increment
        values=np.full(721,2.)
        values[(angle>1.938407346410207)&(angle<4.391592653589793)]=np.inf
        scan.ranges=values.tolist()
        node.on_scan(scan)
        self.assertEqual(node.scan.shape,(220,))
        self.assertEqual(len(node.history),3)
        self.assertTrue(np.all(node.scan[1:-1]==2.))
        original_time=node.scan_time
        scan.header.frame_id='wrong_frame'
        node.on_scan(scan)
        self.assertEqual(node.scan_time,original_time)

    def test_stale_sensor_inputs_publish_zero_velocity(self):
        from types import SimpleNamespace
        from ros_policy_node import PolicyNode
        node=object.__new__(PolicyNode)
        node.scan=None; node.path=None; node.last_action=np.ones(2)
        outputs=[]; node.publisher=SimpleNamespace(publish=outputs.append)
        node.tick()
        self.assertEqual(outputs[0].linear.x,0.)
        self.assertEqual(outputs[0].angular.z,0.)
        np.testing.assert_array_equal(node.last_action,np.zeros(2))


if __name__=='__main__': unittest.main(verbosity=2)
