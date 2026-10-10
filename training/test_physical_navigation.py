"""Behavioral checks for reciprocal yielding and native solid contacts."""
import unittest
import numpy as np
import mujoco
from navrl.physical_env import PhysicalNavigationEnv


class SolidSceneTests(unittest.TestCase):
    def setUp(self):
        self.e=PhysicalNavigationEnv(localizer='particle',max_steps=200)
        self.e.reset(seed=6100,options={'task':0})
    def tearDown(self):self.e.close()
    def place(self,robot,humans):
        self.e.truth=np.array(robot,float);self.e._sync()
        for name,pos in zip(self.e.names[1:],humans):
            q,d=self.e.addresses[name];self.e.data.qpos[q]=pos;self.e.data.qvel[d]=0
        mujoco.mj_forward(self.e.model,self.e.data)
        self.e.minimum_gap=self.e._minimum_gap()
    def assert_separated(self):
        self.assertGreaterEqual(self.e._minimum_gap(),-.0001)
        for c in self.e.data.contact:self.assertGreaterEqual(c.dist,-.0001)
    def test_native_dynamic_bodies_and_time_integration(self):
        self.assertEqual(self.e.model.nmocap,0)
        self.assertEqual(self.e.model.nv,7)
        self.e.step([.8,0]);self.assertAlmostEqual(self.e.data.time,.2)
        self.assertGreater(np.linalg.norm(self.e.data.qvel),0)
    def test_pedestrian_yields_to_stationary_robot_without_overlap(self):
        self.place([4.8,3.2,0],[[1.,5.],[4.8,2.]])
        self.e.walkers[0]['pause']=100.
        w=self.e.walkers[1];w['path']=self.e.human_map.plan([4.8,2.],[4.8,6.]);w['goal']=np.array([4.8,6.]);w['speed']=.8
        yielded=False;stopped=False
        for _ in range(70):
            self.e.step([0,0]);self.assert_separated()
            yielded|=w['state']=='yielding'
            _,d=self.e.addresses[self.e.names[2]]
            stopped|=w['state']=='yielding' and np.linalg.norm(self.e.data.qvel[d])<.03
        self.assertTrue(yielded);self.assertTrue(stopped)
    def test_robot_cannot_drive_through_paused_pedestrian(self):
        self.place([2.8,3.5,0],[[3.45,3.5],[8.5,7.]])
        for w in self.e.walkers:w['pause']=100.
        for _ in range(50):
            self.e.step([1,0]);self.assert_separated()
        self.assertLess(self.e.truth[0],3.45-.25-.1969+.02)
        self.assertGreater(self.e.native_contact_steps,0)
    def test_two_pedestrians_cannot_cross_through_each_other(self):
        self.place([1.,1.,0],[[2.6,3.5],[3.6,3.5]])
        for i,(start,goal) in enumerate([([2.6,3.5],[3.6,3.5]),([3.6,3.5],[2.6,3.5])]):
            w=self.e.walkers[i];w['path']=self.e.human_map.plan(start,goal);w['goal']=np.array(goal);w['speed']=.8
        for _ in range(60):self.e.step([0,0]);self.assert_separated()
    def test_native_wall_contacts_block_forced_pedestrian_motion(self):
        self.place([1.,1.,0],[[3.4,1.5],[8.5,7.]])
        for _ in range(300):
            self.e.data.qfrc_applied[:]=0
            self.e._set_velocity_force(self.e.names[1],np.array([1.,0.]),1.8)
            mujoco.mj_step(self.e.model,self.e.data);mujoco.mj_forward(self.e.model,self.e.data)
        self.assertLessEqual(self.e._humans()[0][0],3.65)
        self.assertGreaterEqual(self.e.wall_gap(self.e._humans()[0]),-.001)
    def test_resets_have_no_spawn_overlap(self):
        for seed in range(6101,6107):
            self.e.reset(seed=seed,options={'task':seed%3});self.assert_separated()


if __name__=='__main__':unittest.main(verbosity=2)
