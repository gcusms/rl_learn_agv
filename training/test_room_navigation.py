import unittest
import numpy as np
import torch
import mujoco
from navrl.geometry import ROOT
from navrl.arm_home import home_configuration
from neural_navigation.env import LearnedNavigationEnv
from navrl.scenarios import configure_column_task

class RoomTests(unittest.TestCase):
    def test_imported_arm_reaches_source_home_in_mujoco(self):
        model=mujoco.MjModel.from_xml_path(str(ROOT/'assets/room_navigation/physical_scene.xml'));data=mujoco.MjData(model);mujoco.mj_forward(model,data)
        base=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,'urdf_arm_base_link');wrist=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,'urdf_link4')
        rot=data.xmat[wrist].reshape(3,3);tool=data.xpos[wrist]+rot@np.array([.13023,-.01835,0])
        local=data.xmat[base].reshape(3,3).T@(tool-data.xpos[base]);home=home_configuration()['home_pose']
        np.testing.assert_allclose(local,np.array([home['x'],home['y'],home['z']])/1000,atol=2e-5)
        self.assertAlmostEqual(np.degrees(np.arctan2(rot[2,0],rot[0,0])),home['pitch'],places=5)

    def test_target_transition_preserves_robot_and_localization(self):
        torch.set_num_threads(4);e=LearnedNavigationEnv(scene_dir=ROOT/'assets/room_navigation')
        try:
            e.reset(seed=27000,options={'task':0});configure_column_task(e,0);e.step([.2,.1])
            q=e.data.qpos.copy();odom=e.odom.copy();pose=e.estimate.copy();time=e.elapsed;loc=e.localizer;events=e._info('running')['column_block_events']
            obs=e.set_navigation_goal([8.6,6.6]);self.assertEqual(obs.shape,(231,))
            np.testing.assert_array_equal(e.data.qpos,q);np.testing.assert_array_equal(e.odom,odom);np.testing.assert_array_equal(e.estimate,pose)
            self.assertIs(e.localizer,loc);self.assertEqual(e.elapsed,time);self.assertEqual(e._info('running')['column_block_events'],events)
            with self.assertRaises(ValueError):e.set_navigation_goal([5,7])
        finally:e.close()

if __name__=='__main__':unittest.main(verbosity=2)
