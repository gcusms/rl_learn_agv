"""Behavior checks for scene separation, reachability and laser-driven decisions."""
import unittest
import json
import numpy as np
from navrl.multimap import MultiMapNavigationEnv, DecisionNavigationEnv
from navrl.decision import NavigationDecision
from navrl.geometry import ROOT


class DecisionTests(unittest.TestCase):
    def test_train_test_maps_disjoint_and_tasks_randomized(self):
        m=json.loads((ROOT/'assets/navigation_dataset/manifest.json').read_text())
        hashes=[x['sha256'] for entries in m['splits'].values() for x in entries]
        self.assertEqual(len(hashes),len(set(hashes)))
        env=MultiMapNavigationEnv(split='train',localizer='particle',dynamic=False)
        try:
            for seed in [710,711,712]:
                obs,info=env.reset(seed=seed,options={'map_index':0});self.assertEqual(obs.shape,(228,))
                self.assertTrue(info['decision']['static_map_reachable'])
            self.assertGreater(len(set(tuple(t['start']) for t in env.task_history)),1)
        finally:env.close()

    def test_unreachable_goal_reported_without_motion(self):
        env=DecisionNavigationEnv(localizer='particle',dynamic=False)
        try:
            env.reset(seed=20,options={'start':[1,1,0],'goal':[0,0]})
            before=env.data.qpos.copy();_,_,done,timeout,info=env.step([1,0])
            self.assertTrue(done);self.assertFalse(timeout);self.assertEqual(info['status'],'unreachable')
            self.assertEqual(info['decision']['state'],'static_unreachable');np.testing.assert_array_equal(before,env.data.qpos)
        finally:env.close()

    def test_decision_observation_does_not_use_truth(self):
        env=DecisionNavigationEnv(localizer='particle',dynamic=True)
        try:
            env.reset(seed=31,options={'start':[1,1,0],'goal':[3,1]})
            a=env._observation(env.scan,initialize=True);env.truth=np.array([8,7,2.])
            b=env._observation(env.scan,initialize=True);np.testing.assert_array_equal(a,b)
        finally:env.close()

    def test_solid_pedestrian_is_visible_to_actual_mujoco_laser(self):
        import mujoco
        env=DecisionNavigationEnv(localizer='particle',dynamic=True)
        try:
            env.reset(seed=15,options={'start':[1,1,0],'goal':[3,1]})
            env.truth=np.array([1.,1.,0.]);env._sync()
            q,d=env.addresses[env.names[1]];env.data.qpos[q]=[2,1];env.data.qvel[d]=0
            mujoco.mj_forward(env.model,env.data)
            env.scan=env._scan();front=env.scan[np.abs(env.lidar.angles)<.05]
            self.assertLess(np.min(front),.8)
            env.estimate=env.truth.copy();env.decision=NavigationDecision(env.map);env.path=env.map.plan([1,1],[3,1])
            env.decision.update(env);self.assertGreater(env.decision.revision,0)
        finally:env.close()

    def test_unknown_laser_obstacle_changes_route_without_truth_access(self):
        env=DecisionNavigationEnv(localizer='particle',dynamic=False)
        try:
            env.reset(seed=30,options={'start':[1,1,0],'goal':[3,1]})
            # Synthetic scan of a narrow unknown obstacle at x=2, y=1.
            # Only scan and estimated pose are modified: no human body positions are supplied.
            env.estimate=np.array([1.,1.,0.]);env.scan=np.full(220,np.inf)
            angles=env.lidar.angles;mask=np.abs(angles)<.18;env.scan[mask]=.88/np.cos(angles[mask])
            env.path=env.map.plan(env.estimate[:2],env.goal);env.decision=NavigationDecision(env.map)
            env.decision.update(env)
            self.assertGreater(env.decision.revision,0);self.assertGreater(env.decision.local_points,0)
            self.assertGreater(np.max(np.abs(env.path[:,1]-1)),.25)
            for a,b in zip(env.path[:-1],env.path[1:]):self.assertTrue(env.decision.local_map.visible(a,b))
        finally:env.close()


if __name__=='__main__':unittest.main(verbosity=2)
