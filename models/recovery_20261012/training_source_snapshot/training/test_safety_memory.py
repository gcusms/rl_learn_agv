import unittest
from types import SimpleNamespace
import numpy as np
from navrl.geometry import LidarProfile
from neural_navigation.runtime import EmergencyStop
from neural_navigation.safety_memory import NearFieldMemory


class SafetyMemoryTests(unittest.TestCase):
    def core(self):
        lidar=LidarProfile()
        return SimpleNamespace(scan=np.full(lidar.beams,np.inf),lidar=lidar,odom=np.zeros(3),elapsed=0.,estimate=np.zeros(3),
                               localizer=SimpleNamespace(std=np.array([.02,.02,.01])),odometry_sanity_failure=False,
                               static_map=SimpleNamespace(free=np.ones((1,1)),lookup=lambda *a,**kw:True),diagnostics=lambda:{'route_complete':True})
    def test_unseen_rear_points_are_not_deleted_by_time_or_map_pose_correction(self):
        c=self.core();c.scan[np.argmin(abs(c.lidar.angles))]=1.
        memory=NearFieldMemory({'near_range_m':1.5});points,_=memory.update(c);self.assertGreater(len(points),0)
        c.scan[:]=np.inf;c.odom[2]=np.pi;c.elapsed=20.;c.estimate[:2]=[4.,5.]
        rear,uncertainty=memory.update(c);self.assertGreater(len(rear),0);self.assertTrue(np.all(rear[:,0]<0));self.assertTrue(np.all(uncertainty<=.04))
        c.odom[2]=0.;c.scan[:]=2.;c.elapsed=20.2
        points,_=memory.update(c);self.assertEqual(len(points),0)

    def test_memory_blocks_unsafe_turn_but_preserves_requested_forward_motion(self):
        c=self.core();c.scan[:]=2.;c.elapsed=1.
        guard=EmergencyStop(memory_config={'near_range_m':1.5})
        guard.memory.entries[(-4,5)]=(np.array([-.15,.22]),.8)
        command,reason=guard.check(c,[.1,-.5],1.,1.)
        np.testing.assert_allclose(command,[.1,0.]);self.assertEqual(reason,'unsafe_turn_blocked')
        # The same hidden point is missed by a current-scan-only guard.
        original,reason=EmergencyStop().check(c,[.1,-.5],1.,1.)
        np.testing.assert_allclose(original,[.1,-.5]);self.assertIsNone(reason)

    def test_only_clearance_increasing_escape_is_allowed_inside_margin(self):
        c=self.core();guard=EmergencyStop(memory_config={'near_range_m':1.5})
        rear=np.array([[-.15,.20]])
        self.assertIsNone(guard._trajectory_check(c,np.array([.2,0]),rear,np.array([.04]),0.,True))
        front=np.array([[.19,.20]])
        self.assertIsNotNone(guard._trajectory_check(c,np.array([.2,0]),front,np.array([.04]),0.,True))

    def test_free_evidence_requires_distinct_contiguous_frames_and_unknown_resets_it(self):
        c=self.core();memory=NearFieldMemory({'free_evidence_margin_m':.045,'free_evidence_frames':3})
        memory.entries[(20,0)]=(np.array([.8,0.]),0.)
        c.scan[:]=2.;memory.update(c);memory.update(c)
        self.assertEqual(len(memory.entries),1)
        c.elapsed=.2;memory.update(c);self.assertEqual(len(memory.entries),1)
        c.elapsed=.4;c.scan[:]=np.nan;memory.update(c)
        c.scan[:]=2.
        for t in [.6,.8]:
            c.elapsed=t;memory.update(c);self.assertEqual(len(memory.entries),1)
        c.elapsed=1.;memory.update(c);self.assertEqual(len(memory.entries),0)

    def test_obstacle_return_and_rear_blind_sector_never_confirm_free(self):
        c=self.core();memory=NearFieldMemory({'free_evidence_margin_m':.045,'free_evidence_frames':3})
        memory.entries[(20,0)]=(np.array([.8,0.]),0.)
        c.scan[:]=.7
        for t in [.2,.4,.6]:c.elapsed=t;memory.update(c)
        self.assertIn((20,0),memory.entries)
        c.odom[2]=np.pi;c.scan[:]=2.
        for t in [.8,1.,1.2]:c.elapsed=t;memory.update(c)
        self.assertIn((20,0),memory.entries)

    def test_rotation_escape_requires_increasing_clearance_without_translation(self):
        c=self.core();guard=EmergencyStop(memory_config={'clearance_increasing_rotation':True})
        point=np.array([[.15,.20]])
        self.assertIsNone(guard._trajectory_check(c,np.array([0.,-.5]),point,np.array([.04]),0.,True))
        self.assertIsNotNone(guard._trajectory_check(c,np.array([0.,.5]),point,np.array([.04]),0.,True))
        self.assertIsNotNone(guard._trajectory_check(c,np.array([0.,-.5]),np.vstack([point,[.15,-.20]]),np.array([.04,.04]),0.,True))
        default=EmergencyStop(memory_config={'near_range_m':1.5})
        self.assertIsNotNone(default._trajectory_check(c,np.array([0.,-.5]),point,np.array([.04]),0.,True))


if __name__=='__main__':unittest.main(verbosity=2)
