import unittest
from unittest.mock import patch
import numpy as np
from neural_navigation.core import RoomRoutingMixin
from prototypes.narrow_alignment import AlignmentNarrowCore
from experiments.fresh_perception_runtime import FreshPerceptionCore


class AlignmentStateTests(unittest.TestCase):
    def core(self):
        core=object.__new__(AlignmentNarrowCore)
        core.estimate=np.zeros(3);core.current_probability=.9
        core.predicted_path=np.array([[0.,0.],[1.,0.],[2.,0.]])
        return core

    def test_alignment_keeps_direction_but_never_claims_complete_motion(self):
        core=self.core();original=core.predicted_path.copy()
        with patch.object(RoomRoutingMixin,'_neural_route'),patch.object(RoomRoutingMixin,'_route_usable',return_value=True),patch('prototypes.narrow_alignment.path_turns_clear',side_effect=[True,False]):
            core._neural_route()
        np.testing.assert_array_equal(core.predicted_path,original)
        self.assertTrue(core.alignment_required);self.assertFalse(core.motion_route_valid)
        self.assertEqual(core.current_probability,0)

    def test_blocked_route_retains_only_a_checked_local_direction(self):
        core=self.core()
        with patch.object(RoomRoutingMixin,'_neural_route'),patch.object(RoomRoutingMixin,'_route_usable',return_value=False),patch.object(AlignmentNarrowCore,'_segment_clear',side_effect=[False,True]):
            core._neural_route()
        np.testing.assert_array_equal(core.predicted_path,[[0.,0.],[2.,0.]])
        self.assertFalse(core.motion_route_valid);self.assertFalse(core.alignment_required)
        self.assertEqual(core.narrow_route_state,'route_geometry_or_scan_blocked')

    def test_unchecked_direction_is_discarded(self):
        core=self.core()
        with patch.object(RoomRoutingMixin,'_neural_route'),patch.object(RoomRoutingMixin,'_route_usable',return_value=False),patch.object(AlignmentNarrowCore,'_segment_clear',return_value=False):
            core._neural_route()
        np.testing.assert_array_equal(core.predicted_path,[[0.,0.]])
        self.assertFalse(core.motion_route_valid)

    def test_entrance_hint_contains_sensor_heading_and_center_offset_without_opening_route(self):
        core=self.core();core.goal=np.array([3.,0.]);core.lidar=object()
        core.predicted_path=np.array([[0.,0.]]);original=core.predicted_path.copy()
        core.motion_route_valid=False
        base=np.zeros(235,np.float32);base[216]=.15
        # Both walls support a center 2 cm to the left, at 0.06 rad heading.
        features=np.array([1.,.24,.02/.25,.06/.35,.17,.15])
        with patch.object(FreshPerceptionCore,'_observation',return_value=base),patch('prototypes.narrow_observation.corridor_features',return_value=features):
            obs=core._observation(np.ones(220))
        self.assertEqual(core.policy_target_kind,'lidar_corridor_alignment_hint')
        self.assertGreater(np.arctan2(obs[217],obs[216]),.06)
        np.testing.assert_array_equal(core.predicted_path,original)
        self.assertFalse(core.motion_route_valid)

    def test_side_exit_goal_is_not_overridden_by_corridor_hint(self):
        core=self.core();core.goal=np.array([1.,1.]);core.lidar=object()
        base=np.zeros(235,np.float32);base[216:218]=[.1,.1]
        features=np.array([1.,.24,0.,0.,.16,.16])
        with patch.object(FreshPerceptionCore,'_observation',return_value=base),patch('prototypes.narrow_observation.corridor_features',return_value=features):
            obs=core._observation(np.ones(220))
        np.testing.assert_allclose(obs[216:218],[.1,.1])
        self.assertEqual(core.policy_target_kind,'checked_waypoint')

if __name__=='__main__':unittest.main()
