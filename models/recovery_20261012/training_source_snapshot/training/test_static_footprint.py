import unittest
from types import SimpleNamespace
import numpy as np
from neural_navigation.static_footprint import static_footprint_clear
from neural_navigation.runtime import EmergencyStop


class StaticFootprintTests(unittest.TestCase):
    def scene(self):
        return SimpleNamespace(origin=np.zeros(3), width=100, height=100,
            res=.02, blocked=np.zeros((100, 100), bool))

    def test_orientation_changes_collision_without_changing_center(self):
        scene = self.scene()
        scene.blocked[49, 64] = True  # Cell center (1.29,1.01).
        self.assertFalse(static_footprint_clear(scene, [1,1,0], [.6,.2], 0.))
        self.assertTrue(static_footprint_clear(scene, [1,1,np.pi/2], [.6,.2], 0.))

    def test_cell_corner_intersection_and_margin_are_conservative(self):
        scene = self.scene(); scene.blocked[43, 58] = True
        self.assertFalse(static_footprint_clear(scene, [1,1,np.pi/4], [.4,.1], 0.))
        scene = self.scene(); scene.blocked[49, 61] = True
        self.assertTrue(static_footprint_clear(scene, [1,1,0], [.4,.2], 0.))
        self.assertFalse(static_footprint_clear(scene, [1,1,0], [.4,.2], .04))

    def test_unknown_cells_and_outside_remain_blocked(self):
        scene = self.scene(); scene.blocked[50,50] = True
        self.assertFalse(static_footprint_clear(scene, [1,1,0], [.4,.2]))
        self.assertFalse(static_footprint_clear(self.scene(), [.05,1,0], [.4,.2]))

    def test_versioned_guard_checks_path_and_preserves_legacy(self):
        scene=self.scene(); scene.free=np.zeros_like(scene.blocked)
        scene.lookup=lambda *args, **kwargs: False
        core=SimpleNamespace(static_map=scene,localizer=SimpleNamespace(std=np.zeros(3)),
            estimate=np.array([1.,1.,0.]))
        legacy=EmergencyStop(footprint=[.4,.2])
        new=EmergencyStop(footprint=[.4,.2],memory_config={'static_footprint_schema':'oriented_rectangle_v1'})
        self.assertFalse(legacy._static_pose_clear(core,core.estimate,[.2,0.],1.))
        self.assertIsNone(new._trajectory_check(core,np.array([.2,0.]),np.zeros((0,2)),np.zeros(0),0.))
        scene.blocked[49,68]=True  # Wall ahead; current body clear, braking sweep blocked.
        self.assertEqual(new._trajectory_check(core,np.array([.2,0.]),np.zeros((0,2)),np.zeros(0),0.),
            'static_braking_footprint')

    def test_explicit_pose_uncertainty_can_veto_nominal_clearance(self):
        scene=self.scene(); scene.blocked[49,64]=True
        core=SimpleNamespace(static_map=scene,localizer=SimpleNamespace(std=np.array([.1,.1,.01])))
        nominal=EmergencyStop(footprint=[.4,.2],memory_config={'static_footprint_schema':'oriented_rectangle_v1'})
        uncertain=EmergencyStop(footprint=[.4,.2],memory_config={
            'static_footprint_schema':'oriented_rectangle_v1','static_footprint_pose_sigma_scale':1.})
        self.assertTrue(nominal._static_pose_clear(core,np.array([1.,1.,0.]),[0.,0.],.2))
        self.assertFalse(uncertain._static_pose_clear(core,np.array([1.,1.,0.]),[0.,0.],.2))


if __name__ == '__main__': unittest.main()
