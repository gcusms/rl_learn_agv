import unittest
from types import SimpleNamespace
import numpy as np
from scipy.spatial import cKDTree
from scipy.ndimage import distance_transform_edt
from navrl.map import StaticMap
from prototypes.surface_localizer import SurfaceLocalizer,map_surfaces


class SurfaceLocalizationTests(unittest.TestCase):
    def localizer(self,nominal=(10.,1.,0.)):
        m=object.__new__(StaticMap);m.res=.02;m.origin=np.zeros(3);m.height=100;m.width=1500
        m.occupied=np.zeros((100,1500),bool);m.occupied[:10]=True;m.occupied[-10:]=True
        m.blocked=m.occupied.copy();m.distance=distance_transform_edt(~m.occupied)*m.res
        m.lookup=m.lookup;m.clearance=m.distance
        lidar=SimpleNamespace(angles=np.linspace(-1.9,1.9,220),mount=np.array([.118389,0.,0.,0.]),range_min=.1,range_max=8.)
        loc=object.__new__(SurfaceLocalizer);loc.map=m;loc.lidar=lidar
        loc.surfaces,loc.normals=map_surfaces(m);loc.tree=cKDTree(loc.surfaces)
        loc.reset(nominal,None);return loc

    def scan(self,loc,seed=3):
        scan=.8/np.maximum(np.abs(np.sin(loc.lidar.angles)),1e-6)
        scan[scan>=loc.lidar.range_max]=np.inf
        return scan+np.random.default_rng(seed).normal(0,.015,len(scan))

    def test_side_walls_correct_lateral_yaw_without_inventing_longitudinal_motion(self):
        loc=self.localizer((10.,1.035,.025))
        for i in range(50):loc.update(np.zeros(3),self.scan(loc,i))
        self.assertTrue(loc.scan_update_accepted);self.assertEqual(loc.observable_rank,2)
        self.assertAlmostEqual(loc.pose[0],10.,places=6)
        self.assertLess(abs(loc.pose[1]-1.),.007);self.assertLess(abs(loc.pose[2]),.005)
        self.assertGreater(loc.std[0],.12)

    def test_lost_scan_preserves_odometry_and_increases_uncertainty(self):
        loc=self.localizer();loc.update(np.zeros(3),self.scan(loc));before=loc.std.copy();pose=loc.pose.copy()
        after=loc.update([.03,0.,.01],np.full(220,np.nan))
        self.assertFalse(loc.scan_update_accepted);self.assertEqual(loc.missing_scan_updates,1)
        self.assertGreater(after[0],pose[0]+.029);self.assertTrue((loc.std>before).all())

    def test_sparse_or_unmatched_scan_does_not_claim_valid_localization(self):
        loc=self.localizer();scan=np.full(220,np.nan);scan[:5]=1.
        loc.update(np.zeros(3),scan);self.assertFalse(loc.scan_update_accepted)
        loc.update(np.zeros(3),np.full(220,.15));self.assertFalse(loc.scan_update_accepted)
        np.testing.assert_allclose(loc.pose,[10.,1.,0.])

    def test_unmapped_foreground_returns_are_rejected_from_wall_fit(self):
        loc=self.localizer((10.,1.03,.02));scan=self.scan(loc);scan[::5]=.2
        loc.update(np.zeros(3),scan)
        self.assertTrue(loc.scan_update_accepted)
        self.assertLess(abs(loc.pose[1]-1.),.01);self.assertLess(abs(loc.pose[2]),.01)

    def test_unequal_wall_ray_counts_do_not_pull_pose_toward_one_raster_edge(self):
        loc=self.localizer((10.,1.02,0.))
        loc.lidar.angles=np.r_[np.linspace(.2,1.9,190),np.linspace(-1.9,-.2,30)]
        scan=.8/np.abs(np.sin(loc.lidar.angles))
        for _ in range(3):loc.update(np.zeros(3),scan)
        self.assertTrue(loc.scan_update_accepted)
        self.assertLess(abs(loc.pose[1]-1.),.002)
        self.assertLess(abs(loc.pose[2]),.002)

if __name__=='__main__':unittest.main()
