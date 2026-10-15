"""Geometry/feedback invariants, without policy, odometry truth or ROS nodes."""
import unittest
import numpy as np
from scipy.ndimage import distance_transform_edt
from neural_navigation.fast_static_localizer import FastStaticLocalizer,sparse_contract,temporal_contract
from neural_navigation.sensors import LidarSpec


class WallMap:
    # Only a horizontal physical wall y=[-.05,0]. No enclosing fake walls.
    def __init__(self):
        self.res=.025;self.origin=np.array([-5.,-1.]);self.width=400;self.height=200
        centers=self.origin[1]+(self.height-np.arange(self.height)-.5)*self.res
        self.occupied=np.repeat(((centers<0)&(centers>=-.05))[:,None],self.width,axis=1)
        self.unknown=np.zeros_like(self.occupied);self.distance=distance_transform_edt(~self.occupied)*self.res
    def lookup(self,points,field,outside=0.):
        points=np.asarray(points);col=np.floor((points[:,0]-self.origin[0])/self.res).astype(int)
        row=self.height-1-np.floor((points[:,1]-self.origin[1])/self.res).astype(int)
        valid=(col>=0)&(col<self.width)&(row>=0)&(row<self.height)
        result=np.full(len(points),outside,dtype=field.dtype);result[valid]=field[row[valid],col[valid]];return result


class FastStaticTests(unittest.TestCase):
    def setUp(self):
        self.lidar=LidarSpec(dict(beams=220,angle_min=-1.1,angle_max=1.1,range_min=.1,range_max=10.,mount=[0.,0.,.12,0.]))
        self.map=WallMap();self.loc=FastStaticLocalizer(self.map,self.lidar)
        self.truth=np.array([0.,1.,-np.pi/2]);self.scan=1./np.cos(self.lidar.angles)
    def test_wall_corrects_normal_but_not_tangent(self):
        self.loc.reset(self.truth+[.1,.06,0.],None)
        for i in range(12):self.loc.update([0.,0.,0.],self.scan,i*.1)
        self.assertAlmostEqual(self.loc.pose[0],.1,places=6)
        self.assertLess(abs(self.loc.pose[1]-1.),.008)
        self.assertGreater(self.loc.std[0],self.loc.std[1])
    def test_unmapped_returns_do_not_drag_pose(self):
        self.loc.reset(self.truth,None);blocked=self.scan.copy();blocked[55:165]=.3
        original=blocked.copy()
        for i in range(20):self.loc.update([0.,0.,0.],blocked,i*.1)
        self.assertLess(np.linalg.norm(self.loc.pose-self.truth),.008)
        self.assertLess(self.loc.static_support_fraction,.65)
        np.testing.assert_array_equal(blocked,original)
    def test_missing_scan_grows_uncertainty_and_keeps_lateral_feedback(self):
        self.loc.reset([0.,0.,0.],None);empty=np.full(220,np.nan)
        self.loc.update([0.,0.,0.],empty,0.);before=self.loc.std.copy()
        pose=self.loc.update([.1,.05,.02],empty,1.)
        np.testing.assert_allclose(pose,[.1,.05,.02],atol=1e-12)
        self.assertTrue(np.all(self.loc.std>before));self.assertFalse(self.loc.scan_update_accepted)
    def test_duplicate_scan_cannot_repeat_information_or_correction(self):
        self.loc.reset(self.truth+[0.,.06,0.],None);self.loc.update([0.,0.,0.],self.scan,0.)
        pose=self.loc.pose.copy();cov=self.loc.covariance.copy()
        for _ in range(20):self.loc.update([0.,0.,0.],self.scan,0.)
        np.testing.assert_array_equal(self.loc.pose,pose);np.testing.assert_array_equal(self.loc.covariance,cov)
        with self.assertRaises(ValueError):self.loc.update([0.,0.,0.],self.scan,-.1)
        with self.assertRaises(ValueError):self.loc.update([.1,0.,0.],self.scan,0.)
    def test_unknown_endpoint_is_not_static_evidence(self):
        self.map.unknown[:]=True;self.loc.reset(self.truth,None)
        self.loc.update([0.,0.,0.],self.scan,0.)
        self.assertEqual(self.loc.static_support_fraction,0.);self.assertFalse(self.loc.scan_update_accepted)
    def test_reset_clears_history(self):
        self.loc.update([0.,0.,0.],self.scan,1.);self.loc.reset(self.truth,None)
        self.assertIsNone(self.loc.last_odom);self.assertIsNone(self.loc.timestamp)
        self.assertEqual(self.loc.missing_scan_updates,0)

    def test_sparse_wall_corrects_only_observed_direction(self):
        sparse=np.full(220,np.nan);sparse[self.loc.indices[[10,20,40,50]]]=self.scan[self.loc.indices[[10,20,40,50]]]
        improved=FastStaticLocalizer(self.map,self.lidar,sparse_contract())
        for loc in (self.loc,improved):
            loc.reset(self.truth+[.1,.06,0.],None)
            for i in range(12):loc.update(np.zeros(3),sparse,i*.1)
        self.assertFalse(self.loc.scan_update_accepted)
        self.assertTrue(improved.scan_update_accepted)
        self.assertEqual(improved.valid_static_hits,4)
        self.assertAlmostEqual(improved.pose[0],.1,places=6)
        self.assertLess(abs(improved.pose[1]-1.),.008)
        # Four matching endpoints must not gain a full scan's confidence.
        self.assertGreater(improved.std[1],self.loc.config['covariance_floor'][1])

    def test_sparse_contract_does_not_force_fit_or_keep_stale_geometry_rank(self):
        loc=FastStaticLocalizer(self.map,self.lidar,sparse_contract());loc.reset(self.truth,None)
        loc.update(np.zeros(3),self.scan,0.);self.assertGreater(loc.observable_rank,0)
        missing=np.full(220,np.nan)
        missing[loc.indices[:2]]=self.scan[loc.indices[:2]]
        loc.update(np.zeros(3),missing,.1)
        self.assertFalse(loc.scan_update_accepted);self.assertEqual(loc.observable_rank,0)
        np.testing.assert_array_equal(loc.observable_eigenvalues,np.zeros(3))

    def test_temporal_support_needs_current_ray_and_physical_history(self):
        loc=FastStaticLocalizer(self.map,self.lidar,temporal_contract())
        loc.reset(self.truth+[0.,.06,0.],None)
        sparse=np.full(220,np.nan);sparse[loc.indices[30]]=self.scan[loc.indices[30]]
        for timestamp in [0.,.2]:
            loc.update(np.zeros(3),sparse,timestamp);self.assertFalse(loc.scan_update_accepted)
        loc.update(np.zeros(3),sparse,.4);self.assertTrue(loc.scan_update_accepted)
        self.assertEqual(loc.information_frames,3)
        self.assertLess(abs(loc.pose[1]-1.),.05)
        before=loc.covariance.copy();loc.update(np.zeros(3),np.full(220,np.nan),.6)
        self.assertFalse(loc.scan_update_accepted)
        self.assertTrue(np.all(np.diag(loc.covariance)>=np.diag(before)))
        # A newly occluding object cannot validate historical wall returns.
        occluded=sparse.copy();occluded[loc.indices[30]]=.3
        loc.update(np.zeros(3),occluded,.8);self.assertFalse(loc.scan_update_accepted)
        loc.reset(self.truth,None);self.assertEqual(len(loc.support_history),0)

    def test_temporal_history_rejects_current_free_space_contradiction(self):
        loc=FastStaticLocalizer(self.map,self.lidar,temporal_contract());loc.reset(self.truth,None)
        for t in [0.,.2,.4]:loc.update(np.zeros(3),self.scan,t)
        changed=self.scan+1.
        # This is a direct support check: all old endpoints are contradicted
        # by the current finite rays; only the provided current point remains.
        points=loc.directions[[30]]*self.scan[loc.indices[[30]],None]
        loc.information_frames=1
        supported=loc._temporal_support(points,np.zeros(3),changed,.6,loc.pose)
        self.assertEqual(len(supported),1);self.assertEqual(loc.information_frames,1)

    def test_temporal_history_compensates_measured_lateral_and_yaw_motion(self):
        loc=FastStaticLocalizer(self.map,self.lidar,temporal_contract());loc.reset(self.truth,None)
        selected=np.array([10,30,50]);points=loc.directions[selected]*self.scan[loc.indices[selected],None]
        old_odom=np.array([10.,-2.,0.]);odom=np.array([10.1,-1.95,.1])
        c,s=np.cos(odom[2]),np.sin(odom[2]);rotation=np.array([[c,-s],[s,c]])
        previous_points=points@rotation.T+odom[:2]-old_odom[:2]
        loc.support_history.append((0.,old_odom,previous_points))
        supported=loc._temporal_support(points,odom,self.scan,.2,self.truth)
        self.assertEqual(loc.information_frames,2)
        np.testing.assert_allclose(supported[3:],points,atol=1e-12)

    def test_opt_in_core_does_not_require_neural_localizer_checkpoint(self):
        import json,tempfile,shutil
        from pathlib import Path
        from navrl.geometry import ROOT
        from navrl.map import StaticMap
        from neural_navigation.core import NavigationCore
        from neural_navigation.fast_static_localizer import contract
        source=ROOT.parent/'models/quality_recovery_20261014/portable'
        with tempfile.TemporaryDirectory() as directory:
            bundle=Path(directory);shutil.copy2(source/'router.pt',bundle/'router.pt')
            (bundle/'localization_contract.json').write_text(json.dumps(contract()))
            m=StaticMap(ROOT/'assets/room_navigation_scripted/decision_map.yaml',radius=.3)
            core=NavigationCore(m,self.lidar,bundle,'shortcut')
            self.assertIsInstance(core.localizer,FastStaticLocalizer)
            core.reset([1.2,1.2,0.],[3.,1.2],0.)
            core.update(np.full(220,np.nan),np.zeros(3),np.zeros(2),.2)
            self.assertFalse(core.localizer.scan_update_accepted)

    def test_runtime_rejects_unhashed_optional_native_component(self):
        import hashlib,tempfile
        from pathlib import Path
        from neural_navigation.goal_pose_runtime import check_files
        with tempfile.TemporaryDirectory() as directory:
            bundle=Path(directory);kernel=bundle/'localization_kernel.so';kernel.write_bytes(b'test kernel')
            with self.assertRaises(ValueError):check_files(bundle,{'files':{}})
            manifest={'files':{'localization_kernel.so':hashlib.sha256(kernel.read_bytes()).hexdigest()}}
            check_files(bundle,manifest);kernel.write_bytes(b'tampered kernel')
            with self.assertRaises(ValueError):check_files(bundle,manifest)


if __name__=='__main__':unittest.main()
