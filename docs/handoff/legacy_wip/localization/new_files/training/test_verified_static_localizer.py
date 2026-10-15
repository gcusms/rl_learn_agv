import json
from pathlib import Path
import shutil
import tempfile
import unittest
import numpy as np
from scipy.ndimage import distance_transform_edt
from training.test_fast_static_localizer import WallMap
from neural_navigation.fast_static_localizer import FastStaticLocalizer,temporal_contract
from neural_navigation.verified_static_localizer import VerifiedStaticLocalizer,contract
from neural_navigation.sensors import LidarSpec


class VerifiedStaticTests(unittest.TestCase):
    def setUp(self):
        self.map=WallMap()
        self.lidar=LidarSpec(dict(beams=220,angle_min=-1.1,angle_max=1.1,range_min=.1,range_max=12.,mount=[0.,0.,.12,0.]))
        self.truth=np.array([0.,1.,-np.pi/2]);self.scan=1./np.cos(self.lidar.angles)

    def test_normal_wall_correction_and_unobservable_tangent_preserved(self):
        loc=VerifiedStaticLocalizer(self.map,self.lidar);loc.reset(self.truth+[.1,.06,0.],None)
        for i in range(12):loc.update(np.zeros(3),self.scan,i*.1)
        self.assertAlmostEqual(loc.pose[0],.1,places=6)
        self.assertLess(abs(loc.pose[1]-1.),.008)
        self.assertTrue(loc.scan_update_accepted)

    def test_endpoint_behind_nearer_wall_cannot_be_static_support(self):
        y=self.map.origin[1]+(self.map.height-np.arange(self.map.height)-.5)*self.map.res
        self.map.occupied|=np.repeat(((y<-.5)&(y>=-.55))[:,None],self.map.width,axis=1)
        self.map.distance=distance_transform_edt(~self.map.occupied)*self.map.res
        scan=1.5/np.cos(self.lidar.angles);unchanged=scan.copy();occupied=self.map.occupied.copy()
        old=FastStaticLocalizer(self.map,self.lidar,temporal_contract());old.reset(self.truth,None)
        loc=VerifiedStaticLocalizer(self.map,self.lidar);loc.reset(self.truth,None)
        for i in range(12):
            old.update(np.zeros(3),scan,i*.1);loc.update(np.zeros(3),scan,i*.1)
        self.assertTrue(old.scan_update_accepted)
        self.assertFalse(loc.scan_update_accepted);self.assertEqual(loc.valid_static_hits,0)
        self.assertEqual(loc.ray_excluded_static_hits,60)
        np.testing.assert_allclose(loc.pose,self.truth,atol=1e-12)
        np.testing.assert_array_equal(scan,unchanged);np.testing.assert_array_equal(self.map.occupied,occupied)

    def test_unmapped_near_object_kept_for_avoidance_but_not_fit(self):
        loc=VerifiedStaticLocalizer(self.map,self.lidar);loc.reset(self.truth,None)
        blocked=self.scan.copy();blocked[55:165]=.3;unchanged=blocked.copy()
        for i in range(15):loc.update(np.zeros(3),blocked,i*.1)
        np.testing.assert_array_equal(blocked,unchanged)
        self.assertLess(np.linalg.norm(loc.pose-self.truth),.008)

    def test_rejected_proposal_has_no_information_gain_or_pose_correction(self):
        class Reject(VerifiedStaticLocalizer):
            def _correction_permitted(self,*args):
                self.correction_reason='test_rejected_proposal';return False
        loc=Reject(self.map,self.lidar);loc.reset(self.truth+[0.,.06,0.],None)
        prior=loc.pose.copy();cov=loc.covariance.copy()
        loc.update(np.zeros(3),self.scan,0.)
        np.testing.assert_array_equal(loc.pose,prior);np.testing.assert_array_equal(loc.covariance,cov)
        self.assertEqual(loc.observable_rank,0);self.assertFalse(loc.scan_update_accepted)
        self.assertEqual(len(loc.support_history),0)
        loc.update(np.zeros(3),self.scan,.1)
        self.assertEqual(loc.missing_scan_updates,2)

    def test_packed_rays_match_dense_reference_with_boundaries_and_unknown(self):
        loc=VerifiedStaticLocalizer(self.map,self.lidar)
        rng=np.random.default_rng(4217)
        self.map.occupied[rng.random(self.map.occupied.shape)<.015]=True
        for _ in range(40):
            pose=rng.uniform([-4.,-.8,-np.pi],[4.,3.,np.pi])
            scan=rng.uniform(.1,10.,220)
            # Endpoint boundaries, no-sample rays, invalid hits and 10m cap.
            scan[:9]=[np.nan,np.inf,0.,.1,.18,.205,.23,10.,11.]
            indices=np.arange(220)
            actual,finite=loc.ray_contradictions(pose,scan,indices)
            angles=self.lidar.angles[finite]+pose[2]
            distance=np.arange(0.,scan[finite].max(),self.map.res/2)
            active=distance[None,:]<np.maximum(0.,scan[finite,None]-.18)
            points=pose[:2]+distance[None,:,None]*np.c_[np.cos(angles),np.sin(angles)][:,None,:]
            occupied=self.map.lookup(points.reshape(-1,2),self.map.occupied,outside=False).reshape(active.shape)
            np.testing.assert_array_equal(actual[finite],np.any(occupied&active,axis=1))

    def test_blank_scan_retains_full_planar_odometry(self):
        loc=VerifiedStaticLocalizer(self.map,self.lidar);loc.reset([0.,0.,0.],None)
        scan=np.full(220,np.nan);loc.update([0.,0.,0.],scan,0.)
        pose=loc.update([.1,.05,.02],scan,1.)
        np.testing.assert_allclose(pose,[.1,.05,.02],atol=1e-12)

    def test_core_contract_is_explicit_opt_in(self):
        from navrl.geometry import ROOT
        from navrl.map import StaticMap
        from neural_navigation.core import NavigationCore
        source=ROOT.parent/'models/quality_recovery_20261014/portable'
        with tempfile.TemporaryDirectory() as directory:
            bundle=Path(directory);shutil.copy2(source/'router.pt',bundle/'router.pt')
            (bundle/'localization_contract.json').write_text(json.dumps(contract()))
            m=StaticMap(ROOT/'assets/room_navigation_scripted/decision_map.yaml',radius=.3)
            core=NavigationCore(m,self.lidar,bundle,'shortcut')
            self.assertIsInstance(core.localizer,VerifiedStaticLocalizer)
            core.reset([1.2,1.2,0.],[3.,1.2,1.1],0.)
            core.update(np.full(220,np.nan),np.zeros(3),np.zeros(2),.2)
            self.assertAlmostEqual(core.goal_yaw,1.1)
            self.assertEqual(core.diagnostics()['localization_ray_contradicted_hits'],0)

    def test_native_verifier_parity_and_rejected_information(self):
        from navrl.geometry import ROOT
        bundle=ROOT.parent/'models/localization_mapping_20261015/temporal_native_candidate/portable'
        kernel=bundle/'localization_kernel.so'
        if not kernel.exists():self.skipTest('Prebuilt reference kernel not available')
        from neural_navigation.goal_pose_runtime import check_files
        check_files(bundle,json.loads((bundle/'manifest.json').read_text()))
        python=VerifiedStaticLocalizer(self.map,self.lidar)
        native=VerifiedStaticLocalizer(self.map,self.lidar,contract(True),kernel)
        for loc in (python,native):loc.reset(self.truth+[.1,.06,0.],None)
        for i in range(12):
            for loc in (python,native):loc.update(np.zeros(3),self.scan,i*.1)
            np.testing.assert_allclose(python.pose,native.pose,atol=1e-10)
            np.testing.assert_allclose(python.covariance,native.covariance,atol=1e-10)
        native.reset(self.truth,None);before=native.covariance.copy()
        native._correction_permitted=lambda *args:False
        native.update(np.zeros(3),self.scan,0.)
        self.assertFalse(native.scan_update_accepted)
        np.testing.assert_array_equal(native.covariance,before)


if __name__=='__main__':unittest.main()
