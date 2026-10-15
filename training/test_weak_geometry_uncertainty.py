import unittest
import numpy as np
from neural_navigation.weak_geometry_uncertainty import WeakGeometryUncertainty,contract
from neural_navigation.fast_static_localizer import FastStaticLocalizer,temporal_contract
from training.test_fast_static_localizer import WallMap
from neural_navigation.sensors import LidarSpec


class WeakGeometryTests(unittest.TestCase):
    def test_systematic_error_does_not_average_away_with_faster_frames(self):
        results=[]
        for step in (.1,.2):
            budget=WeakGeometryUncertainty(contract())
            for _ in range(round(20/step)):result=budget.update(np.zeros((3,3)),np.eye(3),step,.1*step,0.,0)
            results.append(result)
        np.testing.assert_allclose(results[0],results[1],atol=1e-12)
        self.assertAlmostEqual(results[0][0],.24)

    def test_observed_axes_can_recover_without_erasing_weak_axis_history(self):
        budget=WeakGeometryUncertainty(contract())
        budget.update(np.zeros((3,3)),np.eye(3),30.,3.,.5,0)
        result=budget.update(np.diag([0.,1.,1.]),np.eye(3),.2,0.,0.,60)
        self.assertGreater(result[0],.35);np.testing.assert_allclose(result[1:],0.,atol=1e-12)
        np.testing.assert_allclose(budget.update(np.eye(3),np.eye(3),.2,0.,0.,60),0.,atol=1e-12)

    def test_uncertainty_opt_in_preserves_pose_and_blank_scans_grow_budget(self):
        lidar=LidarSpec(dict(beams=220,angle_min=-1.1,angle_max=1.1,range_min=.1,range_max=10.,mount=[0.,0.,.12,0.]))
        m=WallMap();old=FastStaticLocalizer(m,lidar,temporal_contract())
        new=FastStaticLocalizer(m,lidar,temporal_contract(),uncertainty=contract())
        for loc in (old,new):loc.reset([0.,1.,-np.pi/2],None)
        for i in range(101):
            odom=np.array([i*.02,0.,0.]);scan=np.full(220,np.nan)
            np.testing.assert_array_equal(old.update(odom,scan,i*.2),new.update(odom,scan,i*.2))
        self.assertTrue(np.all(new.std>=old.std));self.assertGreater(max(new.std[:2]),.2)
        before=new.std.copy();new.update(odom,scan,20.)
        np.testing.assert_array_equal(new.std,before)
        new.reset([0.,1.,0.],None);np.testing.assert_array_equal(new.weak_geometry_uncertainty.floor,np.zeros(3))


if __name__=='__main__':unittest.main()
