import unittest
from types import SimpleNamespace
import numpy as np
from neural_navigation.predictive_risk import PredictiveRiskContext,BoundedPoseRiskContext
from neural_navigation.safety_context import contract,validate_contract,BOUNDED_PREDICTIVE_SCHEMA


class BoundedPoseRiskTests(unittest.TestCase):
    def core(self):
        return SimpleNamespace(elapsed=1.,estimate=np.zeros(3),odom=np.zeros(3),
            odometry_sanity_failure=False,
            localizer=SimpleNamespace(std=np.array([.05,.05,.03]),
                map_agreement=.9,scan_update_accepted=True),
            laser_fusion=SimpleNamespace(predictor=SimpleNamespace(tracks=[
                dict(time=1.,hits=4,velocity=np.array([-.2,0.]),
                    points=np.array([[.6,0.],[.6,.02]]))])))

    def test_current_support_preserves_original_risk(self):
        core=self.core()
        np.testing.assert_array_equal(BoundedPoseRiskContext([.4,.2]).features(core),
            PredictiveRiskContext([.4,.2]).features(core))
        validate_contract(contract(BOUNDED_PREDICTIVE_SCHEMA))
        bad=contract(BOUNDED_PREDICTIVE_SCHEMA)
        bad['predictive_features']['risk']['pose_support_hold_s']=5
        with self.assertRaises(ValueError):validate_contract(bad)

    def test_short_support_flicker_keeps_warning_but_expires(self):
        core=self.core();ctx=BoundedPoseRiskContext([.4,.2]);ctx.features(core)
        core.elapsed=1.2;core.localizer.map_agreement=.79
        self.assertEqual(ctx.features(core)[-1],1)
        self.assertEqual(PredictiveRiskContext([.4,.2]).features(core)[-1],0)
        core.elapsed=1.61
        np.testing.assert_array_equal(ctx.features(core),np.zeros(13))

    def test_low_agreement_without_history_stays_unknown(self):
        core=self.core();core.localizer.map_agreement=.79
        np.testing.assert_array_equal(BoundedPoseRiskContext([.4,.2]).features(core),np.zeros(13))

    def test_pose_jump_uncertainty_and_map_loss_invalidate_history(self):
        for fault in ('jump','uncertain','map','odom'):
            core=self.core();ctx=BoundedPoseRiskContext([.4,.2]);ctx.features(core)
            core.elapsed=1.2;core.localizer.map_agreement=.79
            if fault=='jump':core.estimate[0]=.11
            if fault=='uncertain':core.localizer.std[0]=.21
            if fault=='map':core.localizer.map_agreement=.69
            if fault=='odom':core.odometry_sanity_failure=True
            np.testing.assert_array_equal(ctx.features(core),np.zeros(13))
            self.assertIsNone(ctx.anchor)

    def test_odometry_propagation_does_not_count_as_pose_jump(self):
        core=self.core();ctx=BoundedPoseRiskContext([.4,.2]);ctx.features(core)
        core.elapsed=1.2;core.localizer.map_agreement=.79
        core.estimate[0]=core.odom[0]=.12
        self.assertEqual(ctx.features(core)[-1],1)

    def test_missing_scan_retains_only_bounded_warning(self):
        core=self.core();ctx=BoundedPoseRiskContext([.4,.2]);ctx.features(core)
        core.elapsed=1.2;core.localizer.scan_update_accepted=False;core.localizer.map_agreement=0
        self.assertEqual(ctx.features(core)[-1],1)
        core.elapsed=1.7
        self.assertEqual(ctx.features(core)[-1],0)


if __name__=='__main__':unittest.main()
