import unittest
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import numpy as np
from experiments.time_consistent_tracking import TimeConsistentPredictor,TimeConsistentFreshRuntime
from experiments.quality_sensor_runtime import QualitySensorNavigationRuntime
from neural_navigation.quality_prediction_context import contract


class TimeConsistency(unittest.TestCase):
    @staticmethod
    def points(x):return np.array([[x,0],[x,.03],[x,.06],[x,.09]])

    def test_matched_physical_time_independent_of_sample_rate(self):
        final=[]
        for dt in (.2,.4):
            predictor=TimeConsistentPredictor({})
            for stamp in np.arange(0,.80001,dt):
                predictor.update(self.points(.1*stamp),float(stamp))
                self.assertEqual(len(predictor.tracks),1)
                if stamp<.8-1e-9:
                    self.assertLess(predictor.tracks[0]['hits'],3)
                    self.assertEqual(len(predictor.predicted_points),0)
            self.assertGreaterEqual(predictor.tracks[0]['hits'],3)
            final.append(predictor.tracks[0]['velocity'].copy())
        np.testing.assert_allclose(final[0],final[1],atol=1e-12)
        np.testing.assert_allclose(final[0],[.1*(1-.55**2),0],atol=1e-12)

    def test_reversal_requires_new_time_window_and_pose_loss_clears(self):
        predictor=TimeConsistentPredictor({})
        for stamp in (0,.4,.8):predictor.update(self.points(.1*stamp),stamp)
        predictor.update(self.points(.06),1.)
        self.assertLess(predictor.tracks[0]['hits'],3)
        self.assertEqual(predictor.tracks[0]['confirmation_started'],1.)
        predictor.update(self.points(.02),1.4)
        self.assertLess(predictor.tracks[0]['hits'],3)
        predictor.update(self.points(-.02),1.8)
        self.assertGreaterEqual(predictor.tracks[0]['hits'],3)
        self.assertGreater(len(predictor.predicted_points),0)
        predictor.update([],2.,False)
        self.assertEqual(predictor.tracks,[])
        self.assertEqual(len(predictor.predicted_points),0)

    def test_wrong_runtime_and_time_contract_rejected_before_loading_actor(self):
        with TemporaryDirectory() as folder:
            bundle=Path(folder)
            (bundle/'quality_observation.json').write_text(json.dumps(contract()))
            (bundle/'manifest.json').write_text(json.dumps(dict(sensor_core='time-consistent',sensor_core_parameters={})))
            with self.assertRaisesRegex(ValueError,'different sensor core'):
                QualitySensorNavigationRuntime(bundle,None,None,None)
            with self.assertRaisesRegex(ValueError,'contract mismatch'):
                TimeConsistentFreshRuntime(bundle,None,None,None)


if __name__=='__main__':unittest.main()
