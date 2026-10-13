import json
import tempfile
import unittest
from pathlib import Path
from experiments.calibrate_track_residuals import audit


def row(time,stamp,center,hits=3):
    obs=[0.]*282;obs[281]=1.
    return dict(timestamp=time,estimate=[0.,0.,0.],policy_observation=obs,
        sensor_diagnostic=dict(tracks=[dict(id=1,time=stamp,center=[center,0.],
            velocity=[.1,0.],hits=hits,points=[[center,0.]])]))


class TrackForecastCalibrationTests(unittest.TestCase):
    def run_audit(self,rows):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/'trace.json';p.write_text(json.dumps(dict(samples=rows)))
            return audit(p)

    def test_future_measurement_cannot_inflate_its_own_forecast_bound(self):
        a=self.run_audit([row(0,0,1),row(.4,.4,1.24,4)])
        b=self.run_audit([row(0,0,1),row(.4,.4,2.24,4)])
        self.assertEqual(a['all_pairs']['measurement_pairs'],1)
        self.assertTrue(a['samples'][0]['directional_exceeded'])
        self.assertEqual(a['samples'][0]['past_residual_samples'],0)
        self.assertAlmostEqual(a['samples'][0]['directional_bound_m'],.05)
        self.assertEqual(a['samples'][0]['directional_bound_m'],b['samples'][0]['directional_bound_m'])

    def test_duplicate_scan_does_not_count_as_new_measurement(self):
        a=self.run_audit([row(0,0,1),row(.2,0,1),row(.4,.4,1.04,4)])
        self.assertEqual(a['all_pairs']['measurement_pairs'],1)
        self.assertFalse(a['samples'][0]['directional_exceeded'])

    def test_reconfirmation_and_unsupported_pose_are_excluded(self):
        a=self.run_audit([row(0,0,1,4),row(.4,.4,1.04,1)])
        self.assertEqual(a['all_pairs']['measurement_pairs'],0)
        last=row(.4,.4,1.04,4);last['policy_observation'][281]=0
        b=self.run_audit([row(0,0,1),last])
        self.assertEqual(b['all_pairs']['measurement_pairs'],0)


if __name__=='__main__':unittest.main()
