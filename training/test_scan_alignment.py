import unittest
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock,patch
import numpy as np
from neural_navigation.sensors import LidarSpec,OdometryBuffer
from neural_navigation.scan_alignment import deskew_endpoints,NativeScanAdapter


class ScanAlignmentTests(unittest.TestCase):
    def test_vectorized_odometry_matches_single_queries_across_yaw_wrap(self):
        buffer=OdometryBuffer();buffer.append(1.,[0,0,np.pi-.04],[.2,.4]);buffer.append(1.1,[.02,0,-np.pi+.04],[.2,.4])
        times=np.array([1.,1.03,1.08,1.1,1.13]);poses,speeds=buffer.sample_many(times)
        for i,t in enumerate(times):
            pose,velocity=buffer.sample(t);np.testing.assert_allclose(poses[i],pose,atol=1e-12);np.testing.assert_allclose(speeds[i],velocity)
        self.assertIsNone(buffer.sample_many([.9,1.]));self.assertIsNone(buffer.sample_many([1.17]))
        self.assertIsNone(buffer.sample_many([1.05],max_gap=.05))

    def test_deskew_landmark_with_translation_rotation_and_sensor_mount(self):
        poses=np.c_[np.linspace(0,.02,20),np.linspace(0,.005,20),np.linspace(.1,.15,20)]
        mount=np.array([.1,-.05,.2,.3]);landmark=np.array([3.,1.])
        c,s=np.cos(poses[:,2]),np.sin(poses[:,2]);origin=poses[:,:2]+np.c_[c*mount[0]-s*mount[1],s*mount[0]+c*mount[1]]
        delta=landmark-origin;ranges=np.linalg.norm(delta,axis=1);angles=np.arctan2(delta[:,1],delta[:,0])-poses[:,2]-mount[3]
        actual=deskew_endpoints(ranges,angles,poses,poses[-1],mount)
        np.testing.assert_allclose(actual,np.repeat(actual[-1:],len(poses),axis=0),atol=1e-12)
        uncompensated=np.c_[ranges*np.cos(angles),ranges*np.sin(angles)]
        self.assertGreater(np.linalg.norm(uncompensated[0]-actual[0]),.1)

    def fixture(self):
        spec=LidarSpec(dict(beams=61,angle_min=-.6,angle_max=.6,range_min=.1,range_max=12.,mount=[0,0,.2,0]))
        b=OdometryBuffer();b.append(0.,[0,0,0],[.2,0]);b.append(.1,[.02,0,0],[.2,0])
        return spec,b

    def test_rolling_wall_scan_is_aligned_to_last_beam_time(self):
        spec,b=self.fixture();adapter=NativeScanAdapter(spec,b)
        angles=np.linspace(-.7,.7,801);time=np.linspace(0,.1,801);ranges=(3.-.2*time)/np.cos(angles)
        frame=adapter.prepare(ranges,angles[0],angles[1]-angles[0],0.,.1/800,.1,12.)
        self.assertTrue(frame.valid);self.assertTrue(frame.compensated);self.assertAlmostEqual(frame.timestamp,.1)
        np.testing.assert_allclose(frame.pose,[.02,0,0])
        corrected_error=np.abs(frame.scan*np.cos(spec.angles)-2.98).mean()
        raw,_=spec.resample(ranges,angles[0],angles[1]-angles[0],.1,12.)
        raw_error=np.abs(raw*np.cos(spec.angles)-2.98).mean()
        self.assertLess(corrected_error,raw_error);self.assertLess(corrected_error,.007)

    def test_stationary_scan_matches_existing_resampling_exactly(self):
        spec,_=self.fixture();b=OdometryBuffer();b.append(0.,[0,0,0],[0,0]);b.append(.1,[0,0,0],[0,0])
        angles=np.linspace(-.7,.7,801);ranges=3/np.cos(angles);ranges[10]=np.nan;ranges[30]=np.inf
        expected,valid=spec.resample(ranges,angles[0],angles[1]-angles[0],.1,12.)
        frame=NativeScanAdapter(spec,b).prepare(ranges,angles[0],angles[1]-angles[0],0.,.1/800,.1,12.)
        np.testing.assert_array_equal(frame.scan,expected);self.assertEqual(frame.valid,valid);self.assertFalse(frame.compensated)

    def test_missing_pose_and_angular_gaps_do_not_create_free_space(self):
        spec,b=self.fixture();adapter=NativeScanAdapter(spec,b)
        frame=adapter.prepare(np.ones(61),-.6,.02,1.,.1/60,.1,12.)
        self.assertFalse(frame.valid);self.assertEqual(frame.reason,'acquisition_odometry_unavailable');self.assertTrue(np.isnan(frame.scan).all())
        frame=adapter.prepare(np.full(61,np.nan),-.6,.02,0.,.1/60,.1,12.)
        self.assertTrue(np.isnan(frame.scan).all())
        frame=adapter.prepare(np.ones(8),0.,.02,0.,0.,.1,12.)
        self.assertFalse(frame.valid);self.assertTrue(np.isnan(frame.scan[spec.angles<-.1]).all())

    def test_reader_rejection_records_emitted_stop_in_action_history(self):
        import run_sensor_navigation
        spec,_=self.fixture();core=SimpleNamespace(lidar=spec,record_action=Mock())
        runtime=SimpleNamespace(core=core,guard=SimpleNamespace(reverse_started=2.),step=Mock())
        with tempfile.TemporaryDirectory() as directory:
            source=Path(directory)/'input.jsonl';output=Path(directory)/'commands.jsonl'
            source.write_text(json.dumps(dict(odometry_samples=[],native_scan=dict(ranges=[2.]*61,angle_min=-.6,angle_increment=.02,
                                                                                   timestamp=1.,time_increment=.1/60,range_min=.1,range_max=12.)))+'\n')
            argv=['reader','--bundle',directory,'--map','unused','--initial-pose','0','0','0','--goal','1','0','--input',str(source),'--output',str(output),'--native-scan']
            with patch('sys.argv',argv),patch.object(run_sensor_navigation,'SensorNavigationRuntime',return_value=runtime):run_sensor_navigation.main()
            result=json.loads(output.read_text());self.assertEqual(result['command'],[0.,0.]);self.assertTrue(result['safety_override'])
            self.assertIn('acquisition_odometry_unavailable',result['error']);runtime.step.assert_not_called()
            np.testing.assert_array_equal(core.record_action.call_args.args[0],[0,0]);self.assertIsNone(runtime.guard.reverse_started)


if __name__=='__main__':unittest.main(verbosity=2)
