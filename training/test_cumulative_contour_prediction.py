import copy
import unittest
import numpy as np
from experiments.cumulative_contour_prediction import CumulativeContourPrediction


def track(time,count,diameter):
    return dict(id=1,time=time,hits=3+int(time*5),center=np.array([1.,0.]),velocity=np.array([.1,0.]),
        points=np.c_[np.linspace(-diameter/2,diameter/2,count)+1,np.zeros(count)])


class CumulativeContourTests(unittest.TestCase):
    def test_gradual_shrink_hold_unknown_and_measurement_recovery(self):
        predictor=CumulativeContourPrediction()
        predictor.update([track(0,40,.426)],0)
        predictor.update([track(.4,36,.36)],.4)
        raw=track(.8,30,.279);before=copy.deepcopy(raw);result=predictor.update([raw],.8)[0]
        self.assertTrue(result['degraded']);self.assertEqual(result['source'],'held')
        np.testing.assert_array_equal(raw['points'],before['points']);np.testing.assert_array_equal(raw['velocity'],before['velocity'])
        self.assertEqual(predictor.update([raw],1.2)[0]['source'],'unknown')
        self.assertEqual(predictor.update([track(1.4,40,.426)],1.4)[0]['source'],'unknown')
        self.assertEqual(predictor.update([track(1.4,40,.426)],1.5)[0]['source'],'unknown')
        self.assertEqual(predictor.update([track(1.6,40,.426)],1.6)[0]['source'],'observed')

    def test_expiry_preserves_age_and_never_revives_missing_measurements(self):
        predictor=CumulativeContourPrediction();predictor.update([track(0,40,.426)],0)
        retained=predictor.update([],1.)[0]
        self.assertEqual(retained['source'],'unknown');self.assertEqual(retained['time'],0)
        self.assertEqual(retained['age'],1.);self.assertTrue(retained['retained_without_measurement'])
        self.assertEqual(predictor.update([],1.3),[])
        with self.assertRaises(ValueError):predictor.update([],1.2)


if __name__=='__main__':unittest.main()
