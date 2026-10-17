import unittest

import numpy as np

from experiments.sensor_release_teacher import SensorReleaseTeacher


class SensorReleaseTeacherTests(unittest.TestCase):
    def setUp(self):
        self.x = np.zeros(625, np.float32)
        self.x[216] = .2  # 0.4m waypoint, not a 12m-normalized goal.
        self.x[218] = .3
        self.x[235:244] = 1.
        self.x[267:269] = 1.
        self.x[434] = .6
        self.x[435] = self.x[438] = 1.
        self.x[439:481].reshape(3,14)[:,12:] = [1.,.25]
        self.x[481:553] = 1.
        self.teacher = SensorReleaseTeacher(np.ones(72, bool))

    def test_straight_release_requires_supported_measured_inputs(self):
        row = self.teacher.select(self.x, 0, True)
        self.assertEqual(row['choice'], 1)
        self.assertTrue(row['teacher_controlled'])
        self.assertGreater(row['predicted_waypoint_progress_m'], .1)

    def test_unknown_history_and_scan_never_authorize_motion(self):
        for section in [slice(439,481), slice(481,625)]:
            x = self.x.copy(); x[section] = 0.
            row = self.teacher.select(x, 0, True)
            self.assertEqual(row['choice'], 0)
            self.assertFalse(row['teacher_controlled'])
        x = self.x.copy();x[481:553]=0.;x[553:625]=1.
        self.assertEqual(self.teacher.select(x,0,True)['choice'],0)

    def test_action_mask_contact_and_trainable_support_are_mandatory(self):
        x = self.x.copy();x[235:244]=0.;x[267:269]=0.
        self.assertEqual(self.teacher.select(x,0,True)['choice'],0)
        self.assertEqual(self.teacher.select(self.x,0,False)['choice'],0)
        x=self.x.copy();x[439:481].reshape(3,14)[:,:12]=1.
        self.assertEqual(self.teacher.select(x,0,True)['choice'],0)
        x=self.x.copy();x[435]=0.
        self.assertEqual(self.teacher.select(x,0,True)['choice'],0)

    def test_short_wait_and_model_motion_remain_unchanged(self):
        x=self.x.copy();x[434]=.1
        self.assertEqual(self.teacher.select(x,0,True)['choice'],0)
        self.assertEqual(self.teacher.select(self.x,6,True)['choice'],6)
        self.assertFalse(self.teacher.armed)

    def test_armed_mode_still_stops_on_unknown_and_rechecks_every_action(self):
        self.assertTrue(self.teacher.select(self.x,0,True)['teacher_controlled'])
        x=self.x.copy();x[434]=0.;x[438]=0.
        self.assertEqual(self.teacher.select(x,0,True)['choice'],0)
        x[438]=1.
        self.assertTrue(self.teacher.select(x,0,True)['teacher_controlled'])
        x[235:244]=0.;x[267:269]=0.
        self.assertEqual(self.teacher.select(x,0,True)['choice'],0)
        self.assertFalse(self.teacher.armed)

    def test_goal_behind_can_choose_a_legal_turn(self):
        x=self.x.copy();x[216]=-.2
        self.assertIn(self.teacher.select(x,0,True)['choice'],[4,5])

    def test_invalid_observation_and_illegal_nominal_are_rejected(self):
        x=self.x.copy();x[1]=np.nan
        with self.assertRaises(ValueError):self.teacher.select(x,0,True)
        with self.assertRaises(ValueError):self.teacher.select(self.x[:481],0,True)
        x=self.x.copy();x[235]=0.
        with self.assertRaises(ValueError):self.teacher.select(x,1,True)


if __name__ == '__main__':
    unittest.main()
