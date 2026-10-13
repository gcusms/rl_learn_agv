import json
import unittest
import numpy as np
import torch
from navrl.geometry import ROOT
from experiments.recovery_curriculum import RecoveryNavigationCurriculum
from experiments.fixed_training_replay import reset_training_episode


class FixedTrainingReplayTests(unittest.TestCase):
    def test_original_training_reset_and_sensor_transitions_recreated_exactly(self):
        torch.set_num_threads(1)
        components = ROOT/'runs/safety_context_20261011/recovery_probes_v3_16000/candidate'
        env = RecoveryNavigationCurriculum(components, stage='robust', motion_profiles=True,
            footprint_clearance_cost=1.)
        actions = [np.array([.5, .2]), np.array([-.05, .3]), np.array([.2, -.4])]*3
        try:
            original, _ = env.reset(seed=92115)
            case = json.loads(env.training_case_json)
            expected = []
            for action in actions:
                obs, reward, done, truncated, info = env.step(action)
                expected.append((obs.copy(), env.active.truth.copy(), reward,
                    info['commanded_velocity'].copy(), info['execution_guard_reason'], done, truncated))
            restored, _ = reset_training_episode(env, case)
            np.testing.assert_array_equal(original, restored)
            for action, before in zip(actions, expected):
                obs, reward, done, truncated, info = env.step(action)
                np.testing.assert_array_equal(before[0], obs)
                np.testing.assert_array_equal(before[1], env.active.truth)
                self.assertEqual(before[2], reward)
                np.testing.assert_array_equal(before[3], info['commanded_velocity'])
                self.assertEqual(before[4:], (info['execution_guard_reason'], done, truncated))
            bad = dict(case, scene='/tmp/not-a-training-layout')
            with self.assertRaises(ValueError): reset_training_episode(env, bad)
        finally:
            env.close()


if __name__ == '__main__': unittest.main()
