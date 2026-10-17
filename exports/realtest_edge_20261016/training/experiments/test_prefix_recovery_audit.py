import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from experiments.audit_prefix_recovery_rollouts import History, initialize_history
from experiments.recovery_demonstrations import convert_observations
from neural_navigation.decision_feedback import FEEDBACK_REASONS, REASON_SLICE, STAT_SLICE
from navrl.geometry import ROOT


class PrefixHistoryTests(unittest.TestCase):
    def test_actual_guard_stop_counts_as_waiting_when_motion_exists(self):
        h = History()
        obs = np.zeros(436, np.float32)
        obs[300+REASON_SLICE.start] = 1.
        mask = np.zeros(12, bool)
        mask[[0, 1]] = True
        h.check(obs, 0.)
        h.record(0., 0, mask, [0., 0.], 'laser_braking_footprint')
        obs[300+REASON_SLICE.start:300+REASON_SLICE.stop] = 0.
        obs[300+REASON_SLICE.start+FEEDBACK_REASONS.index('laser_braking_footprint')] = 1.
        obs[300+STAT_SLICE.start+2:300+STAT_SLICE.start+4] = .04
        h.check(obs, .2)
        obs[300+STAT_SLICE.start+3] = 0.
        with self.assertRaises(ValueError):
            h.check(obs, .2)

    def test_recorded_prefix_primes_reason_and_timer_but_not_ppo_step_count(self):
        source = ROOT.parent/'reports/stop_feedback_20261015/earlier_action_alternatives_complete/earlier_right_rotation.npz'
        import hashlib
        with np.load(source) as original:
            o = original['observations']
            obs = convert_observations(o, original['masks'], original['commands'], original['labels'][:, 0])
            n = 100
            executed_reasons = np.asarray([FEEDBACK_REASONS[i] for i in
                o[1:n+1, 303+REASON_SLICE.start:303+REASON_SLICE.stop].argmax(-1)])
            arrays = dict(observations=obs[:n+1], times=original['labels'][:n+1, 0],
                choices=original['choices'][:n], masks=original['masks'][:n], commands=original['commands'][:n],
                scans=original['scans'][:n], odometry=original['odometry'][:n],
                measured_velocity=original['measured_velocity'][:n], physical_poses=original['labels'][:n+1, 1:4],
                executed_reasons=executed_reasons, metadata_json=np.asarray(json.dumps(dict(
                    source_trace=str(source), source_trace_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                    source_release_s=20.))))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'warmups').mkdir()
            path = root/'warmups/example.npz'
            np.savez_compressed(path, **arrays)
            history, frames = initialize_history(root, 'warmups/example.npz', obs[n])
            self.assertEqual(frames, 100)
            history.check(obs[n], float(arrays['times'][-1]))
            arrays['observations'] = arrays['observations'].copy()
            arrays['observations'][-1, 300+STAT_SLICE.start+3] += .1
            np.savez_compressed(path, **arrays)
            with self.assertRaises(ValueError):
                initialize_history(root, 'warmups/example.npz', arrays['observations'][-1])


if __name__ == '__main__':
    unittest.main()
