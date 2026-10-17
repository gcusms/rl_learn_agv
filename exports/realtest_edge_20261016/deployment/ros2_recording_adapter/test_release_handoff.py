"""Contract and publication-history regressions; no ROS or vehicle is started."""
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

from deployment.ros2_recording_adapter.model_bundle import inspect_bundle
from deployment.ros2_recording_adapter.execution_feedback import restore_published_history
from neural_navigation.decision_feedback import DecisionFeedback, WAIT_SCHEMA


ROOT = Path(__file__).resolve().parents[2]
CANDIDATE = ROOT / 'models/stop_feedback_20261015/release_context_bc2000_s231016/portable'
BASELINE = ROOT / 'models/quality_recovery_20261014/portable'


class ReleaseHandoffTests(unittest.TestCase):
    def inspect(self, bundle, **changes):
        manifest = json.loads((bundle / 'manifest.json').read_text())
        config = dict(bundle=str(bundle),
            manifest_sha256=hashlib.sha256((bundle / 'manifest.json').read_bytes()).hexdigest(),
            required_runtime=manifest.get('required_runtime',
                'experiments.quality_sensor_runtime.QualitySensorNavigationRuntime'),
            observation_dimensions=manifest['observation_contract']['dimensions'],
            action_choices=12)
        config.update(changes)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'model.json'
            path.write_text(json.dumps(config))
            return inspect_bundle(path)

    def test_candidate_and_rollback_load_with_their_own_contracts(self):
        candidate = self.inspect(CANDIDATE)
        self.assertEqual(candidate['observation_dimensions'], 625)
        self.assertEqual(candidate['required_runtime'], 'neural_navigation.release_runtime.ReleaseRuntime')
        self.assertFalse(candidate['deployment_approved'])
        self.assertEqual(self.inspect(BASELINE)['observation_dimensions'], 300)
        with self.assertRaisesRegex(ValueError, 'dimension'):
            self.inspect(CANDIDATE, observation_dimensions=300)
        with self.assertRaisesRegex(ValueError, 'checksum'):
            self.inspect(CANDIDATE, manifest_sha256='0' * 64)
        with self.assertRaisesRegex(ValueError, 'runtime'):
            self.inspect(CANDIDATE,
                required_runtime='experiments.docking_navigation.DockingRuntime')

    def runtime(self):
        feedback = DecisionFeedback(WAIT_SCHEMA)
        feedback.frame_mask = np.ones(12, dtype=bool)
        feedback.frame_timestamp = 1.
        feedback.record(dict(command=[.2, 0.], timestamp=1., stop_reason=None))
        core = SimpleNamespace(last_action=None)
        core.record_action = lambda action: setattr(core, 'last_action', np.asarray(action))
        return SimpleNamespace(core=core, context=None,
            policy=SimpleNamespace(feedback=feedback)), feedback

    def test_unpublished_motion_is_replaced_by_actual_external_stop(self):
        runtime, feedback = self.runtime()
        restore_published_history(runtime, [0., 0.], 'external_stop', 1.2)
        np.testing.assert_array_equal(runtime.core.last_action, [0., 0.])
        self.assertEqual(feedback.previous_reason, 'other')
        self.assertEqual(feedback.idle_since, 1.2)
        self.assertEqual(feedback.idle_with_choices_since, 1.2)
        self.assertEqual(feedback.recorded_timestamp, 1.)
        self.assertEqual(feedback.frame_timestamp, 1.)
        # Restoring the parent's publication must not consume the next probe's
        # record slot, or real inference fails as a duplicate execution.
        feedback.frame_timestamp = 1.4
        feedback.record(dict(command=[.1, 0.], timestamp=1.4, stop_reason=None))
        self.assertEqual(feedback.recorded_timestamp, 1.4)
        self.assertIsNone(feedback.idle_since)

    def test_actual_motion_clears_candidate_wait_and_only_stop_is_distinct(self):
        runtime, feedback = self.runtime()
        restore_published_history(runtime, [0., 0.], 'none', 1.2)
        self.assertEqual(feedback.previous_reason, 'policy_wait')
        restore_published_history(runtime, [.1, -.5], 'none', 1.4)
        np.testing.assert_array_equal(runtime.core.last_action, [.5, -1.])
        self.assertEqual(feedback.previous_reason, 'none')
        self.assertIsNone(feedback.idle_since)
        self.assertIsNone(feedback.idle_with_choices_since)
        feedback.frame_mask[1:] = False
        restore_published_history(runtime, [0., 0.], 'none', 1.6)
        self.assertEqual(feedback.previous_reason, 'only_stop_available')
        self.assertIsNone(feedback.idle_with_choices_since)

    def test_invalid_publication_does_not_update_history(self):
        runtime, feedback = self.runtime()
        for command, stamp in [([np.nan, 0.], 1.2), ([0., 0.], np.inf), ([0.], 1.2)]:
            with self.assertRaises(ValueError):
                restore_published_history(runtime, command, 'external_stop', stamp)
        self.assertIsNone(runtime.core.last_action)
        self.assertEqual(feedback.recorded_timestamp, 1.)


if __name__ == '__main__':
    unittest.main()
