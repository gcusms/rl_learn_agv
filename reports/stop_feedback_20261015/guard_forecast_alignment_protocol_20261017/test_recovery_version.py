"""The recovery adapter admits explicit v2 without relaxing unknown versions."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments.release_recovery import evaluate_recovery


class ReachedPrefix(Exception):
    pass


class RecoveryVersionTests(unittest.TestCase):
    def test_explicit_versions_reach_prefix_validation_unknown_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            bundle = Path(tmp)
            for name in ['neural_navigation.release_runtime.ReleaseRuntime',
                         'neural_navigation.aligned_release_runtime.AlignedReleaseRuntime']:
                with self.subTest(runtime=name):
                    (bundle/'manifest.json').write_text(json.dumps(dict(required_runtime=name)))
                    with patch('experiments.release_recovery.load_prefix', side_effect=ReachedPrefix):
                        with self.assertRaises(ReachedPrefix):
                            evaluate_recovery({}, bundle, None)
            (bundle/'manifest.json').write_text(json.dumps(dict(required_runtime='625')))
            with self.assertRaisesRegex(ValueError, 'explicitly versioned'):
                evaluate_recovery({}, bundle, None)


if __name__ == '__main__':
    unittest.main()
