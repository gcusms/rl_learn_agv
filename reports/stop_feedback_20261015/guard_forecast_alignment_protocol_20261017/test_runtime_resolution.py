"""Explicit version selection must preserve old runtime identity."""
import unittest

from neural_navigation.goal_pose_runtime import resolve_runtime


class RuntimeResolutionTests(unittest.TestCase):
    def test_v1_and_v2_are_distinct_and_selected_only_by_name(self):
        from neural_navigation.release_runtime import ReleaseRuntime
        from neural_navigation.aligned_release_runtime import AlignedReleaseRuntime
        self.assertIs(resolve_runtime('neural_navigation.release_runtime.ReleaseRuntime'), ReleaseRuntime)
        self.assertIs(resolve_runtime('neural_navigation.aligned_release_runtime.AlignedReleaseRuntime'),
            AlignedReleaseRuntime)
        self.assertIsNone(resolve_runtime('625'))
        self.assertIsNone(resolve_runtime(None))


if __name__ == '__main__':
    unittest.main()
