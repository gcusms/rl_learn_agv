from collections import deque
import ctypes
import unittest
import weakref

import mujoco
import numpy as np
import torch

from experiments.recovery_state_snapshot import RecoverySnapshot


class State:
    pass


class Forwarding:
    def __init__(self, base):
        self.base = base

    def __getattr__(self, name):
        return getattr(self.base, name)


class SnapshotTests(unittest.TestCase):
    def setUp(self):
        self.env, self.runtime = State(), State()
        self.env.renderer = None
        self.env.model = mujoco.MjModel.from_xml_string(
            '<mujoco><worldbody><body><freejoint/>'
            '<geom type="sphere" size=".1" mass="1"/>'
            '</body></worldbody></mujoco>')
        self.env.data = mujoco.MjData(self.env.model)
        self.env.history = deque([np.array([1., 2.])], maxlen=3)
        self.env.np_random = np.random.default_rng(17)
        self.env.localizer = Forwarding(State())
        self.env.localizer.env = weakref.proxy(self.env)
        self.runtime.core = self.env.localizer.base
        self.runtime.guard = State()
        self.runtime.guard.entries = {(1, 2): [np.array([3., 4.]), 1.]}
        self.runtime.library = ctypes.CDLL(None)
        self.runtime.function = self.runtime.library.getpid
        self.runtime.actor = torch.jit.trace(torch.nn.Linear(2, 2).eval(), torch.ones(1, 2))

    def test_physics_history_rng_and_cross_references_are_independent(self):
        snapshot = RecoverySnapshot(self.env, self.runtime)
        first, policy = snapshot.restore()
        first.data.qpos[0] = 4.
        first.model.opt.gravity[0] = 8.
        first.history[0][0] = 9.
        policy.guard.entries[(1, 2)][0][0] = 12.
        random_draw = first.np_random.normal(size=4)
        first.localizer.env.history.append(np.array([7.]))
        second, other = snapshot.restore()
        self.assertEqual(second.data.qpos[0], 0.)
        self.assertEqual(second.model.opt.gravity[0], 0.)
        np.testing.assert_array_equal(second.history[0], [1., 2.])
        self.assertEqual(len(second.history), 1)
        self.assertEqual(other.guard.entries[(1, 2)][0][0], 3.)
        self.assertIs(other.core, second.localizer.base)
        self.assertIs(other.library, self.runtime.library)
        self.assertIs(other.function, self.runtime.function)
        np.testing.assert_array_equal(second.np_random.normal(size=4), random_draw)
        self.assertEqual(len(self.env.history), 1)
        np.testing.assert_array_equal(self.env.history[0], [1., 2.])

    def test_shared_inference_weights_cannot_silently_change(self):
        snapshot = RecoverySnapshot(self.env, self.runtime)
        with torch.no_grad():
            next(self.runtime.actor.parameters()).add_(1.)
        with self.assertRaisesRegex(ValueError, 'shared inference model changed'):
            snapshot.restore()

    def test_global_rng_state_restores(self):
        snapshot = RecoverySnapshot(self.env, self.runtime)
        snapshot.restore()
        expected = np.random.normal(size=3), torch.rand(3)
        snapshot.restore()
        np.testing.assert_array_equal(np.random.normal(size=3), expected[0])
        torch.testing.assert_close(torch.rand(3), expected[1], rtol=0, atol=0)

    def test_live_renderer_rejected(self):
        self.env.renderer = object()
        with self.assertRaisesRegex(ValueError, 'headless'):
            RecoverySnapshot(self.env, self.runtime)


if __name__ == '__main__':
    unittest.main()
