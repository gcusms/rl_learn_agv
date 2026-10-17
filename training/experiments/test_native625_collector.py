"""Collector boundary checks; these use stand-ins, not navigation validation."""
import copy
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from experiments.guard_reason_release_actor import GuardReasonReleaseActor
from experiments.native625_onpolicy import Native625OnPolicy
from experiments.recovery_state_snapshot import tensor_digest
from experiments.train_native625_recovery import collect_episode
from neural_navigation.experiment_budget import ExperimentBudget
from neural_navigation.masked_actions import ACTIONS, COMMANDS


ROOT = Path(__file__).resolve().parents[2]


class CollectorEnvironment:
    def __init__(self):
        self.truth = np.array([0., 0., 0.])
        self.goal = np.array([1., 0.])
        self.elapsed = 106.
        self.scan = np.ones(220)
        self.observed_odometry = self.truth.copy()
        self.measured_velocity = np.zeros(2)
        self.minimum_gap = .1
        self.collision_events = self.hard_guard_activations = 0
        self.executions = []

    def _minimum_gap(self):
        return self.minimum_gap

    def step(self, action):
        self.executions.append(np.asarray(action).copy())
        self.truth[0] += .04 * action[0]
        self.truth[2] += .1 * action[1]
        self.elapsed += .2


class CollectorRuntime:
    def __init__(self, observation, *, rewrite=False, arrived=False):
        self.observation = observation.copy()
        self.policy = SimpleNamespace(policy=None)
        self.context = SimpleNamespace(last_executed=np.zeros(2))
        self.rewrite, self.arrived = rewrite, arrived
        self.samples = []

    def step(self, scan, odometry, velocity, stamp, capture_observation):
        action = self.policy.policy.predict(self.observation)
        self.samples.append(self.policy.policy.last_sample)
        return dict(proposed_action=action,
            command=np.zeros(2) if self.rewrite or self.arrived else action * [.2, .5],
            policy_observation=self.observation.copy(),
            stop_reason='estimated_goal_reached' if self.arrived else None)


class CollectorBoundary(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.budget = ExperimentBudget(.1, memory_mib=1400)
        model = ROOT / 'models/stop_feedback_20261015/full_mission_retention_pair_s231016/retained500'
        cls.parent = torch.jit.load(str(model / 'parent_actor.pt'), map_location='cpu').eval()
        cls.state = torch.load(model / 'checkpoint.pt', map_location='cpu', weights_only=False)['actor']
        path = ROOT / 'reports/stop_feedback_20261015/context_broad_diagnostic_20261016/physical/hard3_w11_training_stall_t106.npz'
        with np.load(path) as archive:
            cls.observation = archive['observations'][archive['model_controlled']][0].copy()

    def setUp(self):
        actor = GuardReasonReleaseActor(self.parent)
        actor.load_state_dict(copy.deepcopy(self.state))
        self.learner = Native625OnPolicy(actor, temperature=4.)
        # Select a private RNG seed producing a real legal motion on its first
        # draw, so execution rewrite is tested without changing probabilities.
        self.generator = torch.Generator()
        for seed in range(100):
            self.generator.manual_seed(seed)
            sample = self.learner.sample(self.observation, self.generator)
            if sample.action != 0:
                self.generator.manual_seed(seed)
                break
        else:
            self.fail('Expected a legal motion with positive probability')
        self.environment = CollectorEnvironment()
        self.field = SimpleNamespace(value=lambda xy: float(1. - xy[0]))
        self.report = dict(ppo_decisions=0, ppo_physical_steps=0)

    def tearDown(self):
        self.budget.tick()

    def collect(self, runtime, maximum):
        return collect_episode(self.environment, runtime, self.learner,
            self.generator, self.field, self.budget, maximum, self.report)

    def test_sample_likelihood_mask_and_executed_command_remain_linked(self):
        runtime = CollectorRuntime(self.observation)
        rollout, row = self.collect(runtime, 2)
        np.testing.assert_array_equal(rollout['actions'], [x.action for x in runtime.samples])
        np.testing.assert_allclose(rollout['executed_commands'], COMMANDS[rollout['actions']], atol=2e-8)
        np.testing.assert_array_equal(self.environment.executions, ACTIONS[rollout['actions']])
        with torch.no_grad():
            recomputed = self.learner.evaluate(rollout['observations'], rollout['masks'], rollout['actions'])
        np.testing.assert_allclose(rollout['old_log_probability'], recomputed['log_probability'].numpy(), atol=1e-6)
        np.testing.assert_array_equal(rollout['terminated'], [False, True])
        np.testing.assert_array_equal(rollout['episode_ends'], [False, True])
        self.assertEqual(rollout['next_values'][-1], 0.)
        self.assertAlmostEqual(float(rollout['returns'][-1]), float(rollout['rewards'][-1]), places=5)
        self.assertEqual(self.report['ppo_decisions'], 2)
        self.assertEqual(self.report['ppo_physical_steps'], 2)
        self.assertEqual(row['reward_terms']['terminal'], -5.)
        self.assertTrue(row['training_episode_not_independent_validation'])

    def test_unexpected_projection_fails_before_physics_or_ppo_update(self):
        runtime = CollectorRuntime(self.observation, rewrite=True)
        actor_before = tensor_digest(self.learner.actor.head)
        with self.assertRaisesRegex(ValueError, 'sampled command was changed'):
            self.collect(runtime, 2)
        self.assertNotEqual(runtime.samples[0].action, 0)
        self.assertEqual(self.report['ppo_decisions'], 0)
        self.assertEqual(self.report['ppo_physical_steps'], 0)
        self.assertEqual(len(self.environment.executions), 0)
        self.assertEqual(actor_before, tensor_digest(self.learner.actor.head))

    def test_arrival_keeps_sampled_action_and_independent_physical_check(self):
        self.environment.goal = np.array([.05, 0.])
        runtime = CollectorRuntime(self.observation, arrived=True)
        rollout, row = self.collect(runtime, 2)
        self.assertNotEqual(rollout['actions'][0], 0)
        np.testing.assert_array_equal(rollout['executed_commands'], [[0., 0.]])
        self.assertTrue(rollout['execution_changed'][0])
        self.assertTrue(row['safe_success'])
        self.assertEqual(float(rollout['returns'][0]), 10.)
        self.assertEqual(len(self.environment.executions), 0)
        self.assertEqual(self.report['ppo_decisions'], 1)
        self.assertEqual(self.report['ppo_physical_steps'], 0)

    def test_false_arrival_is_terminal_failure(self):
        runtime = CollectorRuntime(self.observation, arrived=True)
        rollout, row = self.collect(runtime, 2)
        self.assertFalse(row['safe_success'])
        self.assertTrue(row['false_arrival'])
        self.assertEqual(float(rollout['returns'][0]), -20.)
        self.assertEqual(len(self.environment.executions), 0)


if __name__ == '__main__':
    unittest.main()
