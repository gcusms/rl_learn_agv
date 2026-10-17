"""Likelihood, masking, termination and optimizer checks on real625 inputs."""
import copy
from pathlib import Path
import unittest

import numpy as np
import torch

from experiments.guard_reason_release_actor import GuardReasonReleaseActor
from experiments.native625_onpolicy import Native625OnPolicy, generalized_advantage
from experiments.recovery_state_snapshot import tensor_digest
from experiments.refit_phase_tail_pair import optimizer_digest
from neural_navigation.experiment_budget import ExperimentBudget
from neural_navigation.masked_actions import ACTIONS


ROOT = Path(__file__).resolve().parents[2]


class OnPolicyContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.budget = ExperimentBudget(.1, memory_mib=1400)
        model = ROOT / 'models/stop_feedback_20261015/full_mission_retention_pair_s231016/retained500'
        cls.parent = torch.jit.load(str(model / 'parent_actor.pt'), map_location='cpu').eval()
        cls.state = torch.load(model / 'checkpoint.pt', map_location='cpu', weights_only=False)['actor']
        with np.load(ROOT / 'reports/stop_feedback_20261015/full_mission_retention_protocol_20261016/demonstrations.npz') as data:
            cls.observations = data['observations'].copy()
        with np.load(ROOT / 'reports/stop_feedback_20261015/context_broad_diagnostic_20261016/physical/hard3_w11_training_stall_t106.npz') as data:
            cls.late = data['observations'][data['model_controlled']].copy()

    def setUp(self):
        actor = GuardReasonReleaseActor(self.parent)
        actor.load_state_dict(copy.deepcopy(self.state))
        self.brain = Native625OnPolicy(actor, temperature=4.)

    def tearDown(self):
        self.budget.tick()

    def test_greedy_parity_and_original_masks_on_all_real_inputs(self):
        x = torch.from_numpy(self.observations)
        with torch.no_grad():
            terms = self.brain.evaluate(x)
            expected = self.brain.actor(x).numpy()
        np.testing.assert_array_equal(ACTIONS[terms['distribution'].probs.argmax(1)], expected)
        self.assertTrue(torch.equal(terms['masks'], self.brain.actor.masks(x)))
        self.assertTrue((terms['distribution'].probs[~terms['masks']] == 0).all())

    def test_sample_action_and_likelihood_survive_execution_stop(self):
        generator = torch.Generator().manual_seed(8261016)
        sample = self.brain.sample(self.observations[0], generator)
        # Find an actual sampled motion, retaining its real probability.
        for observation in self.observations[:100]:
            sample = self.brain.sample(observation, generator)
            if sample.action:
                break
        self.assertNotEqual(sample.action, 0)
        action, likelihood = sample.action, sample.old_log_probability
        sample.record_execution(dict(proposed_action=ACTIONS[action], command=[0., 0.]))
        self.assertEqual(sample.action, action)
        self.assertEqual(sample.old_log_probability, likelihood)
        self.assertTrue(sample.execution_changed)
        np.testing.assert_array_equal(sample.executed_command, [0., 0.])
        with self.assertRaises(ValueError):
            sample.record_execution(dict(proposed_action=ACTIONS[action], command=[0., 0.]))

    def test_private_sampler_rng_is_not_reset_with_global_torch_state(self):
        generator = torch.Generator().manual_seed(8261017)
        expected = torch.Generator().manual_seed(8261017)
        observation = self.late[0]
        with torch.no_grad():
            probability = self.brain.evaluate(observation[None])['distribution'].probs
        expected_choices = [int(torch.multinomial(probability, 1, generator=expected)[0, 0]) for _ in range(30)]
        actual = []
        for _ in range(30):
            torch.manual_seed(0)  # Physics snapshot restore may reset this RNG.
            actual.append(self.brain.sample(observation, generator).action)
        self.assertEqual(actual, expected_choices)
        self.assertGreater(len(set(actual)), 1)

    def _update(self, observation, action, advantage=1.):
        with torch.no_grad():
            terms = self.brain.evaluate(observation[None], actions=[action])
        return self.brain.update(observation[None], terms['masks'], [action],
            terms['log_probability'], [advantage], [1.], sampled_temperature=4.,
            entropy_coefficient=0., anchor_strength=0.)

    def test_positive_legal_turn_advantage_increases_its_probability(self):
        observation = next(x for x in self.late if self.brain.actor.masks(torch.from_numpy(x[None]))[0, 5])
        with torch.no_grad():
            before = float(self.brain.evaluate(observation[None])['distribution'].probs[0, 5])
        statistics = self._update(observation, 5)
        with torch.no_grad():
            after = float(self.brain.evaluate(observation[None])['distribution'].probs[0, 5])
        self.assertTrue(statistics['actor_updated'])
        self.assertGreater(after, before)

    def test_only_stop_updates_critic_without_decaying_actor_adam(self):
        observation = self.late[0].copy()
        self._update(observation, 9)  # Prime real nonzero Adam moments.
        observation[235:244] = 0.
        observation[267:269] = 0.
        before = tensor_digest(self.brain.actor.head)
        optimizer = optimizer_digest(self.brain.actor_optimizer)
        critic = tensor_digest(self.brain.critic)
        statistics = self._update(observation, 0)
        self.assertEqual(statistics['eligible_samples'], 0)
        self.assertFalse(statistics['actor_updated'])
        self.assertEqual(tensor_digest(self.brain.actor.head), before)
        self.assertEqual(optimizer_digest(self.brain.actor_optimizer), optimizer)
        self.assertNotEqual(tensor_digest(self.brain.critic), critic)

    def test_temperature_or_original_mask_mismatch_is_rejected(self):
        observation = self.late[0][None]
        with torch.no_grad():
            terms = self.brain.evaluate(observation, actions=[0])
        with self.assertRaises(ValueError):
            self.brain.update(observation, terms['masks'], [0], terms['log_probability'],
                [1.], [1.], sampled_temperature=2.)
        wrong = terms['masks'].clone()
        wrong[0, 1] = ~wrong[0, 1]
        with self.assertRaises(ValueError):
            self.brain.evaluate(observation, masks=wrong, actions=[0])


class AdvantageBoundaries(unittest.TestCase):
    def test_truncation_uses_final_value_and_does_not_cross_reset(self):
        advantages, returns = generalized_advantage([1., 100.], [2., 50.], [3., 999.],
            [False, True], [True, True], gamma=.9, gae_lambda=.95)
        torch.testing.assert_close(returns, torch.tensor([3.7, 100.]))
        torch.testing.assert_close(advantages, torch.tensor([1.7, 50.]))

    def test_finite_task_deadline_is_terminal(self):
        _, returns = generalized_advantage([-5.], [2.], [999.], [True], [True])
        torch.testing.assert_close(returns, torch.tensor([-5.]))


if __name__ == '__main__':
    unittest.main()
