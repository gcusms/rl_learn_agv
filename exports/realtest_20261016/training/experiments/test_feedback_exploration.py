from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from experiments.feedback_exploration import SCHEMA, temperature
from experiments.stop_feedback_ppo import warm_start, StopFeedbackPPO
from experiments.test_stop_feedback_ppo import Interface, TrainingInterface
from experiments.train_stop_feedback import export_candidate
from navrl.geometry import ROOT
from neural_navigation.decision_feedback import WAIT_SCHEMA, STAT_SLICE
from neural_navigation.masked_actions import ACTIONS, masks_from_observation


class TimedInterface(TrainingInterface):
    feedback_schema = WAIT_SCHEMA

    def observation(self):
        observation = super().observation()
        observation[300+STAT_SLICE.start+3] = min(self.index*.2/5., 1.)
        observation[300+STAT_SLICE.start+4] = 1.
        observation[300+STAT_SLICE.start+7] = 1.
        return observation


class ExplorationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        env = Interface(); env.feedback_schema = WAIT_SCHEMA
        cls.model = warm_start(ROOT.parent/'models/quality_recovery_20261014/policy_2000.zip',
                               env, exploration_schema=SCHEMA)
        path = ROOT.parent/'models/stop_feedback_20261015/ppo_v2_4000_s221015/rollouts/rollout_002750.npz'
        with np.load(path, allow_pickle=False) as data:
            cls.observations = data['observations'][::5, 0].copy()
        cls.masks = masks_from_observation(cls.observations[:, :282])

    def test_temperature_is_bounded_and_requires_valid_choices_away_from_goal(self):
        x = torch.zeros(6, 436)
        x[:, 235] = 1.; x[:, 218] = .2
        offset = 300+STAT_SLICE.start
        x[:, offset+3] = .6; x[:, offset+4] = 1.; x[:, offset+7] = 1.
        x[1, offset+4] = 0.  # Unknown predictor is not clear or exploratory.
        x[2, offset+7] = 0.  # Invalid scan.
        x[3, 235] = 0.      # Only stop remains.
        x[4, 218] = .01     # Preserve close approach and docking.
        x[5, offset+3] = .4 # Two seconds of known waiting.
        torch.testing.assert_close(temperature(x), torch.tensor([3., 1., 1., 1., 1., 2.]))

    def test_real_wait_states_gain_exploration_without_changing_argmax_or_masks(self):
        policy = self.model.policy; x = torch.from_numpy(self.observations)
        with torch.no_grad():
            nominal = policy.get_nominal_distribution(x, self.masks).distribution.probs.clone()
            explored = policy.get_distribution(x, self.masks).distribution.probs.clone()
        active = (temperature(x) > 1.) & (nominal[:, 0] > .9)
        self.assertTrue(active.any())
        self.assertTrue((explored[active, 0] < nominal[active, 0]).all())
        torch.testing.assert_close(explored.sum(-1), torch.ones(len(x)))
        self.assertEqual(int(torch.count_nonzero(explored[~torch.from_numpy(self.masks)])), 0)
        self.assertTrue((explored[:, 0] > 0).all())
        torch.testing.assert_close(nominal.argmax(-1), explored.argmax(-1), rtol=0, atol=0)

    def test_sampled_and_updated_likelihoods_match_including_forced_stops(self):
        policy = self.model.policy; x = torch.from_numpy(self.observations.copy())
        masks = self.masks.copy(); masks[0] = False; masks[0, 0] = True
        with torch.no_grad():
            choices, _, old_log = policy(x, action_masks=masks)
            _, update_log, entropy = policy.evaluate_actions(x, choices, action_masks=masks)
            predict_log = policy.get_distribution(x, masks).log_prob(choices)
        torch.testing.assert_close(old_log, update_log, atol=1e-6, rtol=0)
        torch.testing.assert_close(old_log, predict_log, atol=1e-6, rtol=0)
        self.assertEqual(int(choices[0]), 0)
        self.assertAlmostEqual(float(entropy[0]), 0.)
        self.assertTrue(masks[np.arange(len(choices)), choices.numpy()].all())

    def test_checkpoint_export_preserves_probabilities_and_deterministic_commands(self):
        model = self.model
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory); model.save(path/'policy.zip')
            restored = StopFeedbackPPO.load(path/'policy.zip', device='cpu')
            self.assertEqual(restored.exploration_schema, SCHEMA)
            x = torch.from_numpy(self.observations)
            with torch.no_grad():
                before = model.policy.get_distribution(x, self.masks).distribution.probs.clone()
                after = restored.policy.get_distribution(x, self.masks).distribution.probs.clone()
            torch.testing.assert_close(before, after, rtol=0, atol=0)
            source = ROOT.parent/'models/localization_mapping_20261015/ray_consistent_candidate/portable'
            export_candidate(restored, source, path/'portable')
            actor = torch.jit.load(str(path/'portable/actor.pt'))
            runtime_obs = np.concatenate([self.observations[:, :300],
                np.zeros((len(x), 3), np.float32), self.observations[:, 300:]], axis=1)
            expected = ACTIONS[restored.predict(self.observations, deterministic=True, action_masks=self.masks)[0]]
            with torch.no_grad():
                np.testing.assert_array_equal(actor(torch.from_numpy(runtime_obs)).numpy(), expected)

    def test_on_policy_update_runs_with_temperature_and_original_nominal_anchor(self):
        env = TimedInterface()
        model = warm_start(ROOT.parent/'models/quality_recovery_20261014/policy_2000.zip',
                           env, exploration_schema=SCHEMA)
        model.learn(total_timesteps=250)
        self.assertEqual(model.num_timesteps, 250)
        for key in ['train/loss', 'train/approx_kl', 'train/clear_reference_kl']:
            self.assertTrue(np.isfinite(model.logger.name_to_value[key]), key)

    def test_v1_feedback_cannot_enable_v2_exploration(self):
        with self.assertRaises(ValueError):
            warm_start(ROOT.parent/'models/quality_recovery_20261014/policy_2000.zip',
                       Interface(), exploration_schema=SCHEMA)


if __name__ == '__main__': unittest.main()
