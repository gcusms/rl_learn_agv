from pathlib import Path
import copy
import json
import tempfile
import unittest

import numpy as np
import torch

from experiments.recovery_demonstrations import load_demonstrations
from experiments.stop_feedback_navigation import FeedbackInterface
from experiments.steady_base_recovery import SteadyBaseRecoveryPolicy, contract
from experiments.stop_recovery_policy import StopRecoveryPPO, export_recovery, warm_start_recovery
from experiments.train_prefix_recovery import frozen_base_digest
from navrl.geometry import ROOT
from neural_navigation.decision_feedback import WAIT_SCHEMA
from neural_navigation.masked_actions import ACTIONS, masks_from_observation


class Interface(FeedbackInterface):
    feedback_schema = WAIT_SCHEMA


PROJECT = ROOT.parent
CHECKPOINT = PROJECT/'models/quality_recovery_20261014/policy_2000.zip'
SOURCE = PROJECT/'reports/stop_feedback_20261015/turn_recovery_20261015/RESULTS.json'


class SteadyBaseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.initial = warm_start_recovery(CHECKPOINT, Interface(), seed=231015,
            policy_class=SteadyBaseRecoveryPolicy)
        cls.prior = warm_start_recovery(CHECKPOINT, Interface(), seed=231015)
        with np.load(SOURCE.parent/'recover_forward.npz', allow_pickle=False) as data:
            cls.full = np.concatenate([data['observations'][:, :300], data['observations'][:, 303:]], axis=-1)
        cls.masks = masks_from_observation(cls.full[:, :282])

    def test_only_frozen_region_sampling_changes_and_stop_stays_possible(self):
        x = torch.from_numpy(self.full)
        with torch.no_grad():
            old = self.prior.policy.get_distribution(x, self.masks).distribution.probs.clone()
            new = self.initial.policy.get_distribution(x, self.masks).distribution.probs.clone()
            active = self.initial.policy.recovery_support(x)
        self.assertTrue(active.any())
        self.assertTrue((~active).any())
        torch.testing.assert_close(new[active], old[active], rtol=0, atol=0)
        np.testing.assert_array_equal(new.argmax(-1), old.argmax(-1))
        self.assertTrue((new[~active].max(-1).values > .979).all())
        self.assertTrue((new[~active, 0] > 0.).all())
        self.assertEqual(int(torch.count_nonzero(new[~torch.from_numpy(self.masks)])), 0)
        self.assertEqual(frozen_base_digest(self.initial.policy), frozen_base_digest(self.prior.policy))

    def test_sampling_likelihood_and_masked_execution_describe_the_same_action(self):
        x = torch.from_numpy(self.full)
        policy = self.initial.policy
        with torch.no_grad():
            actions, _, sampled_log = policy(x, action_masks=self.masks)
            _, update_log, entropy = policy.evaluate_actions(x, actions, action_masks=self.masks)
            distribution_log = policy.get_distribution(x, self.masks).log_prob(actions)
        torch.testing.assert_close(sampled_log, update_log, rtol=0, atol=1e-6)
        torch.testing.assert_close(sampled_log, distribution_log, rtol=0, atol=1e-6)
        self.assertTrue(self.masks[np.arange(len(actions)), actions.numpy()].all())
        self.assertTrue(torch.isfinite(entropy).all())
        forced = self.full[:1].copy()
        forced[:, 235:244] = 0.; forced[:, 267:269] = 0.
        forced_mask = masks_from_observation(forced[:, :282])
        with torch.no_grad():
            only_stop = policy.get_distribution(torch.from_numpy(forced), forced_mask).distribution.probs
        torch.testing.assert_close(only_stop, torch.tensor([[1.]+[0.]*11]), rtol=0, atol=0)

    def test_recovery_update_keeps_frozen_actor_and_has_finite_actual_ppo_likelihoods(self):
        observations, masks, choices, _ = load_demonstrations(
            [dict(report=str(SOURCE), id='recover_forward')], self.initial.policy)

        class TrainingInterface(Interface):
            def reset(self, *, seed=None, options=None):
                self.index = 0
                return observations[0].copy(), {}

            def action_masks(self):
                return masks[self.index].copy()

            def step(self, choice):
                reward = (1. if int(choice) == choices[self.index] else -1.)/60.
                self.index += 1
                return observations[self.index % len(observations)].copy(), reward, self.index == len(observations), False, {}

        model = warm_start_recovery(CHECKPOINT, TrainingInterface(), seed=17,
            policy_class=SteadyBaseRecoveryPolicy)
        before = frozen_base_digest(model.policy)
        residual = model.policy.action_net.residual.weight.detach().clone()
        model.learn(total_timesteps=250)
        self.assertEqual(model.num_timesteps, 250)
        self.assertEqual(before, frozen_base_digest(model.policy))
        self.assertFalse(torch.equal(residual, model.policy.action_net.residual.weight))
        for key in ['train/loss', 'train/approx_kl', 'train/actor_recovery_fraction']:
            self.assertTrue(np.isfinite(model.logger.name_to_value[key]), key)

    def test_roundtrip_and_export_preserve_versioned_distribution_and_choices(self):
        model = copy.deepcopy(self.initial)
        with torch.no_grad():
            model.policy.action_net.residual.bias[1] = 8.
        x = torch.from_numpy(self.full)
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            model.save(folder/'policy.zip')
            restored = StopRecoveryPPO.load(folder/'policy.zip', device='cpu')
            self.assertEqual(restored.recovery_contract, contract())
            with torch.no_grad():
                before = model.policy.get_distribution(x, self.masks).distribution.probs.clone()
                after = restored.policy.get_distribution(x, self.masks).distribution.probs.clone()
            torch.testing.assert_close(before, after, rtol=0, atol=0)
            source = PROJECT/'models/localization_mapping_20261015/ray_consistent_candidate/portable'
            export_recovery(restored, source, folder/'portable')
            manifest = json.loads((folder/'portable/manifest.json').read_text())
            self.assertEqual(manifest['navigation_preservation_contract']['recovery_contract'], contract())
            actor = torch.jit.load(str(folder/'portable/actor.pt'))
            directional = np.concatenate([self.full[:, :300], np.zeros((len(self.full), 3), np.float32), self.full[:, 300:]], axis=-1)
            with torch.no_grad():
                actual = actor(torch.from_numpy(directional)).numpy()
            expected = ACTIONS[restored.predict(self.full, deterministic=True, action_masks=self.masks)[0]]
            np.testing.assert_array_equal(actual, expected)


if __name__ == '__main__':
    unittest.main()
