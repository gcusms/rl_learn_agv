"""Contract checks for untrained proactive recovery; these are not navigation scores."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from experiments.proactive_recovery import ProactiveRecoveryPolicy, contract
from experiments.recovery_demonstrations import load_demonstrations
from experiments.stop_recovery_policy import StopRecoveryPPO, export_recovery, warm_start_recovery
from experiments.test_steady_base_recovery import CHECKPOINT, Interface, PROJECT, SOURCE
from experiments.train_prefix_recovery import frozen_base_digest
from neural_navigation.decision_feedback import STAT_SLICE
from neural_navigation.masked_actions import ACTIONS, masks_from_observation


class ProactiveRecoveryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.initial = warm_start_recovery(CHECKPOINT, Interface(), seed=231015,
            policy_class=ProactiveRecoveryPolicy)
        cls.prior = warm_start_recovery(CHECKPOINT, Interface(), seed=231015)
        with np.load(SOURCE.parent/'recover_forward.npz', allow_pickle=False) as data:
            cls.full = np.concatenate([data['observations'][:, :300], data['observations'][:, 303:]], axis=-1)
            cls.times = data['labels'][:, 0].copy()
        cls.masks = masks_from_observation(cls.full[:, :282])

    def test_successful_early_maneuver_is_learnable_without_changing_initial_choices(self):
        x = torch.from_numpy(self.full)
        with torch.no_grad():
            old = self.prior.policy.get_distribution(x, self.masks).distribution.probs.clone()
            new = self.initial.policy.get_distribution(x, self.masks).distribution.probs.clone()
            old_support = self.prior.policy.recovery_support(x)
            new_support = self.initial.policy.recovery_support(x)
        early = (self.times >= 6.-1e-8) & (self.times < 7.4-1e-8)
        self.assertEqual(int(early.sum()), 7)
        self.assertFalse(old_support[early].any())
        self.assertTrue(new_support[early].all())
        self.assertTrue(self.masks[early, 5].all())
        np.testing.assert_array_equal(new.argmax(-1), old.argmax(-1))
        self.assertEqual(frozen_base_digest(self.initial.policy), frozen_base_digest(self.prior.policy))
        self.assertTrue((new[:, 0] > 0.).all())
        self.assertEqual(int(torch.count_nonzero(new[~torch.from_numpy(self.masks)])), 0)

    def test_invalid_sensing_and_near_goal_are_protected_even_with_large_residual(self):
        model = copy.deepcopy(self.initial)
        early = self.full[(self.times >= 6.-1e-8) & (self.times < 7.4-1e-8)]
        protected = np.tile(early, (3, 1))
        n = len(early)
        protected[:n, 300+STAT_SLICE.start+7] = 0.  # No valid scan.
        protected[n:2*n, 300+STAT_SLICE.start+4] = 0.  # No valid forecast.
        protected[2*n:, 218:220] = 0.  # Goal approach/docking.
        full = np.concatenate([protected, early])
        masks = masks_from_observation(full[:, :282])
        x = torch.from_numpy(full)
        with torch.no_grad():
            before = model.policy.get_distribution(x, masks).distribution.probs.clone()
            model.policy.action_net.residual.bias.fill_(-100.)
            model.policy.action_net.residual.bias[5] = 100.
            after = model.policy.get_distribution(x, masks).distribution.probs.clone()
        support = model.policy.recovery_support(x)
        self.assertFalse(support[:3*n].any())
        self.assertTrue(support[3*n:].all())
        torch.testing.assert_close(before[:3*n], after[:3*n], rtol=0, atol=0)
        self.assertTrue((after[3*n:, 5] > .99).all())
        self.assertEqual(int(torch.count_nonzero(after[~torch.from_numpy(masks)])), 0)

    def test_sampling_update_and_saved_export_share_the_new_contract(self):
        model = copy.deepcopy(self.initial)
        x = torch.from_numpy(self.full)
        with torch.no_grad():
            actions, _, sampled = model.policy(x, action_masks=self.masks)
            _, evaluated, entropy = model.policy.evaluate_actions(x, actions, action_masks=self.masks)
        torch.testing.assert_close(sampled, evaluated, rtol=0, atol=1e-6)
        self.assertTrue(torch.isfinite(entropy).all())
        self.assertTrue(self.masks[np.arange(len(actions)), actions.numpy()].all())
        with torch.no_grad():
            model.policy.action_net.residual.bias[5] = 8.
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            model.save(folder/'policy.zip')
            restored = StopRecoveryPPO.load(folder/'policy.zip', device='cpu')
            self.assertEqual(restored.recovery_contract, contract())
            export_recovery(restored, PROJECT/'models/localization_mapping_20261015/ray_consistent_candidate/portable', folder/'portable')
            manifest = json.loads((folder/'portable/manifest.json').read_text())
            self.assertEqual(manifest['navigation_preservation_contract']['recovery_contract'], contract())
            runtime_obs = np.concatenate([self.full[:, :300], np.zeros((len(self.full), 3), np.float32), self.full[:, 300:]], axis=-1)
            with torch.no_grad():
                actual = torch.jit.load(str(folder/'portable/actor.pt'))(torch.from_numpy(runtime_obs)).numpy()
                before = model.policy.get_distribution(x, self.masks).distribution.probs.clone()
                after = restored.policy.get_distribution(x, self.masks).distribution.probs.clone()
            torch.testing.assert_close(before, after, rtol=0, atol=0)
            np.testing.assert_array_equal(actual, ACTIONS[restored.predict(self.full,
                deterministic=True, action_masks=self.masks)[0]])

    def test_offline_intervals_include_early_turn_and_late_waits_but_exclude_prefix_waits(self):
        intervals = [[6., 7.4], [20., 140.]]
        observations, masks, choices, rows = load_demonstrations([
            dict(report=str(SOURCE), id='recover_forward', sensor_intervals_s=intervals)], self.initial.policy)
        np.testing.assert_array_equal(choices[:7], [5]*7)
        self.assertTrue((choices[7:] == 0).any())
        self.assertTrue((choices[7:] == 1).any())
        expected = (self.times >= 6.-1e-8) & (self.times < 7.4-1e-8) | (self.times >= 20.-1e-8)
        expected &= self.initial.policy.recovery_support(torch.from_numpy(self.full)).numpy()
        self.assertEqual(len(observations), int(expected.sum()))
        self.assertEqual(rows[0]['sensor_intervals_s'], intervals)
        self.assertTrue(masks[np.arange(len(choices)), choices].all())
        for invalid in [[], [[7., 6.]], [[6., 8.], [7., 9.]], [[-1., 1.]], [[0., float('nan')]], [6., 7.4]]:
            with self.subTest(intervals=invalid), self.assertRaises(ValueError):
                load_demonstrations([dict(report=str(SOURCE), id='recover_forward',
                    sensor_intervals_s=invalid)], self.initial.policy)


if __name__ == '__main__':
    unittest.main()
