from pathlib import Path
from types import SimpleNamespace
import copy
import json
import tempfile
import unittest

import numpy as np
import torch
from sb3_contrib import MaskablePPO

from experiments.recovery_demonstrations import load_demonstrations, initialize_from_demonstrations
from experiments.stop_recovery_policy import warm_start_recovery, StopRecoveryPPO, export_recovery
from experiments.stop_feedback_navigation import FeedbackInterface
from experiments.train_prefix_recovery import frozen_base_digest
from neural_navigation.decision_feedback import WAIT_SCHEMA, STAT_SLICE
from neural_navigation.masked_actions import ACTIONS, masks_from_observation
from navrl.geometry import ROOT


PROJECT = ROOT.parent
CHECKPOINT = PROJECT/'models/quality_recovery_20261014/policy_2000.zip'
SOURCE = PROJECT/'reports/stop_feedback_20261015/turn_recovery_20261015/RESULTS.json'


class Interface(FeedbackInterface):
    feedback_schema = WAIT_SCHEMA


class RecoveryPolicyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.initial = warm_start_recovery(CHECKPOINT, Interface())
        cls.observations, cls.masks, cls.choices, cls.sources = load_demonstrations(
            [dict(report=str(SOURCE), id='recover_forward')], cls.initial.policy)
        source = SOURCE.parent/'recover_forward.npz'
        with np.load(source) as data:
            cls.full = np.concatenate([data['observations'][:, :300], data['observations'][:, 303:]], axis=-1)
        cls.model = copy.deepcopy(cls.initial)

    def setUp(self):
        self.model = copy.deepcopy(self.initial)

    def test_initialization_exactly_matches_original_masked_actor(self):
        old = MaskablePPO.load(CHECKPOINT, device='cpu')
        masks = masks_from_observation(self.full[:, :282])
        with torch.no_grad():
            before = old.policy.get_distribution(torch.from_numpy(self.full[:, :300]), masks).distribution.probs.clone()
            after = self.model.policy.get_distribution(torch.from_numpy(self.full), masks).distribution.probs.clone()
        torch.testing.assert_close(before, after, rtol=0, atol=0)
        self.assertEqual(len(self.choices), 17)
        self.assertEqual(int(np.count_nonzero(self.choices)), 3)

    def test_large_residual_cannot_change_any_protected_state(self):
        policy = self.model.policy
        observations = np.concatenate([self.full, self.observations.copy(), self.observations.copy()])
        n = len(self.observations)
        observations[-2*n:-n, 300+STAT_SLICE.start+7] = 0.  # No scan.
        observations[-n:, 218:220] = 0.  # Preserve approach/docking.
        x = torch.from_numpy(observations)
        masks = masks_from_observation(observations[:, :282])
        with torch.no_grad():
            before = policy.get_distribution(x, masks).distribution.probs.clone()
            policy.action_net.residual.bias.fill_(-100.)
            policy.action_net.residual.bias[1] = 100.
            after = policy.get_distribution(x, masks).distribution.probs.clone()
        supported = policy.recovery_support(x)
        self.assertTrue(supported.any())
        self.assertTrue((~supported).any())
        torch.testing.assert_close(before[~supported], after[~supported], rtol=0, atol=0)
        self.assertTrue((after[supported & torch.from_numpy(masks[:, 1]), 1] > .99).all())
        self.assertEqual(int(torch.count_nonzero(after[~torch.from_numpy(masks)])), 0)

    def test_demonstration_learning_updates_recovery_and_preserves_base_and_critic(self):
        before_base = frozen_base_digest(self.model.policy)
        before_critic = copy.deepcopy(self.model.policy.value_net.state_dict())
        result = initialize_from_demonstrations(self.model, self.observations, self.masks,
            self.choices, SimpleNamespace(tick=lambda: None), maximum_updates=200)
        self.assertTrue(result['completed'])
        self.assertEqual(result['environment_steps'], 0)
        self.assertEqual(before_base, frozen_base_digest(self.model.policy))
        for key, value in before_critic.items():
            torch.testing.assert_close(value, self.model.policy.value_net.state_dict()[key], rtol=0, atol=0)
        chosen = self.model.predict(self.observations, deterministic=True, action_masks=self.masks)[0]
        np.testing.assert_array_equal(chosen, self.choices)

    def test_checkpoint_and_torchscript_preserve_learned_choices_and_contract(self):
        with torch.no_grad():
            self.model.policy.action_net.residual.bias[1] = 8.
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            self.model.save(folder/'policy.zip')
            restored = StopRecoveryPPO.load(folder/'policy.zip', device='cpu')
            np.testing.assert_array_equal(restored.predict(self.full, deterministic=True,
                action_masks=masks_from_observation(self.full[:, :282]))[0],
                self.model.predict(self.full, deterministic=True,
                action_masks=masks_from_observation(self.full[:, :282]))[0])
            source = PROJECT/'models/localization_mapping_20261015/ray_consistent_candidate/portable'
            export_recovery(restored, source, folder/'portable')
            manifest = json.loads((folder/'portable/manifest.json').read_text())
            self.assertFalse(manifest['navigation_preservation_contract']['navigation_actor_all_parameters_trainable'])
            actor = torch.jit.load(str(folder/'portable/actor.pt'))
            runtime_obs = np.concatenate([self.full[:, :300], np.zeros((len(self.full), 3), np.float32), self.full[:, 300:]], axis=-1)
            with torch.no_grad():
                command = actor(torch.from_numpy(runtime_obs)).numpy()
            expected = ACTIONS[restored.predict(self.full, deterministic=True,
                action_masks=masks_from_observation(self.full[:, :282]))[0]]
            np.testing.assert_array_equal(command, expected)

    def test_failed_or_tampered_episodes_are_not_demonstrations(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'RESULTS.json'
            report = json.loads(SOURCE.read_text())
            for row in report['rows']:
                if row['specification']['id'] == 'recover_forward':
                    row['safe_success'] = False
            path.write_text(json.dumps(report))
            with self.assertRaises(ValueError):
                load_demonstrations([dict(report=str(path), id='recover_forward')], self.model.policy)

    def test_actual_ppo_update_uses_recovery_support_and_keeps_base_frozen(self):
        observations, masks, choices = self.observations, self.masks, self.choices

        class TrainingInterface(Interface):
            def reset(self, *, seed=None, options=None):
                self.index = 0
                return observations[0].copy(), {}

            def action_masks(self):
                return masks[self.index].copy()

            def step(self, choice):
                reward = (1. if int(choice) == choices[self.index] else -1.)/60.
                self.index += 1
                done = self.index == len(observations)
                return observations[self.index % len(observations)].copy(), reward, done, False, {}

        model = warm_start_recovery(CHECKPOINT, TrainingInterface(), seed=17)
        before_base = frozen_base_digest(model.policy)
        before_residual = model.policy.action_net.residual.weight.detach().clone()
        model.learn(total_timesteps=250)
        self.assertEqual(model.num_timesteps, 250)
        self.assertEqual(before_base, frozen_base_digest(model.policy))
        self.assertFalse(torch.equal(before_residual, model.policy.action_net.residual.weight))
        for key in ['train/loss', 'train/approx_kl', 'train/actor_recovery_fraction']:
            self.assertTrue(np.isfinite(model.logger.name_to_value[key]), key)


if __name__ == '__main__':
    unittest.main()
