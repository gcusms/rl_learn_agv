import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from experiments.demonstration_retention import configure, loss
from experiments.recovery_demonstrations import load_demonstrations
from experiments.stop_feedback_navigation import FeedbackInterface
from experiments.stop_recovery_policy import StopRecoveryPPO, warm_start_recovery, export_recovery
from experiments.train_prefix_recovery import frozen_base_digest
from neural_navigation.decision_feedback import WAIT_SCHEMA, STAT_SLICE
from neural_navigation.masked_actions import ACTIONS
from navrl.geometry import ROOT


PROJECT = ROOT.parent
CHECKPOINT = PROJECT/'models/quality_recovery_20261014/policy_2000.zip'


class Interface(FeedbackInterface):
    feedback_schema = WAIT_SCHEMA


class DemonstrationRetentionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.initial = warm_start_recovery(CHECKPOINT, Interface())
        cls.obs, cls.masks, cls.choices, _ = load_demonstrations([dict(
            report=str(PROJECT/'reports/stop_feedback_20261015/turn_recovery_20261015/RESULTS.json'),
            id='recover_forward')], cls.initial.policy)

    def setUp(self):
        self.model = copy.deepcopy(self.initial)

    def configure(self, model=None, **kwargs):
        return configure(self.model if model is None else model,
            self.obs, self.masks, self.choices, **kwargs)

    def environment(self, supported=True):
        observations, masks, choices = self.obs.copy(), self.masks, self.choices
        if not supported:
            observations[:, 300+STAT_SLICE.start+7] = 0.

        class RecordedInterface(Interface):
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

        return RecordedInterface()

    def test_configuration_does_not_change_inference_or_global_sampling_rng(self):
        x = torch.from_numpy(self.obs)
        with torch.no_grad():
            before = self.model.policy.get_distribution(x, self.masks).distribution.probs.clone()
        torch_state, numpy_state = torch.get_rng_state(), np.random.get_state()
        specification = self.configure(seed=71)
        with torch.no_grad():
            after = self.model.policy.get_distribution(x, self.masks).distribution.probs.clone()
        torch.testing.assert_close(before, after, rtol=0, atol=0)
        torch.testing.assert_close(torch_state, torch.get_rng_state(), rtol=0, atol=0)
        current_numpy = np.random.get_state()
        self.assertEqual(numpy_state[0], current_numpy[0])
        np.testing.assert_array_equal(numpy_state[1], current_numpy[1])
        self.assertEqual(numpy_state[2:], current_numpy[2:])
        self.assertEqual(specification['environment_steps_added'], 0)
        self.assertFalse(specification['runtime_teacher'])

    def test_illegal_labels_masks_and_unsupported_examples_are_rejected(self):
        masks = self.masks.copy()
        masks[0, self.choices[0]] = False
        observations = self.obs.copy()
        observations[0, 300+STAT_SLICE.start+7] = 0.
        invalid = [(self.obs, masks, self.choices),
                   (observations, self.masks, self.choices),
                   (self.obs, self.masks, self.choices.astype(float)+.1),
                   (self.obs, self.masks.astype(int)+2, self.choices)]
        for args in invalid:
            with self.subTest(shapes=[a.shape for a in args]), self.assertRaises(ValueError):
                configure(self.model, *args)
        for kwargs in [dict(coefficient=float('nan')), dict(coefficient=.6), dict(batch_size=3)]:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.configure(**kwargs)

    def test_replay_loss_has_only_residual_actor_gradients(self):
        self.configure()
        self.model.policy.optimizer.zero_grad()
        value, count = loss(self.model)
        value.backward()
        self.assertTrue(torch.isfinite(value))
        self.assertEqual(count, 128)
        gradients = [(name, p.grad) for name, p in self.model.policy.named_parameters()
                     if p.grad is not None and torch.count_nonzero(p.grad)]
        self.assertTrue(gradients)
        for name, _ in gradients:
            self.assertTrue(name.startswith(('mlp_extractor.policy_net.residual.', 'action_net.residual.')), name)

    def test_save_reload_preserves_demonstrations_and_next_replay_sample(self):
        self.configure(seed=51)
        loss(self.model)  # Advance the independent replay generator once.
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'policy.zip'
            self.model.save(path)
            restored = StopRecoveryPPO.load(path, device='cpu')
            self.assertEqual(self.model.demonstration_retention_contract,
                             restored.demonstration_retention_contract)
            np.testing.assert_array_equal(self.model._demonstration_retention['observations'],
                                          restored._demonstration_retention['observations'])
            expected, n = loss(self.model)
            actual, restored_n = loss(restored)
            torch.testing.assert_close(expected, actual, rtol=0, atol=0)
            self.assertEqual(n, restored_n)

    def test_real_ppo_update_keeps_original_rollout_actions_likelihoods_and_step_count(self):
        model = warm_start_recovery(CHECKPOINT, self.environment(), seed=23)
        self.configure(model=model)
        initial = copy.deepcopy(model.policy)
        original_base = frozen_base_digest(model.policy)
        model.n_epochs = 1
        model.target_kl = None
        model.learn(total_timesteps=250)
        self.assertEqual(model.num_timesteps, 250)
        self.assertEqual(frozen_base_digest(model.policy), original_base)
        for key in ['train/demonstration_retention_loss', 'train/demonstration_retention_samples']:
            self.assertGreater(model.logger.name_to_value[key], 0., key)
        buffer = model.rollout_buffer
        observations = torch.from_numpy(buffer.observations.reshape(-1, 436))
        actions = torch.from_numpy(buffer.actions.reshape(-1)).long()
        masks = buffer.action_masks.reshape(-1, 12).astype(bool)
        with torch.no_grad():
            sampled_log = initial.get_distribution(observations, masks).log_prob(actions).numpy()
        np.testing.assert_allclose(sampled_log, buffer.log_probs.reshape(-1), rtol=0, atol=1e-5)
        self.assertTrue(masks[np.arange(250), actions.numpy()].all())

    def test_no_replay_or_actor_update_without_supported_live_samples(self):
        model = warm_start_recovery(CHECKPOINT, self.environment(supported=False), seed=24)
        self.configure(model=model)
        # Populate actor Adam momentum before verifying that an unsupported
        # live rollout cannot update it, even via residual optimizer state.
        value, _ = loss(model)
        model.policy.optimizer.zero_grad()
        value.backward()
        model.policy.optimizer.step()
        before = {name: p.detach().clone() for name, p in model.policy.named_parameters()
                  if '.residual.' in name}
        rng = copy.deepcopy(model._demonstration_retention['generator'].bit_generator.state)
        model.n_epochs = 1
        model.learn(total_timesteps=250)
        for name, expected in before.items():
            torch.testing.assert_close(expected, dict(model.policy.named_parameters())[name], rtol=0, atol=0)
        self.assertEqual(rng, model._demonstration_retention['generator'].bit_generator.state)
        self.assertEqual(model.logger.name_to_value['train/demonstration_retention_samples'], 0.)

    def test_runtime_export_remains_masked_model_inference(self):
        self.configure()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)/'portable'
            source = PROJECT/'models/localization_mapping_20261015/ray_consistent_candidate/portable'
            export_recovery(self.model, source, output)
            card = json.loads((output/'MODEL_CARD.json').read_text())
            self.assertEqual(card['training_demonstration_retention'],
                             self.model.demonstration_retention_contract)
            runtime_obs = np.concatenate([self.obs[:, :300], np.zeros((len(self.obs), 3), np.float32),
                                          self.obs[:, 300:]], axis=-1)
            actor = torch.jit.load(str(output/'actor.pt'))
            with torch.no_grad():
                actual = actor(torch.from_numpy(runtime_obs)).numpy()
            choices = self.model.predict(self.obs, deterministic=True, action_masks=self.masks)[0]
            np.testing.assert_array_equal(actual, ACTIONS[choices])


if __name__ == '__main__':
    unittest.main()
