"""Training-only checks; recorded observations are not navigation episodes."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from experiments.long_wait_exploration import configure, loss
from experiments.stop_feedback_navigation import FeedbackInterface
from experiments.stop_recovery_policy import StopRecoveryPPO, export_recovery
from experiments.train_prefix_recovery import frozen_base_digest
from neural_navigation.decision_feedback import STAT_SLICE, WAIT_SCHEMA
from neural_navigation.masked_actions import ACTIONS
from navrl.geometry import ROOT


PROJECT = ROOT.parent
CHECKPOINT = PROJECT/'models/stop_feedback_20261015/w11_recovery_bc800_s231015/policy_bc.zip'
TRACE = PROJECT/'reports/stop_feedback_20261015/proactive_augmented_ppo8000_w11_20261016/w11_training_stall_t106.npz'


class Interface(FeedbackInterface):
    feedback_schema = WAIT_SCHEMA


class LongWaitExplorationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.initial = StopRecoveryPPO.load(CHECKPOINT, device='cpu')
        with np.load(TRACE, allow_pickle=False) as data:
            # At the release itself the observable wait timer is still short.
            # Select a later actually recorded frame, after two seconds idle.
            index = int(np.argmin(abs(data['labels'][:, 0]-108.2)))
            assert abs(data['labels'][index, 0]-108.2) < 1e-7
            raw = data['observations'][index]
            cls.obs = np.r_[raw[:300], raw[303:]].astype(np.float32)[None]
            cls.mask = data['masks'][index:index+1].astype(bool)
        assert np.flatnonzero(cls.mask[0]).tolist() == [0, 9]
        assert cls.initial.policy.recovery_support(torch.from_numpy(cls.obs)).all()
        assert cls.obs[0, 300+STAT_SLICE.start+3]*5. >= 2.

    def setUp(self):
        self.model = copy.deepcopy(self.initial)

    def environment(self, supported=True):
        observation = self.obs[0].copy()
        mask = self.mask[0].copy()
        if not supported:
            observation[300+STAT_SLICE.start+7] = 0.

        class RecordedInterface(Interface):
            def reset(self, *, seed=None, options=None):
                self.index = 0
                return observation.copy(), {}

            def action_masks(self):
                return mask.copy()

            def step(self, choice):
                self.index += 1
                return observation.copy(), (.01 if int(choice) == 9 else -.001), self.index == 25, False, {}

        return RecordedInterface()

    def test_configuration_preserves_inference_rng_and_step_count(self):
        x = torch.from_numpy(self.obs)
        with torch.no_grad():
            before = self.model.policy.get_distribution(x, self.mask).distribution.probs.clone()
        torch_state, numpy_state = torch.get_rng_state(), np.random.get_state()
        steps = self.model.num_timesteps
        specification = configure(self.model)
        with torch.no_grad():
            after = self.model.policy.get_distribution(x, self.mask).distribution.probs.clone()
        torch.testing.assert_close(before, after, rtol=0, atol=0)
        torch.testing.assert_close(torch_state, torch.get_rng_state(), rtol=0, atol=0)
        current = np.random.get_state()
        self.assertEqual(numpy_state[0], current[0])
        np.testing.assert_array_equal(numpy_state[1], current[1])
        self.assertEqual(numpy_state[2:], current[2:])
        self.assertEqual(steps, self.model.num_timesteps)
        self.assertEqual(specification['environment_steps_added'], 0)
        self.assertFalse(specification['runtime_teacher'])
        self.assertFalse(specification['runtime_rule_added'])
        self.assertTrue(specification['includes_stop'])

    def test_saturated_legal_choices_have_finite_actor_only_gradient(self):
        configure(self.model)
        self.model.policy.optimizer.zero_grad()
        value, count = loss(self.model, torch.from_numpy(self.obs), self.mask)
        value.backward()
        self.assertTrue(torch.isfinite(value))
        self.assertEqual(count, 1)
        bias = self.model.policy.action_net.residual.bias.grad
        self.assertGreater(float(bias[0]), .49)
        self.assertLess(float(bias[9]), -.49)
        # Categorical normalizes before and after applying the mask; float32
        # cancellation can leave rounding-scale gradients on excluded logits.
        torch.testing.assert_close(bias[~torch.from_numpy(self.mask[0])], torch.zeros(10), rtol=0, atol=1e-7)
        with torch.no_grad():
            probabilities = self.model.policy.get_distribution(torch.from_numpy(self.obs), self.mask).distribution.probs
        torch.testing.assert_close(probabilities[0, ~torch.from_numpy(self.mask[0])], torch.zeros(10), rtol=0, atol=0)
        gradients = [name for name, p in self.model.policy.named_parameters()
                     if p.grad is not None and torch.count_nonzero(p.grad)]
        self.assertTrue(gradients)
        for name in gradients:
            self.assertTrue(name.startswith(('mlp_extractor.policy_net.residual.', 'action_net.residual.')), name)

    def test_short_wait_only_stop_invalid_inputs_and_close_goal_are_excluded(self):
        configure(self.model)
        for mode in ('short_wait', 'only_stop', 'invalid_scan', 'invalid_forecast', 'close_goal'):
            obs, masks = self.obs.copy(), self.mask.copy()
            if mode == 'short_wait':
                obs[0, 300+STAT_SLICE.start+3] = .2
            elif mode == 'only_stop':
                masks[0, 1:] = False
                obs[0, 235:244] = 0.
                obs[0, 267:269] = 0.
            elif mode == 'invalid_scan':
                obs[0, 300+STAT_SLICE.start+7] = 0.
            elif mode == 'invalid_forecast':
                obs[0, 300+STAT_SLICE.start+4] = 0.
            else:
                obs[0, 218:220] = 0.
            with self.subTest(mode=mode):
                value, count = loss(self.model, torch.from_numpy(obs), masks)
                self.assertEqual(count, 0)
                self.assertEqual(float(value), 0.)

    def test_already_exploring_policy_is_excluded(self):
        configure(self.model)
        with torch.no_grad():
            self.model.policy.action_net.residual.bias[9] += 30.
        value, count = loss(self.model, torch.from_numpy(self.obs), self.mask)
        self.assertEqual((float(value), count), (0., 0))

    def test_invalid_contract_observations_or_original_masks_are_rejected(self):
        for kwargs in (dict(coefficient=0.), dict(coefficient=.011), dict(coefficient=float('nan')),
                       dict(minimum_wait_seconds=.5), dict(minimum_wait_seconds=float('inf'))):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                configure(self.model, **kwargs)
        configure(self.model)
        invalid_obs = self.obs.copy()
        invalid_obs[0, 0] = np.nan
        invalid_masks = self.mask.copy()
        invalid_masks[0, 4] = True
        for obs, masks in ((invalid_obs, self.mask), (self.obs[:, :-1], self.mask),
                           (self.obs, invalid_masks), (self.obs, self.mask.astype(int)+2)):
            with self.subTest(shape=obs.shape), self.assertRaises(ValueError):
                loss(self.model, torch.from_numpy(obs), masks)

    def test_save_reload_preserves_training_contract_and_loss(self):
        configure(self.model)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'policy.zip'
            self.model.save(path)
            restored = StopRecoveryPPO.load(path, device='cpu')
            self.assertEqual(self.model.long_wait_exploration_contract, restored.long_wait_exploration_contract)
            expected, count = loss(self.model, torch.from_numpy(self.obs), self.mask)
            actual, restored_count = loss(restored, torch.from_numpy(self.obs), self.mask)
            torch.testing.assert_close(expected, actual, rtol=0, atol=0)
            self.assertEqual(count, restored_count)

    def test_recorded_observation_ppo_update_keeps_original_likelihoods_actions_and_masks(self):
        model = self.model
        model.set_env(self.environment())
        configure(model, coefficient=.01)
        initial = copy.deepcopy(model.policy)
        base = frozen_base_digest(model.policy)
        model.n_epochs, model.target_kl = 1, None
        model.learn(total_timesteps=250)
        self.assertEqual(model.num_timesteps, 250)
        self.assertEqual(frozen_base_digest(model.policy), base)
        self.assertGreater(model.logger.name_to_value['train/long_wait_exploration_loss'], 0.)
        self.assertGreater(model.logger.name_to_value['train/long_wait_exploration_samples'], 0.)
        buffer = model.rollout_buffer
        observations = torch.from_numpy(buffer.observations.reshape(-1, 436))
        actions = torch.from_numpy(buffer.actions.reshape(-1)).long()
        masks = buffer.action_masks.reshape(-1, 12).astype(bool)
        with torch.no_grad():
            expected_log = initial.get_distribution(observations, masks).log_prob(actions).numpy()
            old_move = initial.get_distribution(torch.from_numpy(self.obs), self.mask).distribution.probs[0, 9]
            new_move = model.policy.get_distribution(torch.from_numpy(self.obs), self.mask).distribution.probs[0, 9]
        np.testing.assert_allclose(expected_log, buffer.log_probs.reshape(-1), rtol=0, atol=1e-5)
        np.testing.assert_array_equal(masks, np.broadcast_to(self.mask, (250, 12)))
        self.assertTrue(masks[np.arange(250), actions.numpy()].all())
        self.assertGreater(float(new_move), float(old_move))

    def test_no_actor_update_on_unsupported_rollout_even_with_adam_momentum(self):
        model = self.model
        configure(model)
        value, _ = loss(model, torch.from_numpy(self.obs), self.mask)
        model.policy.optimizer.zero_grad()
        value.backward()
        model.policy.optimizer.step()
        before = {name: p.detach().clone() for name, p in model.policy.named_parameters() if '.residual.' in name}
        model.set_env(self.environment(supported=False))
        model.n_epochs, model.target_kl = 1, None
        model.learn(total_timesteps=250)
        for name, expected in before.items():
            torch.testing.assert_close(expected, dict(model.policy.named_parameters())[name], rtol=0, atol=0)
        self.assertEqual(model.logger.name_to_value['train/long_wait_exploration_samples'], 0.)

    def test_export_has_training_metadata_and_same_masked_commands(self):
        configure(self.model)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)/'portable'
            source = PROJECT/'models/localization_mapping_20261015/ray_consistent_candidate/portable'
            export_recovery(self.model, source, output)
            card = json.loads((output/'MODEL_CARD.json').read_text())
            self.assertEqual(card['training_long_wait_exploration'], self.model.long_wait_exploration_contract)
            actor = torch.jit.load(str(output/'actor.pt'))
            runtime_obs = np.concatenate([self.obs[:, :300], np.zeros((1, 3), np.float32), self.obs[:, 300:]], axis=-1)
            with torch.no_grad():
                actual = actor(torch.from_numpy(runtime_obs)).numpy()
            choices = self.model.predict(self.obs, deterministic=True, action_masks=self.mask)[0]
            np.testing.assert_array_equal(actual, ACTIONS[choices])


if __name__ == '__main__':
    unittest.main()
