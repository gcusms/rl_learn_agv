import unittest
import torch
import tempfile
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from experiments.feasibility_feedback_ppo import feasibility_loss, FeasibilityFeedbackPPO
from neural_navigation.safety_context import BASE_DIM, OBSERVATION_DIM, REASON_START, REASONS


class FeedbackLossTests(unittest.TestCase):
    def observation(self, reason='laser_braking_footprint'):
        obs = torch.zeros((1, OBSERVATION_DIM))
        obs[:, BASE_DIM] = 1.  # straight forward accepted
        obs[:, BASE_DIM+14] = 1.  # previous action overridden
        obs[:, BASE_DIM+REASON_START+REASONS.index(reason)] = 1.
        obs[:, -1] = 1.
        return obs

    def test_rejected_turn_gets_gradient_toward_accepted_forward(self):
        mean = torch.tensor([[.025, -.94]], requires_grad=True)
        loss, fraction = feasibility_loss(self.observation(), mean)
        self.assertEqual(fraction.item(), 1.)
        loss.backward()
        self.assertLess(mean.grad[0, 0].item(), 0.)
        self.assertLess(mean.grad[0, 1].item(), 0.)

    def test_faults_waiting_no_alternative_and_executed_turn_exempt(self):
        observations = [self.observation(x) for x in
            ['scan_unavailable', 'localization_uncertain', 'learned_route_incomplete', 'none']]
        none = self.observation(); none[:, BASE_DIM:BASE_DIM+9] = 0.; observations.append(none)
        voluntary = self.observation(); voluntary[:, BASE_DIM+14] = 0.; observations.append(voluntary)
        turning = self.observation(); turning[:, BASE_DIM+13] = .5; observations.append(turning)
        brief = self.observation(); brief[:, -1] = .1; observations.append(brief)
        obs = torch.cat(observations)
        mean = torch.zeros((len(obs), 2), requires_grad=True)
        loss, fraction = feasibility_loss(obs, mean)
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(loss.item(), 0.)
        self.assertEqual(fraction.item(), 0.)
        loss.backward(); self.assertEqual(torch.count_nonzero(mean.grad).item(), 0)

    def test_accepted_probe_has_zero_loss(self):
        mean = torch.tensor([[1., 0.]], requires_grad=True)
        loss, _ = feasibility_loss(self.observation(), mean)
        self.assertEqual(loss.item(), 0.)
        loss.backward(); self.assertEqual(torch.count_nonzero(mean.grad).item(), 0)

    def test_clipped_out_of_bounds_mean_keeps_corrective_gradient(self):
        mean = torch.tensor([[-2., -3.]], requires_grad=True)
        loss, _ = feasibility_loss(self.observation(), mean)
        loss.backward()
        self.assertTrue(torch.isfinite(mean.grad).all())
        self.assertTrue((mean.grad < 0).all())

    def test_only_accepted_probes_attract_proposal(self):
        obs = self.observation(); obs[:, BASE_DIM] = 0.; obs[:, BASE_DIM+3] = 1.
        mean = torch.tensor([[0., -1.]], requires_grad=True)
        loss, _ = feasibility_loss(obs, mean)
        loss.backward()
        self.assertEqual(mean.grad[0, 0].item(), 0.)
        self.assertLess(mean.grad[0, 1].item(), 0.)

    def test_optimizer_and_plain_ppo_export_without_reference_actor(self):
        observation = self.observation()[0].numpy()
        class FeedbackEnv(gym.Env):
            observation_space = gym.spaces.Box(-1., 1., (OBSERVATION_DIM,), dtype=np.float32)
            action_space = gym.spaces.Box(-1., 1., (2,), dtype=np.float32)
            def reset(self, *, seed=None, options=None):
                super().reset(seed=seed); return observation.copy(), {}
            def step(self, action):
                return observation.copy(), 0., True, False, {}
        torch.set_num_threads(1)
        model = FeasibilityFeedbackPPO('MlpPolicy', FeedbackEnv(), anchor_strength=.1,
            n_steps=8, batch_size=8, n_epochs=1, learning_rate=.01, seed=821, device='cpu',
            policy_kwargs=dict(net_arch=dict(pi=[8], vf=[8])))
        before = model.reference_loss(torch.tensor(observation[None]))[0].item()
        model.learn(8)
        after = model.reference_loss(torch.tensor(observation[None]))[0].item()
        self.assertLess(after, before)
        self.assertEqual(model.last_feedback_statistics['eligible_state_fraction'], 1.)
        self.assertIsNone(model.reference_policy)
        with tempfile.TemporaryDirectory() as path:
            model.save(path+'/policy.zip')
            restored = PPO.load(path+'/policy.zip', device='cpu')
            np.testing.assert_array_equal(model.predict(observation, deterministic=True)[0],
                restored.predict(observation, deterministic=True)[0])


if __name__ == '__main__':
    unittest.main()
