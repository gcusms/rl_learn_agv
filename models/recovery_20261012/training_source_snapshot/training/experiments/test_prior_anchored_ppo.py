import tempfile
from pathlib import Path
import unittest
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
from stable_baselines3 import PPO
from prior_anchored_ppo import PriorAnchoredPPO
from neural_navigation.safety_context import BASE_DIM, OBSERVATION_DIM, REASON_START


class TinyNavigation(gym.Env):
    observation_space = spaces.Box(-1, 1, (OBSERVATION_DIM,), dtype=np.float32)
    action_space = spaces.Box(-1, 1, (2,), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0
        return np.zeros(OBSERVATION_DIM, np.float32), {}

    def step(self, action):
        self.steps += 1
        return np.zeros(OBSERVATION_DIM, np.float32), float(action[0]), self.steps == 4, False, {}


class PriorAnchorTest(unittest.TestCase):
    def model(self, cls=PriorAnchoredPPO, **kwargs):
        torch.set_num_threads(1)
        return cls('MlpPolicy', TinyNavigation(), n_steps=8, batch_size=8, n_epochs=1,
                   seed=123, device='cpu', policy_kwargs=dict(net_arch=dict(pi=[8], vf=[8])), **kwargs)

    def healthy_observations(self):
        obs = torch.zeros((4, OBSERVATION_DIM))
        obs[:, BASE_DIM:BASE_DIM+9] = 1
        obs[:, BASE_DIM+REASON_START] = 1
        obs[:, 232] = 1
        obs[:, 234] = 1
        return obs

    def test_reference_gradient_and_recovery_exemptions(self):
        model = self.model(anchor_strength=.1)
        with torch.no_grad():
            model.policy.action_net.weight.zero_()
            model.policy.action_net.bias.copy_(torch.tensor([.5, 0.]))
        model.snapshot_reference()
        obs = self.healthy_observations()
        self.assertEqual(model.reference_loss(obs)[0].item(), 0.)
        with torch.no_grad():
            model.policy.action_net.bias[0] += .3
        loss, fraction = model.reference_loss(obs)
        self.assertAlmostEqual(loss.item(), .25, places=6)
        self.assertEqual(fraction.item(), 1.)
        loss.backward()
        self.assertGreater(model.policy.action_net.bias.grad[0].item(), 0.)
        self.assertTrue(all(p.grad is None and not p.requires_grad for p in model.reference_policy.parameters()))
        for column in [BASE_DIM, BASE_DIM+REASON_START, 232, 234]:
            changed = obs.clone()
            changed[:, column] = 0
            self.assertEqual(model.reference_loss(changed)[0].item(), 0.)
        changed = obs.clone()
        changed[:, -1] = .2
        self.assertEqual(model.reference_loss(changed)[0].item(), 0.)

    def test_disabled_anchor_matches_sb3_updates_exactly(self):
        states = []
        for cls in [PPO, PriorAnchoredPPO]:
            model = self.model(cls)
            model.learn(16)
            states.append({key: value.clone() for key, value in model.policy.state_dict().items()})
        for key in states[0]:
            torch.testing.assert_close(states[0][key], states[1][key], rtol=0, atol=0)

    def test_standard_ppo_load_keeps_inference_and_excludes_reference(self):
        model = self.model(anchor_strength=.1)
        model.snapshot_reference()
        obs = self.healthy_observations().numpy()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'policy.zip'
            model.save(path)
            restored = PPO.load(path, device='cpu')
            np.testing.assert_array_equal(model.predict(obs, deterministic=True)[0],
                                          restored.predict(obs, deterministic=True)[0])
            self.assertFalse(hasattr(restored, 'reference_policy'))

    def test_enabled_but_exempt_rollouts_match_sb3_and_keep_reference_frozen(self):
        normal = self.model(PPO)
        normal.learn(16)
        anchored = self.model(anchor_strength=.1)
        anchored.snapshot_reference()
        before = {key: value.clone() for key, value in anchored.reference_policy.state_dict().items()}
        anchored.learn(16)
        self.assertEqual(anchored.last_anchor_statistics['eligible_state_fraction'], 0.)
        for key, value in normal.policy.state_dict().items():
            torch.testing.assert_close(value, anchored.policy.state_dict()[key], rtol=0, atol=0)
        for key, value in before.items():
            torch.testing.assert_close(value, anchored.reference_policy.state_dict()[key], rtol=0, atol=0)

    def test_real_checkpoint_load_retains_weights_and_optimizer(self):
        from navrl.geometry import ROOT
        torch.set_num_threads(1)
        path = ROOT/'runs/safety_context_20261011/recovery_probes_v3_16000/candidate/policy.zip'
        original = PPO.load(path, device='cpu')
        anchored = PriorAnchoredPPO.load(path, device='cpu', anchor_strength=.1, anchor_scale=.3)
        self.assertEqual(anchored.anchor_strength, .1)
        anchored.snapshot_reference()
        obs = np.random.default_rng(29).uniform(-1, 1, (32, OBSERVATION_DIM)).astype(np.float32)
        np.testing.assert_array_equal(original.predict(obs, deterministic=True)[0],
                                      anchored.predict(obs, deterministic=True)[0])
        torch.testing.assert_close(original.policy.optimizer.state_dict(),
                                   anchored.policy.optimizer.state_dict(), rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
