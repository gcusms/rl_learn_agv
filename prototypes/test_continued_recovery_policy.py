import tempfile
import unittest
from pathlib import Path
import gymnasium as gym
import numpy as np
import torch
from experiments.focused_recovery_policy import recovery_support
from experiments.masked_navigation import PortableMaskedActor
from neural_navigation.masked_actions import ACTIONS, masks_from_observation
from prototypes.continued_recovery_policy import ContinuedRecoveryPolicy, continued_support


class ContinuedRecoveryTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(72)
        self.policy = ContinuedRecoveryPolicy(gym.spaces.Box(-1,1,(300,),np.float32),
            gym.spaces.Discrete(12), lambda _:3e-4)
        self.obs = torch.zeros(4,300)
        self.obs[:,235:244] = 1; self.obs[:,267:269] = 1
        self.obs[:,250] = 1; self.obs[:,294] = 1; self.obs[:,229] = 1
        self.obs[:,232] = -.191687; self.obs[:,216] = .2; self.obs[:,248] = 1
        with torch.no_grad():
            self.policy.action_net.base.weight.zero_()
            self.policy.action_net.base.bias.zero_()
            self.policy.action_net.base.bias[7] = 1

    def logits(self, obs):
        return self.policy.action_net(self.policy.mlp_extractor.policy_net(obs))

    def test_continuation_crosses_old_boundary_and_ends_after_alignment(self):
        self.assertFalse(bool(recovery_support(self.obs).any()))
        self.assertTrue(bool(continued_support(self.obs).all()))
        aligned = self.obs.clone(); aligned[:,232] = .8
        self.assertFalse(bool(continued_support(aligned).any()))

    def test_continuation_needs_actual_pure_turn_trusted_pose_and_legal_choice(self):
        for index, value in [(247,.05),(248,0),(294,0),(229,.1),(216,0),(264,1)]:
            obs = self.obs.clone(); obs[:,index] = value
            self.assertFalse(bool(continued_support(obs).any()))
        obs = self.obs.clone(); obs[:,235:244] = 0; obs[:,267:269] = 0
        self.assertFalse(bool(continued_support(obs).any()))
        obs = self.obs.clone(); obs[:,248] = 0; obs[:,246] = 1
        self.assertFalse(bool(continued_support(obs).any()))

    def test_actual_stop_remains_trainable_with_low_pose_quality(self):
        obs = self.obs.clone(); obs[:,294] = 0; obs[:,250] = 0; obs[:,255] = 1
        self.assertTrue(bool(continued_support(obs).all()))

    def test_real_gradient_changes_continuation_but_keeps_ordinary_base_exact(self):
        ordinary = self.obs.clone(); ordinary[:,248] = 0; ordinary[:,232] = .8
        baseline = self.logits(ordinary).detach().clone()
        before = self.logits(self.obs).detach().clone()
        frozen = {n:p.clone() for n,p in self.policy.named_parameters() if '.base.' in n}
        self.policy.optimizer.zero_grad()
        (-self.logits(self.obs)[:,4].mean()).backward(); self.policy.optimizer.step()
        self.assertTrue(bool((self.logits(self.obs)[:,4] > before[:,4]).all()))
        torch.testing.assert_close(self.logits(ordinary),baseline,atol=0,rtol=0)
        for n,p in self.policy.named_parameters():
            if n in frozen: torch.testing.assert_close(p,frozen[n],atol=0,rtol=0)

    def test_native_mask_export_and_reload_agree(self):
        with torch.no_grad(): self.policy.action_net.residual.bias[4] = 6
        obs = self.obs.clone(); obs[1,232] = .8
        obs[2,235:244] = 0; obs[2,267:269] = 0
        masks = masks_from_observation(obs.numpy()[:,:282])
        choices = self.policy.get_distribution(obs,action_masks=masks).mode()
        self.assertEqual(choices.tolist(),[4,7,0,4])
        actor = torch.jit.trace(PortableMaskedActor(self.policy),obs)
        with torch.no_grad(): actual = actor(obs).numpy()
        np.testing.assert_array_equal(actual,ACTIONS[choices.numpy()])
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/'policy.pt'; self.policy.save(path)
            loaded = ContinuedRecoveryPolicy.load(path,device='cpu')
            torch.testing.assert_close(loaded.get_distribution(obs,action_masks=masks).mode(),choices)


if __name__ == '__main__': unittest.main()
