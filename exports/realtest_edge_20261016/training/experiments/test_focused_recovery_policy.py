import unittest
import gymnasium as gym
import numpy as np
import torch
from experiments.focused_recovery_policy import FocusedRecoveryPolicy,recovery_support
from experiments.masked_navigation import PortableMaskedActor
from neural_navigation.masked_actions import ACTIONS,masks_from_observation


class FocusedRecoveryTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(72);torch.set_num_threads(1)
        self.policy=FocusedRecoveryPolicy(gym.spaces.Box(-1,1,(300,),np.float32),
            gym.spaces.Discrete(12),lambda _:3e-4)
        self.obs=torch.zeros(16,300)
        self.obs[:,235:244]=1;self.obs[:,267:269]=1
        self.obs[:,250]=1;self.obs[:,294]=1;self.obs[:,229]=1
        self.obs[:,232]=.7;self.obs[:,216]=.2;self.obs[:,295]=1
        with torch.no_grad():
            self.policy.action_net.base.weight.zero_();self.policy.action_net.base.bias.zero_()
            self.policy.action_net.base.bias[7]=1

    def logits(self,obs):return self.policy.action_net(self.policy.mlp_extractor.policy_net(obs))

    def test_normal_dynamic_driving_remains_unchanged_after_recovery_gradient(self):
        self.assertFalse(bool(recovery_support(self.obs).any()))
        baseline=self.logits(self.obs).detach().clone()
        recovering=self.obs.clone();recovering[:,232]=-.3
        self.assertTrue(bool(recovery_support(recovering).all()))
        frozen={n:p.detach().clone() for n,p in self.policy.named_parameters() if '.base.' in n}
        self.policy.optimizer.zero_grad();(-self.logits(recovering)[:,4].mean()).backward();self.policy.optimizer.step()
        torch.testing.assert_close(self.logits(self.obs),baseline,rtol=0,atol=0)
        self.assertTrue(bool((self.logits(recovering)[:,4]>baseline[:,4]).all()))
        for n,p in self.policy.named_parameters():
            if n in frozen:torch.testing.assert_close(p,frozen[n],rtol=0,atol=0)

    def test_previous_stop_and_wait_remain_trainable_with_unknown_pose(self):
        for index,value in [(250,0),(266,.04)]:
            obs=self.obs.clone();obs[:,index]=value
            self.assertTrue(bool(recovery_support(obs).all()))
            obs[:,294]=0;self.assertTrue(bool(recovery_support(obs).all()))
        obs=self.obs.clone();obs[:,250]=0;obs[:,264]=1
        self.assertFalse(bool(recovery_support(obs).any()))
        obs[:,264]=0;obs[:,235:244]=0;obs[:,267:269]=0
        self.assertFalse(bool(recovery_support(obs).any()))

    def test_goal_turn_support_requires_trusted_nontrivial_route(self):
        obs=self.obs.clone();obs[:,232]=-.3
        self.assertTrue(bool(recovery_support(obs).all()))
        for index,value in [(229,.1),(216,0),(294,0)]:
            altered=obs.clone();altered[:,index]=value
            self.assertFalse(bool(recovery_support(altered).any()))

    def test_portable_uses_focused_gate_and_native_masks(self):
        with torch.no_grad():self.policy.action_net.residual.bias[4]=4
        obs=self.obs.clone();obs[:8,232]=-.3
        masks=masks_from_observation(obs.numpy()[:,:282])
        expected=self.policy.get_distribution(obs,action_masks=masks).mode()
        self.assertTrue(bool((expected[:8]==4).all()));self.assertTrue(bool((expected[8:]==7).all()))
        actor=torch.jit.trace(PortableMaskedActor(self.policy),obs)
        torch.testing.assert_close(actor(obs),torch.from_numpy(ACTIONS)[expected],rtol=0,atol=0)


if __name__=='__main__':unittest.main()
