import unittest
import gymnasium as gym
import numpy as np
import torch
from experiments.soft_prior_recovery_policy import SoftPriorRecoveryPolicy
from experiments.masked_navigation import PortableMaskedActor
from neural_navigation.masked_actions import ACTIONS,masks_from_observation


class SoftPriorTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(73);torch.set_num_threads(1)
        self.policy=SoftPriorRecoveryPolicy(gym.spaces.Box(-1,1,(300,),np.float32),
            gym.spaces.Discrete(12),lambda _:3e-4)
        self.obs=torch.zeros(16,300)
        self.obs[:,235:244]=1;self.obs[:,267:269]=1
        self.obs[:,250]=1;self.obs[:,294]=1;self.obs[:,229]=1
        self.obs[:,232]=.7;self.obs[:,216]=.2
        with torch.no_grad():
            self.policy.action_net.base.weight.zero_();self.policy.action_net.base.bias.zero_()
            self.policy.action_net.base.bias[6]=1;self.policy.action_net.base.bias[5]=-24

    def test_only_recovery_sampling_is_broadened_and_initial_mode_retained(self):
        masks=np.ones((16,12),bool)
        clear=self.policy.get_distribution(self.obs,action_masks=masks)
        before=clear.distribution.probs.clone();mode=clear.mode()
        recovering=self.obs.clone();recovering[:,232]=-.3
        after=self.policy.get_distribution(recovering,action_masks=masks)
        self.assertTrue(bool((after.distribution.probs[:,5]>before[:,5]*1000000).all()))
        torch.testing.assert_close(mode,after.mode(),rtol=0,atol=0)
        ordinary=self.policy.get_distribution(self.obs,action_masks=masks)
        torch.testing.assert_close(before,ordinary.distribution.probs,rtol=0,atol=0)

    def test_sampling_likelihood_matches_ppo_evaluation_and_masks(self):
        obs=self.obs.clone();obs[:,232]=-.3
        masks=np.ones((16,12),bool);masks[:,2]=False;masks[:8,1:]=False
        with torch.no_grad():
            actions,_,old_logprob=self.policy(obs,action_masks=masks)
            _,new_logprob,_=self.policy.evaluate_actions(obs,actions,action_masks=masks)
        self.assertTrue(bool(torch.tensor(masks)[torch.arange(16),actions].all()))
        torch.testing.assert_close(old_logprob,new_logprob,rtol=0,atol=0)
        self.assertTrue(bool((actions[:8]==0).all()))

    def test_portable_contains_scaled_prior_and_learned_residual(self):
        obs=self.obs.clone();obs[:8,232]=-.3
        with torch.no_grad():self.policy.action_net.residual.bias[4]=.5
        masks=masks_from_observation(obs.numpy()[:,:282])
        expected=self.policy.get_distribution(obs,action_masks=masks).mode()
        self.assertTrue(bool((expected[:8]==4).all()));self.assertTrue(bool((expected[8:]==6).all()))
        actor=torch.jit.trace(PortableMaskedActor(self.policy),obs)
        torch.testing.assert_close(actor(obs),torch.from_numpy(ACTIONS)[expected],rtol=0,atol=0)


if __name__=='__main__':unittest.main()
