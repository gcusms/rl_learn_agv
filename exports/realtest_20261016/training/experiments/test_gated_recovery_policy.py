import copy
import unittest
import gymnasium as gym
import numpy as np
import torch
from experiments.gated_recovery_policy import GatedRecoveryPolicy,GatedRecoveryPPO,sensor_clear
from experiments.masked_navigation import PortableMaskedActor
from neural_navigation.masked_actions import ACTIONS,masks_from_observation
from experiments.test_alignment_reward import SampleEnv
from stable_baselines3.common.logger import configure


class GatedRecoveryTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(71);torch.set_num_threads(1)
        self.policy=GatedRecoveryPolicy(gym.spaces.Box(-1,1,(300,),np.float32),
            gym.spaces.Discrete(12),lambda _:3e-5)
        self.observations=torch.zeros(16,300)
        for index in list(range(235,244))+[267,268,250,294]:self.observations[:,index]=1
        self.observations[:,234]=.6;self.observations[:,232]=.8
        with torch.no_grad():
            self.policy.action_net.base.weight.zero_();self.policy.action_net.base.bias.zero_()
            self.policy.action_net.base.bias[1]=1

    def logits(self,obs):return self.policy.action_net(self.policy.mlp_extractor.policy_net(obs))

    def test_zero_head_exact_initial_base_logits(self):
        obs=torch.randn(256,300)
        expected=self.policy.action_net.base(self.policy.mlp_extractor.policy_net.base(obs))
        torch.testing.assert_close(self.logits(obs),expected,rtol=0,atol=0)

    def test_learning_changes_recovery_not_protected_states_or_base(self):
        clear=self.observations;self.assertTrue(bool(sensor_clear(clear).all()))
        recovery=clear.clone();recovery[:,232]=-.5
        baseline=self.logits(clear).detach().clone()
        frozen={k:v.clone() for k,v in self.policy.state_dict().items() if '.base.' in k}
        self.policy.optimizer.zero_grad();loss=-self.logits(recovery)[:,4].mean();loss.backward();self.policy.optimizer.step()
        torch.testing.assert_close(self.logits(clear),baseline,rtol=0,atol=0)
        self.assertTrue(bool((self.logits(recovery)[:,4]>baseline[:,4]).all()))
        for name,value in frozen.items():torch.testing.assert_close(self.policy.state_dict()[name],value,rtol=0,atol=0)
        self.assertTrue(all(not p.requires_grad for n,p in self.policy.named_parameters() if '.base.' in n))

    def test_clear_base_stop_remains_trainable(self):
        with torch.no_grad():
            self.policy.action_net.base.bias[0]=2
            self.policy.action_net.residual.bias[4]=4
        self.assertTrue(bool((self.logits(self.observations).argmax(-1)==4).all()))

    def test_portable_includes_residual_and_native_masks(self):
        with torch.no_grad():self.policy.action_net.residual.bias[4]=4
        obs=self.observations.clone();obs[:8,232]=-.5;obs[8:,235:244]=0;obs[8:,267:269]=0
        masks=masks_from_observation(obs.numpy()[:,:282])
        expected=self.policy.get_distribution(obs,action_masks=masks).mode()
        self.assertTrue(bool((expected[:8]==4).all()));self.assertTrue(bool((expected[8:]==0).all()))
        actor=torch.jit.trace(PortableMaskedActor(self.policy),obs)
        torch.testing.assert_close(actor(obs),torch.from_numpy(ACTIONS)[expected],rtol=0,atol=0)

    def test_critic_rate_is_retained_across_repeated_schedule_updates(self):
        model=GatedRecoveryPPO(GatedRecoveryPolicy,SampleEnv(),learning_rate=3e-4,n_steps=8,batch_size=8)
        model.set_logger(configure(format_strings=[]))
        for _ in range(3):
            model._update_learning_rate(model.policy.optimizer)
            rates={group['role']:group['lr'] for group in model.policy.optimizer.param_groups}
            self.assertAlmostEqual(rates['recovery'],3e-4)
            self.assertAlmostEqual(rates['critic'],3e-5)


if __name__=='__main__':unittest.main()
