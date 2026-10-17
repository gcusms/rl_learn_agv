import copy
import unittest
import gymnasium as gym
import numpy as np
import torch
from experiments.recovery_temperature_policy import RecoveryTemperaturePolicy
from experiments.repairable_recovery_ppo import repair_rank_drift
from experiments.check_recovery_temperature_migration import identical


class RankRepairTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1);torch.manual_seed(71)
        self.policy=RecoveryTemperaturePolicy(gym.spaces.Box(-1,1,(300,),np.float32),
            gym.spaces.Discrete(12),lambda _:3e-5)
        with torch.no_grad():
            self.policy.action_net.weight.zero_();self.policy.action_net.bias.fill_(-5.)
            self.policy.action_net.bias[1]=.1;self.policy.action_net.bias[11]=0.
        self.teacher=copy.deepcopy(self.policy);self.teacher.requires_grad_(False)
        with torch.no_grad():self.policy.action_net.bias[11]=.2
        self.observations=torch.zeros(16,300);self.masks=torch.ones(16,12,dtype=torch.bool)

    def test_repairs_preexisting_flip_without_touching_critic_or_adam(self):
        critic={k:v.clone() for k,v in self.policy.state_dict().items() if 'value' in k}
        optimizer=copy.deepcopy(self.policy.optimizer.state_dict())
        result=repair_rank_drift(self.policy,self.teacher,self.observations,self.masks)
        self.assertTrue(result['repaired']);self.assertGreater(result['steps'],0)
        choices=self.policy.get_distribution(self.observations,action_masks=self.masks).mode()
        self.assertTrue(bool((choices==1).all()))
        identical(critic,{k:v for k,v in self.policy.state_dict().items() if 'value' in k})
        identical(optimizer,self.policy.optimizer.state_dict())

    def test_failed_repair_restores_weights_and_reference_remains_frozen(self):
        original=copy.deepcopy(self.policy.state_dict());teacher=copy.deepcopy(self.teacher.state_dict())
        result=repair_rank_drift(self.policy,self.teacher,self.observations,self.masks,max_steps=0)
        self.assertFalse(result['repaired']);self.assertTrue(result['restored_original'])
        identical(original,self.policy.state_dict());identical(teacher,self.teacher.state_dict())


if __name__=='__main__':unittest.main()
