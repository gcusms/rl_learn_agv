import copy
import unittest
import gymnasium as gym
import numpy as np
import torch
from experiments.recovery_temperature_policy import RecoveryTemperaturePolicy
from experiments.backtracking_recovery_ppo import commit_constrained_candidate,parameter_partition
from experiments.check_recovery_temperature_migration import identical
from experiments.greedy_stable_quality_ppo import rank_violation


class BacktrackingTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1);torch.manual_seed(712)
        self.policy=RecoveryTemperaturePolicy(gym.spaces.Box(-1,1,(300,),np.float32),
            gym.spaces.Discrete(12),lambda _:3e-5)
        # Establish Adam history, then a second real update for both networks.
        for _ in range(2):
            self.before=copy.deepcopy(self.policy.state_dict())
            self.optimizer=copy.deepcopy(self.policy.optimizer.state_dict())
            self.policy.optimizer.zero_grad()
            sum(p.sum() for p in self.policy.parameters()).backward()
            self.policy.optimizer.step()
        self.proposed=copy.deepcopy(self.policy.state_dict())
        self.proposed_optimizer=copy.deepcopy(self.policy.optimizer.state_dict())
        self.actor,self.critic=parameter_partition(self.policy)

    def delta(self):
        return max(float((p-self.before[name]).detach().abs().max()) for name,p in self.actor.items())

    def test_backtracking_keeps_critic_and_same_gradient_moments(self):
        full=self.delta()
        result=commit_constrained_candidate(self.policy,self.before,self.optimizer,self.delta,full*.6)
        self.assertTrue(result['actor_step_accepted']);self.assertEqual(result['actor_step_fraction'],.5)
        self.assertLessEqual(self.delta(),full*.6)
        for name,p in self.critic.items():torch.testing.assert_close(p,self.proposed[name],rtol=0,atol=0)
        identical(self.proposed_optimizer,self.policy.optimizer.state_dict())

    def test_total_actor_rejection_restores_actor_adam_only(self):
        result=commit_constrained_candidate(self.policy,self.before,self.optimizer,self.delta,0,max_backtracks=1)
        self.assertFalse(result['actor_step_accepted']);self.assertTrue(result['critic_step_retained'])
        for name,p in self.actor.items():torch.testing.assert_close(p,self.before[name],rtol=0,atol=0)
        for name,p in self.critic.items():torch.testing.assert_close(p,self.proposed[name],rtol=0,atol=0)
        actual=self.policy.optimizer.state_dict()
        by_id={id(p):key for group,saved in zip(self.policy.optimizer.param_groups,actual['param_groups'])
            for p,key in zip(group['params'],saved['params'])}
        for p in self.actor.values():identical(actual['state'][by_id[id(p)]],self.optimizer['state'][by_id[id(p)]])
        for p in self.critic.values():identical(actual['state'][by_id[id(p)]],self.proposed_optimizer['state'][by_id[id(p)]])

    def test_preexisting_violation_fully_restores(self):
        result=commit_constrained_candidate(self.policy,self.before,self.optimizer,lambda:1.,.02,max_backtracks=1)
        self.assertFalse(result['critic_step_retained'])
        identical(self.before,self.policy.state_dict());identical(self.optimizer,self.policy.optimizer.state_dict())

    def test_actual_masked_rank_guard_accepts_only_preserved_winner(self):
        with torch.no_grad():
            self.policy.action_net.weight.zero_();self.policy.action_net.bias.fill_(-5.)
            self.policy.action_net.bias[1]=.1;self.policy.action_net.bias[11]=0.
        observations=torch.zeros(16,300);masks=torch.ones(16,12,dtype=torch.bool)
        with torch.no_grad():
            teacher=self.policy.get_distribution(observations,action_masks=masks).distribution.probs.clone()
        before=copy.deepcopy(self.policy.state_dict())
        optimizer=copy.deepcopy(self.policy.optimizer.state_dict())
        with torch.no_grad():self.policy.action_net.bias[11]=.2
        def constraint():
            current=self.policy.get_distribution(observations,action_masks=masks).distribution.probs
            kl=(teacher*(teacher.log()-current.log())).sum(-1).mean()
            worst=rank_violation(teacher,current).max()
            return kl+torch.where(worst>0,.02+worst,worst)
        self.assertGreater(float(constraint().detach()),.02)
        result=commit_constrained_candidate(self.policy,before,optimizer,constraint,.02)
        self.assertTrue(result['actor_step_accepted']);self.assertLess(result['actor_step_fraction'],1)
        current=self.policy.get_distribution(observations,action_masks=masks).distribution.probs
        self.assertTrue(bool((current.argmax(-1)==1).all()))
        self.assertEqual(float(rank_violation(teacher,current).max().detach()),0.)


if __name__=='__main__':unittest.main()
