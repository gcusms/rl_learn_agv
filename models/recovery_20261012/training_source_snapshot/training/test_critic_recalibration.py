import copy
import unittest
import numpy as np
import torch
from sb3_contrib import MaskablePPO
from training.test_quality_prediction_context import Toy
from experiments.critic_recalibration import fit_critic


class CriticRecalibrationTests(unittest.TestCase):
    def test_fit_changes_only_critic_and_restores_reward_units(self):
        torch.set_num_threads(1)
        m=MaskablePPO('MlpPolicy',Toy(300),device='cpu',seed=18,n_steps=16,batch_size=16,n_epochs=1)
        m.learn(16);obs=torch.zeros((32,300));obs[:,0]=torch.linspace(-1,1,32)
        targets=20*obs[:,0];masks=np.ones((32,12),bool)
        with torch.no_grad():before=m.policy.get_distribution(obs,action_masks=masks).distribution.probs.clone()
        actor=m.policy.action_net.weight;state=copy.deepcopy(m.policy.optimizer.state[actor])
        fit_critic(m.policy,obs,targets,updates=500,scale=20.)
        with torch.no_grad():
            after=m.policy.get_distribution(obs,action_masks=masks).distribution.probs.clone()
            torch.testing.assert_close(before,after,atol=0,rtol=0)
            mse=torch.mean((m.policy.predict_values(obs).flatten()-targets)**2).item()
        self.assertLess(mse,1.)
        for key,value in state.items():torch.testing.assert_close(value,m.policy.optimizer.state[actor][key],atol=0,rtol=0)
        self.assertNotIn(m.policy.value_net.weight,m.policy.optimizer.state)

    def test_invalid_labels_rejected(self):
        m=MaskablePPO('MlpPolicy',Toy(300),device='cpu',n_steps=16,batch_size=16)
        with self.assertRaises(ValueError):fit_critic(m.policy,torch.zeros((2,300)),torch.tensor([0.,float('nan')]))

    def test_normalized_value_units_are_explicit(self):
        torch.set_num_threads(1)
        m=MaskablePPO('MlpPolicy',Toy(300),device='cpu',seed=19,n_steps=16,batch_size=16)
        obs=torch.zeros((32,300));obs[:,0]=torch.linspace(-1,1,32)
        fit_critic(m.policy,obs,20*obs[:,0],updates=500,scale=20.,restore_reward_units=False)
        with torch.no_grad():mse=torch.mean((m.policy.predict_values(obs).flatten()-obs[:,0])**2).item()
        self.assertLess(mse,.005)


if __name__=='__main__':unittest.main()
