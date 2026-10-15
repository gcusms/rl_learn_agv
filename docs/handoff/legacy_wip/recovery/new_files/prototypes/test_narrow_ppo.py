import unittest
from pathlib import Path
import tempfile
import gymnasium as gym
import numpy as np
import torch
from sb3_contrib import MaskablePPO
from prototypes.narrow_ppo import NarrowPolicy,PortableFineActor
from prototypes.narrow_observation import masks,ACTIONS

class Interface(gym.Env):
    observation_space=gym.spaces.Box(-1,1,(318,),np.float32)
    action_space=gym.spaces.Discrete(20)
    def reset(self,**kwargs):return np.zeros(318,np.float32),{}
    def step(self,action):return np.zeros(318,np.float32),0.,True,False,{}

class MigrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.model=MaskablePPO(NarrowPolicy,Interface(),device='cpu',n_steps=10,batch_size=5,seed=73120)
        cls.teacher=MaskablePPO.load('models/recovery_20261012/candidate/policy.zip',device='cpu')
    def test_old_logits_and_frozen_teacher(self):
        x=torch.rand(3,318);x[:,308]=1
        with torch.no_grad():
            old=self.teacher.policy.action_net(self.teacher.policy.mlp_extractor.policy_net(x[:,:300]))
            new=self.model.policy.mlp_extractor.forward_actor(x)
        torch.testing.assert_close(old,new[:,:12],rtol=0,atol=0)
        net=self.model.policy.mlp_extractor.policy_net
        self.assertFalse(any(p.requires_grad for p in net.teacher.parameters()))
        self.assertFalse(any(p.requires_grad for p in net.teacher_head.parameters()))
        optimizer_ids={id(p) for group in self.model.policy.optimizer.param_groups for p in group['params']}
        self.assertFalse(any(id(p) in optimizer_ids for p in net.teacher.parameters()))
    def test_portable_and_checkpoint_parity(self):
        x=np.zeros(318,np.float32);x[308]=1;x[300:308]=1
        actor=torch.jit.trace(PortableFineActor(self.model.policy).eval(),torch.from_numpy(x)[None])
        index,_=self.model.predict(x,action_masks=masks(x),deterministic=True)
        np.testing.assert_array_equal(actor(torch.from_numpy(x)[None])[0].detach().numpy(),ACTIONS[int(index)])
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'policy.zip';self.model.save(path)
            loaded=MaskablePPO.load(path,device='cpu')
            other,_=loaded.predict(x,action_masks=masks(x),deterministic=True)
            self.assertEqual(index,other)
    def test_residual_has_gradient_in_alignment_state(self):
        x=torch.zeros(1,318);x[0,315]=1
        net=self.model.policy.mlp_extractor.policy_net
        net.zero_grad();net(x)[0,12].backward()
        self.assertGreater(float(net.residual[-1].bias.grad.abs().sum()),0)
        self.assertTrue(all(p.grad is None for p in net.teacher.parameters()))
    def test_blocked_unrecognized_state_is_trainable_and_export_keeps_safe_fine_action(self):
        x=torch.zeros(1,318);x[0,316]=1;x[0,300]=1
        net=self.model.policy.mlp_extractor.policy_net
        net.zero_grad();net(x)[0,12].backward()
        self.assertGreater(float(net.residual[-1].bias.grad[12]),0.)
        self.assertTrue(all(p.grad is None for p in net.teacher.parameters()))
        portable=PortableFineActor(self.model.policy).eval()
        with torch.no_grad():portable.net.residual[-1].bias[12]=100.
        np.testing.assert_array_equal(portable(x)[0].detach().numpy(),ACTIONS[12])
        # A dangerous/unsupported action remains masked even with a high score.
        x[0,300]=0
        np.testing.assert_array_equal(portable(x)[0].detach().numpy(),ACTIONS[0])

if __name__=='__main__':unittest.main()
