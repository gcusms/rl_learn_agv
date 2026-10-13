import tempfile
from pathlib import Path
import unittest
import gymnasium as gym
import numpy as np
import torch
from sb3_contrib import MaskablePPO
from experiments.anchored_masked_ppo import AnchoredMaskedPPO


def clear_observation():
    obs=np.zeros(282,np.float32);obs[235:244]=1;obs[267:269]=1
    obs[250]=1;obs[234]=1;obs[232]=1
    return obs


class Toy(gym.Env):
    observation_space=gym.spaces.Box(-1,1,(282,),np.float32)
    action_space=gym.spaces.Discrete(12)
    def reset(self,*,seed=None,options=None):
        super().reset(seed=seed);self.steps=0
        return clear_observation(),{}
    def action_masks(self):return np.ones(12,bool)
    def step(self,action):
        self.steps+=1
        return clear_observation(),float(action),self.steps==8,False,{}


class AnchoredMaskedTests(unittest.TestCase):
    def model(self,**kwargs):
        torch.set_num_threads(1)
        model=AnchoredMaskedPPO('MlpPolicy',Toy(),device='cpu',seed=12,n_steps=16,batch_size=16,n_epochs=2,**kwargs)
        with torch.no_grad():
            model.policy.action_net.weight.zero_();model.policy.action_net.bias.zero_()
            model.policy.action_net.bias[1]=1.
        model.snapshot_reference()
        return model

    def test_wait_blockage_and_supported_motion_are_not_anchored(self):
        m=self.model(reference_strength=1.)
        obs=torch.tensor(np.tile(clear_observation(),(4,1)))
        masks=torch.ones((4,12),dtype=torch.bool)
        obs[1,281]=1.;obs[2,266]=.1;masks[3,1:]=False
        np.testing.assert_array_equal(m.eligible(obs,masks).numpy(),[True,False,False,False])
        with torch.no_grad():m.reference_policy.action_net.bias[0]=2.
        self.assertFalse(m.eligible(obs[:1],masks[:1]).item())

    def test_reference_uses_current_masks_and_has_no_gradient(self):
        m=self.model(reference_strength=1.)
        obs=torch.tensor(clear_observation()[None]);masks=torch.ones((1,12),dtype=torch.bool)
        with torch.no_grad():m.policy.action_net.bias[2]+=1
        loss,fraction=m.reference_loss(obs,masks);self.assertGreater(loss.item(),0);self.assertEqual(fraction.item(),1.)
        loss.backward()
        self.assertTrue(any(p.grad is not None for p in m.policy.parameters()))
        self.assertTrue(all(p.grad is None for p in m.reference_policy.parameters()))
        masks[:,2]=False
        # Eligibility requires all probes accepted: an unavailable alternative
        # is not preserved by this clear-state regularizer.
        loss,fraction=m.reference_loss(obs,masks)
        self.assertEqual(loss.item(),0.);self.assertEqual(fraction.item(),0.)

    def test_budget_rollback_restores_policy_and_optimizer(self):
        m=self.model(reference_strength=1.,reference_kl_budget=1e-12,learning_rate=.1)
        before={k:v.clone() for k,v in m.policy.state_dict().items()}
        m.learn(16)
        self.assertEqual(m.last_reference_statistics['rolled_back_optimizer_steps'],1)
        self.assertEqual(m.last_reference_statistics['accepted_optimizer_steps'],0)
        self.assertEqual(m.last_reference_statistics['cache_states'],16)
        for k,v in m.policy.state_dict().items():torch.testing.assert_close(v,before[k],atol=0,rtol=0)
        self.assertEqual(len(m.policy.optimizer.state),0)

    def test_zero_strength_matches_native_training_and_export_has_no_reference(self):
        initial=self.model(reference_strength=0.)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'policy.zip';initial.save(path)
            native=MaskablePPO.load(path,env=Toy(),device='cpu')
            anchored=AnchoredMaskedPPO.load(path,env=Toy(),device='cpu')
            self.assertIsNone(anchored.reference_policy);self.assertIsNone(anchored.reference_cache)
            native.set_random_seed(21);native.learn(16)
            anchored.set_random_seed(21);anchored.learn(16)
            for k,v in native.policy.state_dict().items():
                torch.testing.assert_close(v,anchored.policy.state_dict()[k],atol=0,rtol=0)


if __name__=='__main__':unittest.main()
