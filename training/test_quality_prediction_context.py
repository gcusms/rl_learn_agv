import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
import gymnasium as gym
import numpy as np
import torch
from sb3_contrib import MaskablePPO
from neural_navigation.quality_prediction_context import QualityPredictionContext,TOTAL_DIM
from experiments.quality_masked_navigation import warm_start_quality,QualityAnchoredPPO
from experiments.masked_navigation import PortableMaskedActor
from training.test_contour_velocity import track


def core(tracks=(),time=0.):
    return SimpleNamespace(localizer=SimpleNamespace(std=np.zeros(3),map_agreement=.9,scan_update_accepted=True),
        estimate=np.zeros(3),measured_velocity=np.zeros(2),elapsed=time,
        laser_fusion=SimpleNamespace(predictor=SimpleNamespace(tracks=list(tracks))))


class Toy(gym.Env):
    action_space=gym.spaces.Discrete(12)
    def __init__(self,dim):self.observation_space=gym.spaces.Box(-1,1,(dim,),np.float32)
    def reset(self,*,seed=None,options=None):
        super().reset(seed=seed);self.steps=0;obs=np.zeros(self.observation_space.shape,np.float32)
        obs[235:244]=1;obs[267:269]=1;return obs,{}
    def action_masks(self):return np.ones(12,bool)
    def step(self,action):
        self.steps+=1;obs=np.zeros(self.observation_space.shape,np.float32);obs[235:244]=1;obs[267:269]=1
        return obs,float(action),self.steps==8,False,{}


class QualityContextTests(unittest.TestCase):
    def test_reference_protection_does_not_anchor_unknown_quality(self):
        model=QualityAnchoredPPO('MlpPolicy',Toy(300),device='cpu',n_steps=16,batch_size=16,
            n_epochs=1,reference_strength=2.)
        with torch.no_grad():model.policy.action_net.bias[1]=10.
        model.snapshot_reference()
        obs=torch.zeros((2,300));obs[:,250]=1;obs[:,234]=1;obs[:,232]=1;obs[:,294]=1
        obs[1,297]=1;masks=torch.ones((2,12),dtype=torch.bool)
        np.testing.assert_array_equal(model.eligible(obs,masks).numpy(),[True,False])
        model.learn(16)

    def test_prefix_unchanged_and_unknown_pose_differs_from_empty_supported(self):
        context=QualityPredictionContext([.4,.3]);base=np.linspace(-1,1,282,dtype=np.float32)
        c=core();obs=context.augment(base,c)
        np.testing.assert_array_equal(obs[:282],base);self.assertEqual(obs.shape,(300,))
        self.assertEqual(obs[294],1.)
        c.localizer.scan_update_accepted=False;unknown=context.augment(base,c)
        self.assertEqual(unknown[294],0.);np.testing.assert_array_equal(unknown[:282],base)

    def test_held_and_unknown_velocity_flags_remain_explicit(self):
        context=QualityPredictionContext([.4,.3]);context.features(core([track(0)],0))
        bad=track(.4,7,.18,6,(.14,.11))
        held=context.features(core([bad],.4))
        self.assertEqual(held[14],1.);self.assertEqual(held[16],1.)
        unknown=context.features(core([bad],.7))
        self.assertEqual(unknown[15],1.);self.assertEqual(unknown[14],0.)

    def test_warm_start_preserves_actor_critic_and_adam_with_zero_new_columns(self):
        torch.set_num_threads(1)
        old=MaskablePPO('MlpPolicy',Toy(282),device='cpu',seed=13,n_steps=16,batch_size=16,n_epochs=1)
        old.learn(16)
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'policy.zip';old.save(path);new=warm_start_quality(path,Toy(TOTAL_DIM))
        obs=np.random.default_rng(1).uniform(-1,1,(32,300)).astype(np.float32)
        masks=np.random.default_rng(2).random((32,12))>.5;masks[:,0]=True
        with torch.no_grad():
            a=old.policy.get_distribution(torch.from_numpy(obs[:,:282].copy()),action_masks=masks).distribution.probs.clone()
            b=new.policy.get_distribution(torch.from_numpy(obs),action_masks=masks).distribution.probs.clone()
            torch.testing.assert_close(a,b,atol=1e-7,rtol=0)
            torch.testing.assert_close(old.policy.predict_values(torch.from_numpy(obs[:,:282].copy())),
                new.policy.predict_values(torch.from_numpy(obs)),atol=1e-6,rtol=0)
        old_params=dict(old.policy.named_parameters());new_params=dict(new.policy.named_parameters())
        for name in ('mlp_extractor.policy_net.0.weight','mlp_extractor.value_net.0.weight'):
            p=new_params[name];self.assertEqual(torch.count_nonzero(p[:,282:]).item(),0)
            for key in ('exp_avg','exp_avg_sq'):
                state=new.policy.optimizer.state[p][key]
                torch.testing.assert_close(state[:,:282],old.policy.optimizer.state[old_params[name]][key],atol=0,rtol=0)
                self.assertEqual(torch.count_nonzero(state[:,282:]).item(),0)
        portable=PortableMaskedActor(new.policy)
        obs[:,235:244]=1;obs[:,267:269]=1
        traced=torch.jit.trace(portable,torch.from_numpy(obs))
        with torch.no_grad():torch.testing.assert_close(portable(torch.from_numpy(obs)),traced(torch.from_numpy(obs)),atol=0,rtol=0)
        new.learn(16)  # Expanded actor can update using native MaskablePPO.


if __name__=='__main__':unittest.main()
