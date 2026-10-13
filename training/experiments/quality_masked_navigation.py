"""Append-only experimental curriculum and parity-preserving masked warm start."""
import copy
import gymnasium as gym
import numpy as np
import torch
from sb3_contrib import MaskablePPO
from neural_navigation.quality_prediction_context import QualityPredictionContext,TOTAL_DIM,PREFIX_DIM
from experiments.anchored_masked_ppo import AnchoredMaskedPPO


class QualityAnchoredPPO(AnchoredMaskedPPO):
    def snapshot_reference(self):
        if self.observation_space.shape!=(TOTAL_DIM,) or self.action_space.n!=12:
            raise ValueError('Quality masked policy required')
        self.reference_policy=copy.deepcopy(self.policy)
        self.reference_policy.set_training_mode(False);self.reference_policy.requires_grad_(False)
        self.reference_cache=None

    def eligible(self,observations,masks):
        return (super().eligible(observations,masks)&(observations[:,294]>.5)
            &(observations[:,295:299]<.5).all(dim=1))

    def remember_clear_states(self):
        observations=torch.as_tensor(self.rollout_buffer.observations.reshape(-1,TOTAL_DIM),device=self.device)
        masks=torch.as_tensor(self.rollout_buffer.action_masks.reshape(-1,12),device=self.device)
        keep=self.eligible(observations,masks);observations=observations[keep];masks=masks[keep]
        if len(observations)>128:
            index=torch.as_tensor(np.linspace(0,len(observations)-1,128).astype(int),device=self.device)
            observations=observations[index];masks=masks[index]
        if len(observations):
            if self.reference_cache is not None:
                observations=torch.cat([self.reference_cache[0],observations]);masks=torch.cat([self.reference_cache[1],masks])
            self.reference_cache=(observations[-1024:].detach().clone(),masks[-1024:].detach().clone())


class QualityMaskedWrapper(gym.Wrapper):
    def __init__(self,env,reward_scale=1.):
        super().__init__(env)
        if not np.isfinite(reward_scale) or not 0<reward_scale<=1:
            raise ValueError('Positive finite reward scale at most one required')
        self.reward_scale=float(reward_scale)
        if env.observation_space.shape!=(PREFIX_DIM,):raise ValueError('Native masked V4 curriculum required')
        self.observation_space=gym.spaces.Box(-1,1,(TOTAL_DIM,),np.float32)
        self.prediction=QualityPredictionContext(env.hardware['robot_footprint_m'])

    def reset(self,**kwargs):
        self.prediction.reset();obs,info=self.env.reset(**kwargs)
        return self.prediction.augment(obs,self.env.core),info

    def step(self,action):
        obs,reward,done,truncated,info=self.env.step(action)
        info=dict(info,native_reward=float(reward),training_reward_scale=self.reward_scale)
        return self.prediction.augment(obs,self.env.core),reward*self.reward_scale,done,truncated,info

    def action_masks(self):return self.env.action_masks()


def resume_quality(checkpoint,env,seed=92401):
    """Resume native 300-field weights/Adam; no remigration or reward-unit guessing."""
    model=QualityAnchoredPPO.load(checkpoint,env=env,device='cpu',seed=seed)
    if model.observation_space.shape!=(TOTAL_DIM,) or model.action_space.n!=12:
        raise ValueError('Native quality checkpoint required')
    scale=float(getattr(model,'training_reward_scale',1.))
    scales=env.get_attr('reward_scale') if hasattr(env,'get_attr') else [env.reward_scale]
    if not all(x==scale for x in scales):
        raise ValueError('Checkpoint value units and training reward scale disagree')
    model.reference_strength=2.;model.reference_kl_budget=.02
    model.snapshot_reference()
    return model


def warm_start_quality(checkpoint,env,seed=92401,model_class=MaskablePPO,**model_options):
    old=MaskablePPO.load(checkpoint,device='cpu')
    if old.observation_space.shape!=(PREFIX_DIM,) or old.action_space.n!=12:
        raise ValueError('Native 282-field masked actor required')
    if env.observation_space.shape!=(TOTAL_DIM,):raise ValueError('Quality observation wrapper required')
    new=model_class('MlpPolicy',env,device='cpu',seed=seed,n_steps=old.n_steps,
        batch_size=old.batch_size,n_epochs=old.n_epochs,learning_rate=3e-5,
        target_kl=old.target_kl,ent_coef=old.ent_coef,gamma=old.gamma,gae_lambda=old.gae_lambda,
        policy_kwargs=copy.deepcopy(old.policy_kwargs),**model_options)
    source=old.policy.state_dict();target=new.policy.state_dict()
    inputs={'mlp_extractor.policy_net.0.weight','mlp_extractor.value_net.0.weight'}
    for name,value in target.items():
        if name in inputs:value.zero_();value[:,:PREFIX_DIM]=source[name]
        else:
            if value.shape!=source[name].shape:raise ValueError('Unexpected policy architecture')
            value.copy_(source[name])
    new.policy.load_state_dict(target);previous=dict(old.policy.named_parameters())
    for name,parameter in new.policy.named_parameters():
        state=old.policy.optimizer.state.get(previous[name])
        if not state:continue
        migrated={}
        for key,value in state.items():
            if torch.is_tensor(value):
                if name in inputs and value.shape==previous[name].shape:
                    expanded=torch.zeros_like(parameter);expanded[:,:PREFIX_DIM]=value;migrated[key]=expanded
                else:migrated[key]=value.clone()
            else:migrated[key]=copy.deepcopy(value)
        new.policy.optimizer.state[parameter]=migrated
    new.num_timesteps=old.num_timesteps;new._n_updates=old._n_updates
    return new
