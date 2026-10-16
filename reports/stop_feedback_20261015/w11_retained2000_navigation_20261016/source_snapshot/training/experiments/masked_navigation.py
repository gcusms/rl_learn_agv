"""PPO chooses sensor-validated velocity combinations, including legal wait.

Masks remove currently rejected commands, never certify future dynamic safety.
All-unknown/all-rejected states retain the zero command. Simulator labels do
not enter the observation, masks or categorical actor.
"""
import copy
import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO
from sb3_contrib import MaskablePPO
from experiments.recovery_curriculum import RecoveryNavigationCurriculum
from neural_navigation.safety_context import (
    OBSERVATION_DIM, PREDICTIVE_DIM, PREDICTIVE_SCHEMAS)
from neural_navigation.masked_actions import ACTIONS,COMMANDS,action_contract,masks_from_observation


class MaskedNavigationCurriculum(RecoveryNavigationCurriculum):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        if self.hardware['policy_observation']['schema'] not in PREDICTIVE_SCHEMAS:
            raise ValueError('Masked velocity choices require the V4 probe contract')
        from neural_navigation.masked_actions import validate_hardware_action
        validate_hardware_action(self.hardware)
        self.action_space=gym.spaces.Discrete(len(ACTIONS))

    def action_masks(self):
        return masks_from_observation(self.previous_observation).copy()

    def step(self,action):
        if not self.action_space.contains(action):
            raise ValueError('Discrete velocity choice required')
        index=int(action); mask=self.action_masks()
        if not mask[index]:
            raise ValueError('Rejected command selected despite action mask')
        obs,reward,done,truncated,info=super().step(ACTIONS[index])
        expected=COMMANDS[index]
        executed=np.asarray(info['commanded_velocity'])
        changed=not np.allclose(expected,executed,atol=2e-8,rtol=0)
        info.update(masked_choice=index,masked_choice_valid=True,
            masked_current_nonzero_choices=int(mask[1:].sum()),
            masked_chose_wait_with_nonzero_available=bool(index==0 and mask[1:].any()),
            nonzero_choice_changed_by_guard=bool(index!=0 and changed and
                info['execution_guard_reason']!='estimated_goal_reached'))
        return obs,reward,done,truncated,info


class FocusedMaskedCurriculum(MaskedNavigationCurriculum):
    """Rehearse recorded training failures while retaining general tasks.

    Recorded initial conditions enter only simulator reset, never observations.
    These repeated tasks must be reported separately from independent evals.
    """
    def __init__(self,*args,recorded_cases=(),replay_probability=.25,replay_seed=0,**kwargs):
        if not 0<=replay_probability<=.5:
            raise ValueError('Replay probability must retain at least half general tasks')
        self.recorded_cases=list(recorded_cases)
        self.replay_probability=replay_probability
        self.replay_rng=np.random.default_rng(replay_seed)
        super().__init__(*args,**kwargs)

    def reset(self,seed=None,options=None):
        if seed is not None:self.replay_rng=np.random.default_rng(seed)
        if self.recorded_cases and not options and self.replay_rng.random()<self.replay_probability:
            from experiments.fixed_training_replay import reset_training_episode
            case=self.recorded_cases[int(self.replay_rng.integers(len(self.recorded_cases)))]
            observation,info=reset_training_episode(self,case)
            info['focused_training_replay']=True
            return observation,info
        # A recorded reset reseeds the native simulation; give general tasks
        # fresh deterministic seeds instead of repeating its subsequent RNG.
        observation,info=super().reset(seed=int(self.replay_rng.integers(2**31-1)),options=options)
        info['focused_training_replay']=False
        return observation,info


class Interface(gym.Env):
    observation_space=gym.spaces.Box(-1,1,(PREDICTIVE_DIM,),np.float32)
    action_space=gym.spaces.Discrete(len(ACTIONS))
    def reset(self,*,seed=None,options=None):
        raise RuntimeError('Attach the real masked simulation before training')
    def step(self,action):
        raise RuntimeError('Interface is not a simulator')


def warm_start(checkpoint,seed=92201,n_steps=250,batch_size=250,n_epochs=4):
    old=PPO.load(checkpoint,device='cpu')
    dim=old.observation_space.shape[0]
    if dim not in (OBSERVATION_DIM,PREDICTIVE_DIM) or old.action_space.shape!=(2,) or old.use_sde:
        raise ValueError('Flat continuous V3/V4 Gaussian policy required')
    allowed=('net_arch','activation_fn','ortho_init','features_extractor_class',
        'features_extractor_kwargs','share_features_extractor','normalize_images',
        'optimizer_class','optimizer_kwargs')
    kwargs={k:copy.deepcopy(v) for k,v in old.policy_kwargs.items() if k in allowed}
    new=MaskablePPO('MlpPolicy',Interface(),device='cpu',seed=seed,n_steps=n_steps,
        batch_size=batch_size,n_epochs=n_epochs,learning_rate=3e-5,target_kl=.02,ent_coef=.003,
        gamma=old.gamma,gae_lambda=old.gae_lambda,policy_kwargs=kwargs)
    source=old.policy.state_dict();target=new.policy.state_dict()
    inputs={'mlp_extractor.policy_net.0.weight','mlp_extractor.value_net.0.weight'}
    head={'action_net.weight','action_net.bias'}
    for name,value in target.items():
        if name in head: continue
        if name not in source: raise ValueError('Unexpected categorical architecture')
        if name in inputs and dim==OBSERVATION_DIM:
            value.zero_();value[:,:dim]=source[name]
        else:
            if value.shape!=source[name].shape: raise ValueError('Policy projection changed')
            value.copy_(source[name])
    # Gaussian log densities at the grid choices, dropping only the term
    # common to every choice. This is an explicit new categorical action
    # representation, not continuous-action prediction parity.
    precision=old.policy.log_std.detach().exp().square().reciprocal()
    if not torch.isfinite(precision).all(): raise ValueError('Finite teacher variance required')
    choices=torch.from_numpy(ACTIONS)
    projection=choices*precision
    target['action_net.weight'].copy_(projection@source['action_net.weight'])
    target['action_net.bias'].copy_(projection@source['action_net.bias']-
        .5*(choices.square()*precision).sum(dim=1))
    new.policy.load_state_dict(target)
    previous=dict(old.policy.named_parameters())
    for name,parameter in new.policy.named_parameters():
        if name in head: continue  # New choice head gets a fresh Adam history.
        state=old.policy.optimizer.state.get(previous[name])
        if not state: continue
        migrated={}
        for key,value in state.items():
            if torch.is_tensor(value):
                if name in inputs and dim==OBSERVATION_DIM and value.shape==previous[name].shape:
                    expanded=torch.zeros_like(parameter);expanded[:,:dim]=value;migrated[key]=expanded
                else: migrated[key]=value.clone()
            else: migrated[key]=copy.deepcopy(value)
        new.policy.optimizer.state[parameter]=migrated
    new.num_timesteps=old.num_timesteps;new._n_updates=old._n_updates
    new.masking_migration=dict(source_dimensions=dim,target_dimensions=PREDICTIVE_DIM,
        actor_critic_hidden_weights_retained=True,shared_parameter_adam_retained=True,
        categorical_head_initialized_from_gaussian_density=True,new_head_optimizer_reset=True,
        continuous_action_parity_claimed=False,exact_rollout_or_rng_resume=False)
    return new


class PortableMaskedActor(torch.nn.Module):
    def __init__(self,policy):
        super().__init__()
        self.net=copy.deepcopy(policy.mlp_extractor.policy_net)
        self.head=copy.deepcopy(policy.action_net)
        self.register_buffer('actions',torch.from_numpy(ACTIONS.copy()))

    def forward(self,observations):
        x=observations.flatten(1)
        masks=torch.cat([torch.ones_like(x[:,:1],dtype=torch.bool),
            x[:,235:244]>.5,x[:,267:269]>.5],dim=1)
        logits=self.head(self.net(x))
        chosen=torch.where(masks,logits,torch.full_like(logits,-1e8)).argmax(dim=1)
        return self.actions[chosen]
