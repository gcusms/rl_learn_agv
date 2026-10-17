"""Explicit V3-to-V4 warm start; preserve actor, critic and Adam history."""
import copy
import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO
from neural_navigation.safety_context import OBSERVATION_DIM, PREDICTIVE_DIM


class ObservationInterface(gym.Env):
    observation_space = gym.spaces.Box(-1, 1, (PREDICTIVE_DIM,), np.float32)
    action_space = gym.spaces.Box(-1, 1, (2,), np.float32)
    def reset(self, *, seed=None, options=None):
        raise RuntimeError('Attach the real simulation environment before learning')
    def step(self, action):
        raise RuntimeError('Interface is not a simulation environment')


def migrate_predictive_policy(checkpoint, algorithm=PPO, seed=92102, **kwargs):
    old = PPO.load(checkpoint, device='cpu')
    if old.observation_space.shape != (OBSERVATION_DIM,) or old.action_space.shape != (2,):
        raise ValueError('267-input V3 actor required')
    model = algorithm('MlpPolicy', ObservationInterface(), device='cpu', seed=seed,
        n_steps=250, batch_size=250, n_epochs=4, learning_rate=3e-5,
        target_kl=.02, ent_coef=.003, policy_kwargs=copy.deepcopy(old.policy_kwargs), **kwargs)
    first = {'mlp_extractor.policy_net.0.weight', 'mlp_extractor.value_net.0.weight'}
    source = old.policy.state_dict(); target = model.policy.state_dict()
    if source.keys() != target.keys(): raise ValueError('Matching flat policy architecture required')
    for name, value in source.items():
        if name in first:
            if value.shape[1] != OBSERVATION_DIM or target[name].shape != (value.shape[0], PREDICTIVE_DIM):
                raise ValueError('Unexpected actor/critic input projection')
            target[name].zero_(); target[name][:, :OBSERVATION_DIM] = value
        else:
            if target[name].shape != value.shape: raise ValueError('Unexpected policy parameter change')
            target[name].copy_(value)
    model.policy.load_state_dict(target)
    old_parameters = dict(old.policy.named_parameters())
    for name, parameter in model.policy.named_parameters():
        state = old.policy.optimizer.state.get(old_parameters[name])
        if not state: continue
        migrated = {}
        for key, value in state.items():
            if torch.is_tensor(value):
                if name in first and value.shape == old_parameters[name].shape:
                    expanded = torch.zeros_like(parameter)
                    expanded[:, :OBSERVATION_DIM] = value; migrated[key] = expanded
                else: migrated[key] = value.clone()
            else: migrated[key] = copy.deepcopy(value)
        model.policy.optimizer.state[parameter] = migrated
    if len(old.policy.optimizer.param_groups) != len(model.policy.optimizer.param_groups):
        raise ValueError('Optimizer group structure changed')
    for source_group, target_group in zip(old.policy.optimizer.param_groups, model.policy.optimizer.param_groups):
        for key, value in source_group.items():
            if key != 'params': target_group[key] = copy.deepcopy(value)
    model._n_updates = old._n_updates; model.num_timesteps = old.num_timesteps
    model.predictive_migration = dict(source_dimensions=OBSERVATION_DIM,
        target_dimensions=PREDICTIVE_DIM, added_input_columns_initialized_zero=True,
        original_parameters_and_optimizer_moments_retained=True)
    return model
