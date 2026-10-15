"""Warm-start a wider PPO observation without discarding learned navigation."""
import copy
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import FlattenExtractor
from neural_navigation.safety_context import BASE_DIM, OBSERVATION_DIM


def warm_start(parent, env, seed, initial_std=.30, learning_rate=3e-5):
    old = PPO.load(parent, device='cpu')
    if old.observation_space.shape != (BASE_DIM,) or old.action_space.shape != (2,):
        raise ValueError('Warm start requires a legacy 235 x 2 policy')
    if not isinstance(old.policy.features_extractor, FlattenExtractor) or old.use_sde or old.policy.squash_output:
        raise ValueError('Warm start requires a flat nonsquashed Gaussian policy')
    if env.observation_space.shape != (OBSERVATION_DIM,):
        raise ValueError('Safety-context training environment required')
    model = PPO('MlpPolicy', env, device='cpu', seed=seed, n_steps=250,
                batch_size=250, n_epochs=4, learning_rate=learning_rate,
                target_kl=.02, ent_coef=.003, gamma=old.gamma, gae_lambda=old.gae_lambda,
                policy_kwargs=copy.deepcopy(old.policy_kwargs), verbose=0)
    source = old.policy.state_dict()
    target = model.policy.state_dict()
    widened = []
    for name, tensor in target.items():
        original = source[name]
        if tensor.shape == original.shape:
            tensor.copy_(original)
        elif name in ('mlp_extractor.policy_net.0.weight', 'mlp_extractor.value_net.0.weight') and tensor.shape[1] == OBSERVATION_DIM:
            tensor.zero_()
            tensor[:, :BASE_DIM].copy_(original)
            widened.append(name)
        else:
            raise ValueError(f'Unexpected architecture difference: {name}')
    if len(widened) != 2:
        raise ValueError('Expected one actor and one critic input layer')
    model.policy.load_state_dict(target)
    observations = np.random.default_rng(seed).uniform(-1, 1, (256, BASE_DIM)).astype(np.float32)
    extended = np.c_[observations, np.random.default_rng(seed+1).uniform(-1, 1, (256, OBSERVATION_DIM-BASE_DIM))].astype(np.float32)
    before = old.predict(observations, deterministic=True)[0]
    after = model.predict(extended, deterministic=True)[0]
    np.testing.assert_allclose(after, before, atol=2e-6, rtol=2e-6)
    with torch.no_grad():
        model.policy.log_std.fill_(float(np.log(initial_std)))
    return model, dict(max_initial_mean_action_error=float(np.max(np.abs(after-before))),
                       old_std=old.policy.log_std.detach().exp().tolist(),
                       initial_std=float(initial_std), optimizer_reset=True,
                       legacy_weights_retained=True, context_input_weights_initially_zero=True)
