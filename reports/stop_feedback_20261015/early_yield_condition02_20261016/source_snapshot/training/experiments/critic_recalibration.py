"""Critic-only Monte Carlo calibration; actor and its Adam history stay intact.

Training rewards label V for the frozen behavior policy. Small correlated
episodes cannot establish general calibration or successful navigation.
"""
import torch


def critic_parameters(policy):
    if any(True for _ in policy.features_extractor.parameters()):
        raise ValueError('Parameter-free shared extractor required for critic-only reset')
    return list(policy.mlp_extractor.value_net.parameters())+list(policy.value_net.parameters())


def reset_critic(policy,seed=92501):
    critic_parameters(policy)
    if next(policy.parameters()).device.type!='cpu':raise ValueError('Conservative CPU pilot only')
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        for module in [policy.mlp_extractor.value_net,policy.value_net]:
            for layer in module.modules():
                if isinstance(layer,torch.nn.Linear):
                    torch.nn.init.orthogonal_(layer.weight,gain=1.)
                    torch.nn.init.zeros_(layer.bias)


def fit_critic(policy,observations,returns,updates=600,scale=60.,seed=92501,restore_reward_units=True):
    if observations.ndim!=2 or returns.shape!=(len(observations),) or not torch.isfinite(observations).all() or not torch.isfinite(returns).all():
        raise ValueError('Finite aligned Monte Carlo training labels required')
    if updates<1 or not 0<scale<float('inf'):raise ValueError('Positive updates and finite scale required')
    params=critic_parameters(policy);critic_ids={id(p) for p in params}
    untouched={name:p.detach().clone() for name,p in policy.named_parameters() if id(p) not in critic_ids}
    reset_critic(policy,seed);policy.set_training_mode(False)
    optimizer=torch.optim.Adam(params,lr=1e-3);history=[]
    for i in range(updates):
        prediction=policy.predict_values(observations).flatten()
        loss=torch.nn.functional.mse_loss(prediction,returns/scale)
        if not torch.isfinite(loss):raise ValueError('Nonfinite critic fit')
        optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(params,1.);optimizer.step()
        if i%100==0 or i==updates-1:history.append(dict(update=i+1,normalized_mse=float(loss.detach())))
    # Fit normalized targets, then restore actual reward units for PPO. Clear
    # only critic Adam state because gradient/parameter units changed.
    if restore_reward_units:
        with torch.no_grad():policy.value_net.weight.mul_(scale);policy.value_net.bias.mul_(scale)
    for p in params:policy.optimizer.state.pop(p,None);p.grad=None
    for name,p in policy.named_parameters():
        if name in untouched:torch.testing.assert_close(p,untouched[name],atol=0,rtol=0)
    return history
