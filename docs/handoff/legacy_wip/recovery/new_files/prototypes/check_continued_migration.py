"""Actual checkpoint initialization audit, without simulation or training."""
import hashlib
import json
from pathlib import Path
import tempfile
import gymnasium as gym
import numpy as np
import torch
from sb3_contrib import MaskablePPO
from experiments.soft_prior_recovery_policy import warm_start_soft_prior
from experiments.focused_recovery_policy import recovery_support
from experiments.masked_navigation import PortableMaskedActor
from neural_navigation.masked_actions import ACTIONS
from prototypes.continued_recovery_policy import warm_start_continued, continued_support


class InspectionInterface(gym.Env):
    observation_space = gym.spaces.Box(-1,1,(300,),np.float32)
    action_space = gym.spaces.Discrete(12)

    def __init__(self, reward_scale):
        self.reward_scale = reward_scale

    def reset(self, **kwargs):
        raise RuntimeError('Checkpoint inspection only; never reset a simulator')

    def step(self, action):
        raise RuntimeError('Checkpoint inspection never executes an action')


def main():
    torch.set_num_threads(1)
    project = Path(__file__).resolve().parents[1]
    base = project/'training/runs/safety_context_20261011'
    checkpoint = base/'quality_recovery_4000/candidate/policy_2000.zip'
    legacy = MaskablePPO.load(checkpoint,device='cpu')
    reward_scale = float(getattr(legacy,'training_reward_scale',1.))
    reference = warm_start_soft_prior(checkpoint,InspectionInterface(reward_scale),92611,1/6)
    reference_rng = torch.get_rng_state().clone()
    candidate = warm_start_continued(checkpoint,InspectionInterface(reward_scale),92611,1/6,.7)
    torch.testing.assert_close(torch.get_rng_state(),reference_rng,atol=0,rtol=0)
    actual_rows = []
    sources = []
    for name in ['confirmed_free_initial_five/evaluation/independent_dynamic_023.json',
                 'gated_recovery_8000/initial_0.json']:
        path = base/name
        actual_rows.extend(json.loads(path.read_text())['samples'])
        sources.append(dict(path=name,sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    obs = torch.as_tensor(np.asarray([r['policy_observation'] for r in actual_rows],np.float32))
    masks = np.asarray([r['mask'] for r in actual_rows],bool)
    support_same = recovery_support(obs) == continued_support(obs)
    with torch.no_grad():
        original = reference.policy.get_distribution(obs,action_masks=masks)
        modified = candidate.policy.get_distribution(obs,action_masks=masks)
        torch.testing.assert_close(modified.mode(),original.mode(),atol=0,rtol=0)
        legacy_choice = legacy.predict(obs.numpy(),action_masks=masks,deterministic=True)[0]
        np.testing.assert_array_equal(modified.mode().numpy(),legacy_choice)
        torch.testing.assert_close(modified.distribution.probs[support_same],
                                   original.distribution.probs[support_same],atol=0,rtol=0)
        torch.testing.assert_close(candidate.policy.predict_values(obs),
                                   reference.policy.predict_values(obs),atol=0,rtol=0)
        torch.testing.assert_close(candidate.policy.predict_values(obs),
                                   legacy.policy.predict_values(obs),atol=0,rtol=0)
        actor = torch.jit.trace(PortableMaskedActor(candidate.policy),obs)
        np.testing.assert_array_equal(actor(obs).numpy(),ACTIONS[modified.mode().numpy()])
        actions,values,log_probs = candidate.policy(obs,action_masks=masks)
        checked_values,checked_logs,entropy = candidate.policy.evaluate_actions(obs,actions,action_masks=masks)
        torch.testing.assert_close(log_probs,checked_logs,atol=0,rtol=0)
    ref_params = dict(reference.policy.named_parameters())
    state_count = 0
    for name,param in candidate.policy.named_parameters():
        torch.testing.assert_close(param,ref_params[name],atol=0,rtol=0)
        left = reference.policy.optimizer.state.get(ref_params[name],{})
        right = candidate.policy.optimizer.state.get(param,{})
        assert left.keys() == right.keys()
        if left: state_count += 1
        for key in left:
            if torch.is_tensor(left[key]):
                torch.testing.assert_close(left[key],right[key],atol=0,rtol=0)
            else: assert left[key] == right[key]
    assert state_count == 6
    with tempfile.TemporaryDirectory(prefix='continued_checkpoint_') as temp:
        path = Path(temp)/'policy.zip';candidate.save(path)
        loaded = MaskablePPO.load(path,device='cpu')
        np.testing.assert_array_equal(loaded.predict(obs.numpy(),action_masks=masks,deterministic=True)[0],legacy_choice)
        for name,param in loaded.policy.named_parameters():
            original_param = dict(candidate.policy.named_parameters())[name]
            torch.testing.assert_close(param,original_param,atol=0,rtol=0)
            original_state = candidate.policy.optimizer.state.get(original_param,{})
            loaded_state = loaded.policy.optimizer.state.get(param,{})
            assert original_state.keys() == loaded_state.keys()
            for key in original_state:
                if torch.is_tensor(original_state[key]):
                    torch.testing.assert_close(original_state[key],loaded_state[key],atol=0,rtol=0)
                else: assert original_state[key] == loaded_state[key]
    result=dict(actual_sensor_input_frames=len(actual_rows),same_support_probability_exact_frames=int(support_same.sum()),
        added_support_actual_frames=int((continued_support(obs)&~recovery_support(obs)).sum()),
        initial_greedy_commands_exact=True,critic_values_exact=True,weights_exact=True,
        critic_adam_states_exact=state_count,migration_torch_rng_exact=True,
        sampling_and_ppo_logprob_exact=True,portable_command_parity=True,
        native_save_reload_weights_optimizer_and_choices_exact=True,archived_sources=sources,
        checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        simulation_or_training_executed=False,trained_recovery_proven=False,
        limitation='Checkpoint initialization/parity only. Extra support intentionally changes exploration probabilities; no trained autonomous recovery or safety regression proved.')
    (base/'CONTINUED_NATIVE_MIGRATION_CHECK.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result),flush=True)


if __name__ == '__main__': main()
