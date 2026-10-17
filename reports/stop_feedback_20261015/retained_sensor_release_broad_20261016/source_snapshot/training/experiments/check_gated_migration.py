"""Read actual frozen traces/checkpoint; no evaluated frames enter training."""
import argparse
import json
from pathlib import Path
import tempfile
import gymnasium as gym
import numpy as np
import torch
from sb3_contrib import MaskablePPO
from experiments.gated_recovery_policy import warm_start_gated,GatedRecoveryPPO
from experiments.recovery_temperature_policy import migrate_policy
from experiments.check_recovery_temperature_migration import identical
from experiments.masked_navigation import PortableMaskedActor
from experiments.train_quality_navigation import digest


class Interface(gym.Env):
    observation_space=gym.spaces.Box(-1,1,(300,),np.float32)
    action_space=gym.spaces.Discrete(12)
    def __init__(self,scale):self.reward_scale=scale
    def reset(self,**kwargs):raise RuntimeError('Migration interface cannot simulate')
    def step(self,action):raise RuntimeError('Migration interface cannot execute')


def main():
    parser=argparse.ArgumentParser()
    for name in ['checkpoint','trace','output']:parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();torch.set_num_threads(1)
    if args.output.exists():parser.error('Fresh output required')
    old=MaskablePPO.load(args.checkpoint,device='cpu');migrate_policy(old)
    model=warm_start_gated(args.checkpoint,Interface(float(getattr(old,'training_reward_scale',1.))))
    samples=json.loads(args.trace.read_text())['samples'][:128]
    observations=torch.tensor(np.asarray([r['policy_observation'] for r in samples],np.float32))
    masks=np.asarray([r['mask'] for r in samples],bool)
    with torch.no_grad():
        previous=old.policy.get_distribution(observations,action_masks=masks)
        current=model.policy.get_distribution(observations,action_masks=masks)
        torch.testing.assert_close(previous.distribution.probs,current.distribution.probs,rtol=0,atol=0)
        torch.testing.assert_close(previous.mode(),current.mode(),rtol=0,atol=0)
        torch.testing.assert_close(old.policy.predict_values(observations),model.policy.predict_values(observations),rtol=0,atol=0)
    source=dict(old.policy.named_parameters())
    critic_states=0
    for name,p in model.policy.named_parameters():
        if name.startswith(('mlp_extractor.value_net.','value_net.')):
            torch.testing.assert_close(p,source[name],rtol=0,atol=0)
            identical(model.policy.optimizer.state.get(p,{}),old.policy.optimizer.state.get(source[name],{}));critic_states+=1
    with tempfile.TemporaryDirectory(prefix='gated-migration-') as directory:
        path=Path(directory)/'policy.zip';model.save(path)
        restored=GatedRecoveryPPO.load(path,device='cpu')
        np.testing.assert_array_equal(model.predict(observations.numpy(),action_masks=masks,deterministic=True)[0],
            restored.predict(observations.numpy(),action_masks=masks,deterministic=True)[0])
        identical(model.policy.state_dict(),restored.policy.state_dict())
        identical(model.policy.optimizer.state_dict(),restored.policy.optimizer.state_dict())
        actor=torch.jit.trace(PortableMaskedActor(restored.policy),observations)
        with torch.no_grad():
            torch.testing.assert_close(actor(observations),PortableMaskedActor(model.policy)(observations),rtol=0,atol=0)
    args.output.write_text(json.dumps(dict(actual_input_frames=len(samples),checkpoint_sha256=digest(args.checkpoint),
        trace_sha256=digest(args.trace),initial_masked_probabilities_and_choices_exact=True,
        critic_weights_values_and_adam_exact=True,critic_parameter_states=critic_states,
        save_reload_optimizer_and_portable_parity=True,training_or_commands_executed=False,
        limitation='Migration/export proof only; not trained navigation or recovery improvement.'),indent=2))
    print(args.output.read_text())


if __name__=='__main__':main()
