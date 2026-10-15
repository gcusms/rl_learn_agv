"""Demonstration initialization followed by real on-policy recovery PPO.

The initialized physics history is excluded from the PPO rollout. Ordinary
tasks remain in the curriculum; all controlled actions use the same masks and
guard. The base parameters and terminal actor stay frozen; the explicitly
selected proactive residual can also learn near-obstacle moving decisions.
"""
import argparse
import hashlib
import json
from pathlib import Path
import random

import numpy as np
import torch
from stable_baselines3.common.monitor import Monitor

from experiments.experiment_provenance import record_provenance
from experiments.prefix_recovery_training import PrefixRecoveryTraining
from experiments.recovery_demonstrations import load_demonstrations, initialize_from_demonstrations
from experiments.recovery_sampling import ScheduledRecoveryCurriculum
from experiments.stop_feedback_navigation import StopFeedbackWrapper
from experiments.stop_recovery_policy import warm_start_recovery, export_recovery
from experiments.train_localization_recovery import training_case
from experiments.train_stop_feedback import Progress
from neural_navigation.decision_feedback import WAIT_SCHEMA
from neural_navigation.experiment_budget import ExperimentBudget


def frozen_base_digest(policy):
    digest = hashlib.sha256()
    for module in [policy.mlp_extractor.policy_net.base, policy.action_net.base]:
        for value in module.state_dict().values():
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


class RecoveryProgress(Progress):
    def __init__(self, output, budget, base_digest):
        super().__init__(output, budget, record_rollouts=True)
        self.base_digest = base_digest
        self.counts.update(recovery_actor_frames=0, prefix_controlled_frames=0,
            routine_controlled_frames=0, recovery_wait_choices=0, recovery_motion_choices=0)

    def _on_step(self):
        supported = self.model.policy.recovery_support(self.locals['obs_tensor']).cpu().numpy()
        for index, info in enumerate(self.locals['infos']):
            self.counts['recovery_actor_frames'] += int(supported[index])
            self.counts['prefix_controlled_frames' if info['recovery_initialization_trace'] else 'routine_controlled_frames'] += 1
            if supported[index]:
                self.counts['recovery_wait_choices' if info['masked_choice'] == 0 else 'recovery_motion_choices'] += 1
        result = super()._on_step()
        for index, info in enumerate(self.locals['infos']):
            self.trace_steps[-len(self.locals['infos'])+index].update(
                initialization_trace=info['recovery_initialization_trace'],
                initialization_prefix_id=info['recovery_prefix_id'],
                recovery_actor_supported=bool(supported[index]))
        if self.num_timesteps % 250 == 0 and frozen_base_digest(self.model.policy) != self.base_digest:
            raise RuntimeError('The frozen base actor changed during recovery learning')
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['bundle', 'checkpoint', 'training-cases', 'prefixes', 'demonstrations', 'output']:
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--steps', type=int, default=2000)
    parser.add_argument('--seed', type=int, default=231015)
    parser.add_argument('--cpu-cores', type=float, default=.35)
    parser.add_argument('--bc-updates', type=int, default=400,
        help='Maximum supervised initialization updates (1..1000); never counted as PPO steps')
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--steady-base', action='store_true',
        help='Bound random exploration of the frozen base outside learnable recovery; keep stop legal')
    modes.add_argument('--proactive-recovery', action='store_true',
        help='Permit a learned residual during near-obstacle approach, retaining the original action masks')
    parser.add_argument('--initialization-only', action='store_true',
        help='Validate one physical prefix and two model-controlled frames; no PPO training or navigation score')
    args = parser.parse_args()
    if (args.output.exists() or not 1000 <= args.steps <= 8000 or args.steps % 250
            or not 1 <= args.bc_updates <= 1000):
        parser.error('Fresh output, 1000..8000 PPO steps in multiples of 250, and 1..1000 BC updates required')
    args.output = args.output.resolve()
    args.output.mkdir(parents=True)
    torch.set_num_threads(1)
    budget = ExperimentBudget(args.cpu_cores, memory_mib=3072, directory=args.output)
    cases = [training_case(c) for c in json.loads(args.training_cases.read_text())]
    prefixes = json.loads(args.prefixes.read_text())
    demonstration_specs = json.loads(args.demonstrations.read_text())
    physical = ScheduledRecoveryCurriculum(args.bundle.resolve(), stage='robust', route_mode='shortcut',
        motion_profiles=True, recorded_cases=cases, recorded_episodes_per_cycle=1)
    env = PrefixRecoveryTraining(StopFeedbackWrapper(physical, reward_scale=1/60,
        feedback_schema=WAIT_SCHEMA), prefixes, args.output, budget)
    if args.proactive_recovery:
        from experiments.proactive_recovery import ProactiveRecoveryPolicy
        model = warm_start_recovery(args.checkpoint,env,seed=args.seed,
            policy_class=ProactiveRecoveryPolicy)
    elif args.steady_base:
        from experiments.steady_base_recovery import SteadyBaseRecoveryPolicy
        model = warm_start_recovery(args.checkpoint, env, seed=args.seed,
            policy_class=SteadyBaseRecoveryPolicy)
    else:
        model = warm_start_recovery(args.checkpoint, env, seed=args.seed)
    base_digest = frozen_base_digest(model.policy)
    obs, masks, choices, sources = load_demonstrations(demonstration_specs, model.policy)
    np.savez_compressed(args.output/'demonstrations.npz', observations=obs, masks=masks, choices=choices)
    config = dict(steps=args.steps, seed=args.seed, cpu_cores=args.cpu_cores,
        maximum_bc_updates=args.bc_updates,
        recovery_contract=model.recovery_contract, prefixes=prefixes, demonstration_sources=sources,
        demonstration_input_frames=len(obs), prior_checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        bundle_manifest_sha256=hashlib.sha256((args.bundle/'manifest.json').read_bytes()).hexdigest(),
        frozen_base_sha256=base_digest, reward_changed=False, deployment_approved=False,
        initialization_not_counted_as_ppo_steps=True, general_and_previous_failure_tasks_retained=True,
        limitations=['Demonstrations are reused physical diagnostic trajectories.',
                    'Recovery rehearsal and correlated prefixes are not independent generalization.'])
    (args.output/'config.json').write_text(json.dumps(config, indent=2)+'\n')
    record_provenance(args.output, args.bundle)
    def save_status(**values):
        p = args.output/'status.tmp'
        p.write_text(json.dumps(values, indent=2)+'\n')
        p.replace(args.output/'status.json')
    save_status(status='demonstration_initialization', target_steps=args.steps, deployment_approved=False)
    try:
        model.save(args.output/'policy_migrated.zip')
        initialized = initialize_from_demonstrations(model, obs, masks, choices, budget,
            maximum_updates=args.bc_updates)
        (args.output/'INITIALIZATION.json').write_text(json.dumps(initialized, indent=2)+'\n')
        if frozen_base_digest(model.policy) != base_digest:
            raise RuntimeError('Demonstration learning changed the frozen base')
        model.demonstration_initialization = dict(initialized, source_episodes=sources)
        model.save(args.output/'policy_bc.zip')
        export_recovery(model, args.bundle, args.output/'bc_portable')
        if args.initialization_only:
            current, _ = env.reset(seed=args.seed)
            smoke_choices, smoke_commands = [], []
            for _ in range(2):
                choice = model.predict(current, deterministic=True, action_masks=env.action_masks())[0]
                current, _, done, truncated, smoke_info = env.step(int(choice))
                smoke_choices.append(int(choice))
                smoke_commands.append(np.asarray(smoke_info['commanded_velocity']).tolist())
                if done or truncated:
                    raise ValueError('Two-frame smoke test ended unexpectedly')
                budget.tick()
            save_status(status='initialization_verified', controlled_smoke_frames=2,
                controlled_smoke_choices=smoke_choices, controlled_smoke_commands=smoke_commands,
                physical_prefix_frames=env.initialization_frames, additional_ppo_steps=0,
                not_navigation_acceptance=True, resource=budget.statistics(), deployment_approved=False)
            return
        model.save(args.output/'policy_initial.zip')
        save_status(status='ppo_training', target_steps=args.steps, deployment_approved=False)
        model.set_env(Monitor(env))
        progress = RecoveryProgress(args.output, budget, base_digest)
        model.learn(total_timesteps=args.steps, callback=progress, reset_num_timesteps=True)
        model.save(args.output/'policy.zip')
        export_recovery(model, args.bundle, args.output/'portable')
        torch.save(dict(torch_rng=torch.get_rng_state(), numpy_rng=np.random.get_state(),
            python_rng=random.getstate(), curriculum_rng=env.unwrapped.np_random.bit_generator.state),
            args.output/'random_state.pt')
        save_status(status='complete', additional_ppo_steps=model.num_timesteps,
            initialization_frames=env.initialization_frames, initialized_episodes=env.warmup_count,
            counts=progress.counts, reasons=progress.reasons, resource=budget.statistics(),
            frozen_base_sha256=frozen_base_digest(model.policy),
            feedback_schema=WAIT_SCHEMA, recovery_contract=model.recovery_contract, deployment_approved=False)
    except BaseException as error:
        save_status(status='failed', additional_ppo_steps=model.num_timesteps,
            error=repr(error), resource=budget.statistics(), deployment_approved=False)
        raise
    finally:
        env.close()


if __name__ == '__main__':
    main()
