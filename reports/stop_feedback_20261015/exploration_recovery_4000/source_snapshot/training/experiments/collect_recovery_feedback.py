"""Collect sensor-only rejected-state inputs from fresh training episodes.

The deterministic collector is not a PPO rollout: its actions/returns must
never be used for PPO likelihood updates. Inputs may support a separately
declared auxiliary feasibility loss. No evaluation manifest is consumed.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import torch
from stable_baselines3 import PPO
from experiments.recovery_curriculum import RecoveryNavigationCurriculum
from experiments.feasibility_feedback_ppo import feasibility_loss, FEEDBACK_REASONS
from neural_navigation.safety_context import RECOVERY_SCHEMA, validate_contract


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--policy', type=Path, required=True)
    p.add_argument('--failure-pool', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--episodes', type=int, default=6)
    p.add_argument('--seed', type=int, default=92110)
    args = p.parse_args()
    if args.output.exists() or args.episodes < 1:
        p.error('Fresh output and positive episode count required')
    hardware = json.loads((args.policy.parent/'hardware.json').read_text())
    validate_contract(hardware['policy_observation'])
    if hardware['policy_observation']['schema'] != RECOVERY_SCHEMA:
        p.error('V3 sensor feedback required')
    pool = json.loads(args.failure_pool.read_text())
    if pool.get('kind') != 'training' or not pool.get('templates'):
        p.error('Explicit training-only template pool required')
    torch.set_num_threads(1)
    model = PPO.load(args.policy, device='cpu')
    env = RecoveryNavigationCurriculum(args.policy.parent, stage='robust',
        route_mode='shortcut', motion_profiles=True, footprint_clearance_cost=1.,
        failure_templates=pool['templates'])
    args.output.mkdir(parents=True)
    config = dict(kind='training', deterministic_collection=True,
        ppo_rollout_eligible=False, simulator_truth_in_samples=False,
        policy_sha256=hashlib.sha256(args.policy.read_bytes()).hexdigest(),
        failure_pool_sha256=hashlib.sha256(args.failure_pool.read_bytes()).hexdigest(),
        collector_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        seed=args.seed, requested_episodes=args.episodes,
        evaluation_manifest_consumed=False, status='running', episodes=[],
        collection_schema='rejected_sensor_feedback_v2',
        started_unix=time.time(), sample_count=0)
    def save():
        (args.output/'status.json').write_text(json.dumps(config, indent=2))
    save()
    observations, proposals, commands, episode_ids, timestamps = [], [], [], [], []
    current_reasons = []
    try:
        for episode in range(args.episodes):
            obs, _ = env.reset(seed=args.seed+episode)
            case = json.loads(env.training_case_json)
            count = 0
            for step in range(env.max_steps):
                action = model.predict(obs, deterministic=True)[0]
                # Eligibility is computed exclusively from the exact policy
                # input, before the action. Save resulting command separately.
                with torch.no_grad():
                    _, eligible = feasibility_loss(torch.tensor(obs[None]), torch.tensor(action[None]))
                before = obs.copy()
                timestamp = env.active.elapsed
                obs, _, done, truncated, info = env.step(action)
                command = info['commanded_velocity']
                rejected_idle = (info['training_actions_overridden']
                    and info['execution_guard_reason'] in FEEDBACK_REASONS
                    and abs(command[0]) < .01 and abs(command[1]) < .05)
                if eligible.item() > .5 and rejected_idle:
                    observations.append(before); proposals.append(action.copy())
                    commands.append(info['commanded_velocity'].copy())
                    episode_ids.append(episode); timestamps.append(timestamp)
                    current_reasons.append(info['execution_guard_reason'])
                    count += 1
                if done or truncated:
                    break
            config['episodes'].append(dict(index=episode, seed=args.seed+episode,
                case=case, collected_frames=count, steps=step+1,
                verified_success=bool(info['verified_mission_success']),
                safety_termination=bool(info['safety_termination']), timeout=bool(truncated)))
            config['sample_count'] = len(observations); save()
            print(json.dumps(config['episodes'][-1]), flush=True)
        np.savez_compressed(args.output/'sensor_feedback.npz',
            observations=np.asarray(observations, dtype=np.float32).reshape(-1, 267),
            proposed_actions=np.asarray(proposals, dtype=np.float32).reshape(-1, 2),
            commanded_velocity=np.asarray(commands, dtype=np.float32).reshape(-1, 2),
            episode_ids=np.asarray(episode_ids, dtype=np.int32),
            current_reasons=np.asarray(current_reasons, dtype='U48'),
            timestamps=np.asarray(timestamps, dtype=np.float64))
        config.update(status='complete', finished_unix=time.time(),
            sample_file_sha256=hashlib.sha256((args.output/'sensor_feedback.npz').read_bytes()).hexdigest(),
            limitation='Training-only sampled states. Current accepted probes do not certify future safety. '
                'Deterministic trajectory data is not on-policy PPO data; no reward or advantage is stored.')
        save()
    except Exception as error:
        config.update(status='failed', error=str(error)); save(); raise
    finally:
        env.close()


if __name__ == '__main__':
    main()
