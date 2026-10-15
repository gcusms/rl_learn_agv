"""Trace a recorded training task with frozen actors; truth is scoring only."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from stable_baselines3 import PPO
from experiments.recovery_curriculum import RecoveryNavigationCurriculum
from experiments.fixed_training_replay import reset_training_episode
from neural_navigation.safety_context import BASE_DIM, REASON_START, REASONS


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--training-log', type=Path, required=True)
    p.add_argument('--step', type=int, required=True)
    p.add_argument('--policy', type=Path, action='append', required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists(): p.error('Fresh trace output required')
    records = [json.loads(line) for line in args.training_log.read_text().splitlines()]
    matched = [row for row in records if row.get('step') == args.step and 'training_case_json' in row and 'worker' in row]
    if len(matched) != 1: p.error('One actual training episode required')
    case = json.loads(matched[0]['training_case_json'])
    torch.set_num_threads(1); args.output.mkdir(parents=True)
    summaries = []
    for index, path in enumerate(args.policy):
        policy_digest = hashlib.sha256(path.read_bytes()).hexdigest()
        model = PPO.load(path, device='cpu')
        env = RecoveryNavigationCurriculum(path.parent, stage='robust',
            motion_profiles=True, footprint_clearance_cost=1.)
        samples = []
        try:
            obs, _ = reset_training_episode(env, case)
            for step in range(env.max_steps):
                action = model.predict(obs, deterministic=True)[0]
                reason_index = int(np.argmax(obs[BASE_DIM+REASON_START:BASE_DIM+REASON_START+len(REASONS)]))
                row = dict(timestamp=env.active.elapsed, truth=env.active.truth.tolist(),
                    estimate=env.core.estimate.tolist(), odometry=env.active.observed_odometry.tolist(),
                    measured_velocity=env.core.measured_velocity.tolist(),
                    policy_observation=obs.tolist(), proposed_action=action.tolist(),
                    model_command=env.core.velocity_command(action).tolist(),
                    diagnostics=env.core.diagnostics(),
                    scan=[float(x) if np.isfinite(x) else None for x in env.active.scan],
                    actors=env.active.data.qpos[3:].tolist(),
                    safety_context=dict(action_feasibility=obs[235:244].tolist(),
                        policy_previous_stop_reason=REASONS[reason_index],
                        policy_previous_executed_action=obs[247:249].tolist(),
                        policy_wait_fraction=float(obs[-1])))
                obs, reward, done, truncated, info = env.step(action)
                row.update(command=info['commanded_velocity'].tolist(), stop_reason=info['execution_guard_reason'],
                    reward=reward, next_truth=env.active.truth.tolist(), next_actors=env.active.data.qpos[3:].tolist(),
                    collision_events=int(info['collision_events']), safety_termination=bool(info['safety_termination']))
                samples.append(row)
                if done or truncated: break
            result = dict(policy=str(path.resolve()), policy_sha256=policy_digest,
                steps=len(samples), seconds=env.active.elapsed,
                safe_success=bool(info['verified_mission_success']),
                safety_termination=bool(info['safety_termination']), timeout=bool(truncated),
                collisions=int(info['collision_events']))
            if hashlib.sha256(path.read_bytes()).hexdigest() != policy_digest:
                raise ValueError('Frozen checkpoint changed during trace')
            payload = dict(case=dict(case, id=f'training_step_{args.step}'), samples=samples, result=result,
                limitation='Original training reset, current deterministic actor. Simulator truth is saved only for offline scoring; '
                    'the actor and core receive only the recorded sensor inputs. Pre-action timestamps delimit the following control period.')
            (args.output/f'trial_{index}.json').write_text(json.dumps(payload, allow_nan=False))
            summaries.append(result)
            print(json.dumps(result), flush=True)
        finally:
            env.close()
    (args.output/'SUMMARY.json').write_text(json.dumps(dict(kind='training_diagnostic',
        training_log_sha256=hashlib.sha256(args.training_log.read_bytes()).hexdigest(),
        case=case, trials=summaries, completed=True), indent=2))


if __name__ == '__main__': main()
