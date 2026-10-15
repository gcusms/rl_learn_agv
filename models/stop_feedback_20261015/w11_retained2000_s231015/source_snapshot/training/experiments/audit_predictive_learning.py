"""Read-only V4 sensitivity audit on a sensor-derived training replay.

The optional replay uses a frozen actor and recorded training reset. Other
actors are queried off-policy, which cannot prove closed-loop improvement.
No observation perturbation or command from the audit is sent to a robot.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from stable_baselines3 import PPO
from experiments.fixed_training_replay import reset_training_episode
from experiments.recovery_curriculum import RecoveryNavigationCurriculum
from neural_navigation.safety_context import OBSERVATION_DIM, PREDICTIVE_DIM


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def capture(training_log, step, policy):
    rows = [json.loads(line) for line in training_log.read_text().splitlines()]
    selected = [row for row in rows if row.get('step') == step and 'training_case_json' in row and 'worker' in row]
    if len(selected) != 1:
        raise ValueError('One actual recorded training episode required')
    case = json.loads(selected[0]['training_case_json'])
    model = PPO.load(policy, device='cpu')
    env = RecoveryNavigationCurriculum(policy.parent, stage='robust', motion_profiles=True,
        footprint_clearance_cost=1.)
    samples = []
    before_digest = digest(policy)
    try:
        obs, _ = reset_training_episode(env, case)
        if obs.shape != (PREDICTIVE_DIM,):
            raise ValueError('Native V4 observation required')
        for _ in range(env.max_steps):
            action = model.predict(obs, deterministic=True)[0]
            row = dict(timestamp=env.active.elapsed, policy_observation=obs.tolist(),
                proposed_action=action.tolist())
            obs, _, done, truncated, info = env.step(action)
            row.update(command=info['commanded_velocity'].tolist(),
                stop_reason=info['execution_guard_reason'])
            samples.append(row)
            if done or truncated:
                break
        outcome = dict(safe_success=bool(info['verified_mission_success']),
            collisions=int(info['collision_events']), timeout=bool(truncated))
        if digest(policy) != before_digest:
            raise ValueError('Checkpoint changed during capture')
        return dict(kind='training_sensor_replay_v4', training_log_sha256=digest(training_log),
            training_step=step, case=case, policy_sha256=before_digest,
            samples=samples, outcome=outcome, simulator_truth_in_actor_inputs=False)
    finally:
        env.close()


def audit(trace, policies):
    if trace.get('kind') != 'training_sensor_replay_v4':
        raise ValueError('Explicit training replay provenance required')
    observations = np.asarray([row['policy_observation'] for row in trace['samples']], np.float32)
    if observations.ndim != 2 or observations.shape[1] != PREDICTIVE_DIM or not np.isfinite(observations).all():
        raise ValueError('Nonempty finite native V4 observations required')
    supported = observations[:, 281] > .5
    parking_risk = supported & (observations[:, 269] > 0.)
    alternatives = ((observations[:, 235:244] > .5).any(axis=1)
        | (observations[:, 267:269] > .5).any(axis=1))
    output = []
    for path in policies:
        before_digest = digest(path)
        model = PPO.load(path, device='cpu')
        if model.observation_space.shape != (PREDICTIVE_DIM,):
            raise ValueError('V4 checkpoint required')
        actual = model.predict(observations, deterministic=True)[0]
        ablated = observations.copy(); ablated[:, OBSERVATION_DIM:] = 0.
        delta = np.abs(actual-model.predict(ablated, deterministic=True)[0])
        weights = model.policy.mlp_extractor.policy_net[0].weight.detach().cpu().numpy()
        if digest(path) != before_digest:
            raise ValueError('Checkpoint changed during audit')
        output.append(dict(policy=str(path.resolve()), policy_sha256=before_digest,
            added_actor_column_l2=np.linalg.norm(weights[:, OBSERVATION_DIM:], axis=0).tolist(),
            max_action_delta=float(delta.max()), mean_action_delta=float(delta.mean()),
            supported_risk_mean_delta=(float(delta[parking_risk].mean()) if parking_risk.any() else None)))
    return dict(frames=len(observations), supported_track_frames=int(supported.sum()),
        stationary_forecast_risk_frames=int(parking_risk.sum()),
        stationary_risk_with_current_alternative_frames=int((parking_risk & alternatives).sum()),
        actors=output, limitation='Zero-feature ablation is off-distribution; sensitivity is not '
        'understanding, a safety certificate, or closed-loop improvement. No audit data enters PPO.')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--trace', type=Path, required=True)
    parser.add_argument('--training-log', type=Path)
    parser.add_argument('--step', type=int)
    parser.add_argument('--policy', type=Path, action='append', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists(): parser.error('Fresh audit output required')
    torch.set_num_threads(1)
    if args.training_log:
        if args.trace.exists() or args.step is None: parser.error('Fresh capture path and training step required')
        trace = capture(args.training_log, args.step, args.policy[0])
        args.trace.parent.mkdir(parents=True, exist_ok=True)
        args.trace.write_text(json.dumps(trace, allow_nan=False))
    else:
        trace = json.loads(args.trace.read_text())
    result = audit(trace, args.policy)
    result['trace_sha256'] = digest(args.trace)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps(result, indent=2))


if __name__ == '__main__': main()
