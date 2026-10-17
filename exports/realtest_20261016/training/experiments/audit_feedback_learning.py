"""Read-only response audit on captured development failure observations.

These inputs are off-policy for newly trained actors. This audit cannot prove
guard acceptance, navigation improvement or generalization; run closed loop.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from stable_baselines3 import PPO
from experiments.feasibility_feedback_ppo import feasibility_loss
from neural_navigation.safety_context import OBSERVATION_DIM


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(trace_path, policies):
    trace = json.loads(trace_path.read_text())
    rows = []
    for row in trace['samples']:
        command = np.asarray(row['command'])
        flags = row['safety_context']['action_feasibility']
        if row['stop_reason'] != 'estimated_goal_reached' and abs(command[0]) < .01 and abs(command[1]) < .05 and any(flags):
            rows.append(row)
    if not rows:
        raise ValueError('No recorded stationary frames with accepted probes')
    inputs = np.asarray([row['policy_observation'] for row in rows], dtype=np.float32)
    if inputs.shape[1] != OBSERVATION_DIM or not np.isfinite(inputs).all():
        raise ValueError('Finite captured V3 inputs required')
    observations = torch.tensor(inputs)
    results = []
    for policy in policies:
        model = PPO.load(policy, device='cpu')
        with torch.no_grad():
            distribution = model.policy.get_distribution(observations).distribution
            mean = distribution.mean
            loss, fraction = feasibility_loss(observations, mean)
            action = mean.clamp(-1, 1)
            prediction = model.predict(inputs, deterministic=True)[0]
            np.testing.assert_allclose(prediction, action.numpy(), rtol=0, atol=2e-6)
        results.append(dict(policy=str(policy.resolve()), policy_sha256=digest(policy),
            eligible_fraction=float(fraction), auxiliary_loss=float(loss),
            mean_clipped_proposal=action.mean(dim=0).tolist(),
            forward_proposal_fraction=float((action[:, 0] > .05).float().mean()),
            left_turn_proposal_fraction=float((action[:, 1] > .1).float().mean()),
            right_turn_proposal_fraction=float((action[:, 1] < -.1).float().mean())))
    return dict(case_id=trace['case']['id'], trace_sha256=digest(trace_path),
        captured_frames=len(rows), frame_selection='Recorded idle with any accepted probe; excludes reached goal',
        policies=results, limitation='Off-policy captured inputs, not a closed-loop or safety evaluation. '
        'The loss measures distance to a finite accepted probe set; it cannot certify arbitrary proposed actions.')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--trace', type=Path, required=True)
    parser.add_argument('--policy', type=Path, action='append', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Fresh output required')
    torch.set_num_threads(1)
    result = audit(args.trace, args.policy)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
