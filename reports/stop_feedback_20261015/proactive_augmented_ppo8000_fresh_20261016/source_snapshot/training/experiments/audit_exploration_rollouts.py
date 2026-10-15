"""Verify the actual sampling law against matching pre-update checkpoints.

The initial checkpoint matches rollout 250. The periodic callback saves at
1000-step boundaries before that rollout's PPO update; the final policy.zip
does not match its preceding rollout and must not be substituted here.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from experiments.feedback_exploration import SCHEMA, contract, temperature
from experiments.stop_feedback_ppo import StopFeedbackPPO
from neural_navigation.decision_feedback import WAIT_SCHEMA
from neural_navigation.masked_actions import masks_from_observation


def verify(model_run):
    rows = []
    for path in sorted((model_run/'rollouts').glob('*.npz')):
        steps = int(path.stem.rsplit('_', 1)[1])
        if steps != 250 and steps % 1000:
            continue  # No exact matching policy was archived for this rollout.
        checkpoint = model_run/('policy_initial.zip' if steps == 250 else f'policy_step_{steps}.zip')
        model = StopFeedbackPPO.load(checkpoint, device='cpu')
        if model.exploration_schema != SCHEMA or model.feedback_schema != WAIT_SCHEMA:
            raise ValueError('An explicit V2 sensor exploration checkpoint is required')
        if model.exploration_contract != contract():
            raise ValueError('Exploration parameterization differs from the checkpoint')
        with np.load(path, allow_pickle=False) as data:
            metadata = json.loads(str(data['metadata_json']))
            if (metadata['feedback_schema'] != WAIT_SCHEMA
                    or metadata['optimizer_epochs_before_update'] != model._n_updates
                    or model.num_timesteps != (0 if steps == 250 else steps)):
                raise ValueError('Checkpoint does not precede this rollout update')
            obs = data['observations'].reshape(-1, 436)
            masks = data['action_masks'].reshape(-1, 12).astype(bool)
            choices = data['sampled_actions'].reshape(-1).astype(int)
            if not np.array_equal(masks, masks_from_observation(obs[:, :282])):
                raise ValueError('The policy input and sampled masks differ')
            if not masks[np.arange(len(obs)), choices].all():
                raise ValueError('A rejected action was sampled')
            with torch.no_grad():
                x = torch.from_numpy(obs)
                distribution = model.policy.get_distribution(x, masks)
                recalculated = distribution.log_prob(torch.from_numpy(choices)).numpy()
                explored = distribution.distribution.probs.detach().clone()
                nominal = model.policy.get_nominal_distribution(x, masks).distribution.probs.detach().clone()
                temperatures = temperature(x)
            error = float(np.max(abs(recalculated-data['old_log_prob'].reshape(-1))))
            if error > 3e-5:
                raise ValueError('Stored likelihood does not match the actual exploration law')
            if (explored.argmax(-1) != nominal.argmax(-1)).any():
                raise ValueError('Positive temperature unexpectedly changed the deterministic command')
            if torch.count_nonzero(explored[~torch.from_numpy(masks)]) or (explored[:, 0] <= 0).any():
                raise ValueError('Exploration changed stop permission or allowed a masked action')
            active = (temperatures > 1.).numpy()
            rows.append(dict(rollout=path.name, checkpoint=checkpoint.name,
                rollout_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                sampled_frames=len(obs), temperature_active_frames=int(active.sum()),
                active_nonzero_samples=int(np.count_nonzero(active & (choices != 0))),
                active_stop_samples=int(np.count_nonzero(active & (choices == 0))),
                maximum_log_probability_error=error,
                mean_nominal_stop_probability=float(nominal[:, 0].mean()),
                mean_exploratory_stop_probability=float(explored[:, 0].mean()),
                maximum_temperature=float(temperatures.max()),
                deterministic_choices_unchanged=True, no_masked_actions=True))
    if not rows:
        raise ValueError('No complete rollout with a matching checkpoint')
    return dict(completed=True, sampled_frames=sum(r['sampled_frames'] for r in rows),
        audited_rollouts=rows, exploration_contract=contract(), deployment_approved=False,
        limitations=['Only rollouts with an exact pre-update checkpoint are checked.',
                     'Exploration and likelihood correctness are not navigation or recovery success.'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    report = verify(args.model_run)
    (args.output/'RESULTS.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__': main()
