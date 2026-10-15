"""Inspect learned feedback on frozen sensor inputs, without new physics.

The optional zero-input attribution is outside the observation contract. It
can show numerical input dependence, not safe behavior or a better policy.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from navrl.geometry import ROOT
from neural_navigation.decision_feedback import RISK_SLICE
from neural_navigation.decision_feedback_runtime import StopFeedbackRuntime
from neural_navigation.experiment_budget import ExperimentBudget
from neural_navigation.masked_actions import masks_from_observation
from experiments.stop_feedback_ppo import StopFeedbackPPO


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('initial-bundle', 'trace', 'cases', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, action='append', required=True)
    parser.add_argument('--cpu-cores', type=float, default=.1)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    budget = ExperimentBudget(args.cpu_cores, memory_mib=1536, directory=args.output)
    case = next(c for c in json.loads(args.cases.read_text()) if c['id'] == 'motion_profile_015')
    prior = np.asarray(case['start']) + np.asarray(case.get('initial_prior_offset', [0., 0., 0.]))
    runtime = StopFeedbackRuntime(args.initial_bundle, ROOT / case['scene'] / 'decision_map.yaml',
                                  prior, case['goal'], 'shortcut')
    observations = []
    with np.load(args.trace) as trace:
        times = trace['labels'][:, 0].copy()
        for i, t in enumerate(times):
            result = runtime.step(trace['scans'][i], trace['odometry'][i], trace['measured_velocity'][i],
                                  float(t), capture_observation=True)
            obs = np.asarray(result['policy_observation'], np.float32)
            np.testing.assert_array_equal(obs[:303], trace['observations'][i])
            np.testing.assert_array_equal(result['command'], trace['commands'][i])
            observations.append(np.r_[obs[:300], obs[303:]])
            budget.tick()
    observations = np.asarray(observations, np.float32)
    masks = masks_from_observation(observations[:, :282])
    initial = StopFeedbackPPO.load(args.checkpoint[0], device='cpu')
    with torch.no_grad():
        reference_log = initial.policy.get_distribution(torch.from_numpy(observations), action_masks=masks).distribution.logits
        reference_choice = reference_log.argmax(-1).numpy()
    rows = []
    for checkpoint in args.checkpoint:
        model = StopFeedbackPPO.load(checkpoint, device='cpu')
        with torch.no_grad():
            log = model.policy.get_distribution(torch.from_numpy(observations), action_masks=masks).distribution.logits
            choice = log.argmax(-1).numpy()
            neutral = observations.copy()
            neutral[:, 300:] = 0.
            ablated = model.policy.get_distribution(torch.from_numpy(neutral), action_masks=masks).distribution.logits
            layer = model.policy.mlp_extractor.policy_net[0]
            weights = layer.weight[:, 300:]
            contribution = torch.nn.functional.linear(torch.from_numpy(observations[:, 300:]), weights)
            changed = choice != reference_choice
            result = dict(checkpoint=str(checkpoint), checkpoint_sha256=digest(checkpoint),
                collected_steps=int(model.num_timesteps), optimizer_epochs=int(model._n_updates),
                new_input_weights_l2=float(weights.norm()),
                maximum_new_input_first_layer_contribution=float(contribution.abs().max()),
                changed_choices_on_same_states=int(changed.sum()),
                mean_initial_to_current_kl=float((reference_log.exp() * (reference_log - log)).sum(-1).mean()),
                zero_feedback_attribution_kl=float((log.exp() * (log - ablated)).sum(-1).mean()),
                zero_feedback_changed_choices=int((log.argmax(-1) != ablated.argmax(-1)).sum()),
                critical_frames=[dict(timestamp=float(times[i]), parent_choice=int(reference_choice[i]),
                    checkpoint_choice=int(choice[i]), legal_motion_choices=int(masks[i, 1:].sum()),
                    predicted_contact_score=float(observations[i, 300 + RISK_SLICE.start + choice[i]]))
                    for i in range(len(times)) if 7.4 <= times[i] <= 8.8])
        rows.append(result)
        budget.tick()
        del model
    np.savez_compressed(args.output / 'frozen_states.npz', observations=observations, masks=masks, timestamps=times)
    report = dict(completed=True, rows=rows, frames=len(times),
        trace_sha256=digest(args.trace), initial_bundle_manifest_sha256=digest(args.initial_bundle / 'manifest.json'),
        scope='Fixed old-trajectory policy/weight inspection; no new navigation outcome.',
        zero_feedback_is_valid_observation=False, deployment_approved=False,
        limitations=['A changed choice or nonzero feature weight does not prove useful or safe movement.',
                     'Zero-feedback attribution is a numerical intervention outside the feature contract.',
                     'Checkpoint callbacks run before that rollout update; collected steps and optimizer epochs differ.'],
        resource=budget.statistics())
    (args.output / 'RESULTS.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
