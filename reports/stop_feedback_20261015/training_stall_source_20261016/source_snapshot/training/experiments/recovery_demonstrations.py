"""Initialize recovery from checked, complete, physically safe trajectories.

Physical outcomes select training examples; poses/humans/future scans are not
network inputs. These are reused training examples, never independent tests.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from neural_navigation.decision_feedback import STAT_SLICE
from neural_navigation.masked_actions import masks_from_observation, COMMANDS


def convert_observations(observations, masks, commands, times):
    if observations.ndim != 2 or observations.shape[1] != 439:
        raise ValueError('Recorded directional feedback observations required')
    result = np.concatenate([observations[:, :300], observations[:, 303:]], axis=-1).astype(np.float32)
    if not np.array_equal(masks_from_observation(result[:, :282]), masks):
        raise ValueError('Recorded action masks differ from sensor observations')
    if np.any(np.diff(times) <= 0) or not np.isfinite(result).all():
        raise ValueError('Monotonic, finite recorded decisions required')
    # Rebuild V2 waiting from actual executed commands and their original
    # alternatives. Never copy the legacy V1-only-policy-wait timer.
    since = None
    for i, timestamp in enumerate(times):
        result[i, 300+STAT_SLICE.start+3] = 0. if since is None else min((timestamp-since)/5., 1.)
        idle = np.max(np.abs(commands[i])) < 1e-8
        since = (timestamp if since is None else since) if idle and masks[i, 1:].any() else None
    return result


def load_demonstrations(specifications, policy):
    arrays = []
    rows = []
    for specification in specifications:
        report_path = Path(specification['report'])
        report = json.loads(report_path.read_text())
        if not report['completed']:
            raise ValueError('Only completed physical diagnostic reports may supply examples')
        matches = [r for r in report['rows'] if r['specification']['id'] == specification['id']]
        if len(matches) != 1:
            raise ValueError('Unambiguous source episode required')
        row = matches[0]
        if (not row['safe_success'] or not row['physical_arrival'] or row['false_arrival']
                or row['collisions'] or row['physical_guards'] or row['timeout']):
            raise ValueError('Do not imitate a failed or physically unsafe episode')
        path = report_path.parent/(specification['id']+'.npz')
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != row['trace_sha256']:
            raise ValueError('The selected physical trace changed')
        with np.load(path, allow_pickle=False) as data:
            masks = data['masks'].astype(bool)
            commands = data['commands']
            choices = data['choices'].astype(np.int64)
            if not masks[np.arange(len(choices)), choices].all():
                raise ValueError('Illegal demonstration action')
            obs = convert_observations(data['observations'], masks, commands, data['labels'][:, 0])
            away = np.linalg.norm(obs[:, 218:220]*12., axis=-1) > .3
            if not np.allclose(commands[away], COMMANDS[choices[away]], rtol=0, atol=2e-8):
                raise ValueError('A driving demonstration was changed by the guard')
            with torch.no_grad():
                keep = policy.recovery_support(torch.from_numpy(obs)).numpy()
            arrays.append((obs[keep], masks[keep], choices[keep]))
            rows.append(dict(report=str(report_path), id=specification['id'], trace_sha256=digest,
                total_frames=len(choices), recovery_frames=int(keep.sum()),
                nonzero_targets=int(np.count_nonzero(choices[keep])),
                stop_targets=int(np.count_nonzero(choices[keep] == 0))))
    obs, masks, choices = [np.concatenate([a[i] for a in arrays]) for i in range(3)]
    if not len(obs) or not np.any(choices != 0) or not np.any(choices == 0):
        raise ValueError('Both demonstrated recovery movement and valid waiting are required')
    return obs, masks, choices, rows


def initialize_from_demonstrations(model, observations, masks, choices, budget, maximum_updates=400):
    if not 1 <= maximum_updates <= 1000:
        raise ValueError('Bounded demonstration initialization required')
    policy = model.policy
    x = torch.from_numpy(observations)
    labels = torch.from_numpy(choices)
    positive = labels != 0
    policy.set_training_mode(True)
    original_rates = [g['lr'] for g in policy.optimizer.param_groups]
    for group in policy.optimizer.param_groups:
        group['lr'] = 2e-3
    history = []
    try:
        for index in range(maximum_updates):
            distribution = policy.get_distribution(x, masks)
            log_probability = distribution.log_prob(labels)
            loss = -.5*(log_probability[positive].mean()+log_probability[~positive].mean())
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite demonstration loss')
            policy.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), model.max_grad_norm)
            policy.optimizer.step()
            budget.tick()
            with torch.no_grad():
                probabilities = policy.get_distribution(x, masks).distribution.probs.clone()
                target = probabilities[torch.arange(len(labels)), labels]
                row = dict(update=index+1, balanced_cross_entropy=float(loss.detach()),
                    minimum_movement_target_probability=float(target[positive].min()),
                    minimum_wait_target_probability=float(target[~positive].min()),
                    correct_choices=int((probabilities.argmax(-1) == labels).sum()), training_examples=len(labels))
            if (index+1) % 10 == 0 or index == 0:
                history.append(row)
            if row['minimum_movement_target_probability'] >= .9 and row['minimum_wait_target_probability'] >= .9:
                history.append(row)
                break
    finally:
        for group, rate in zip(policy.optimizer.param_groups, original_rates):
            group['lr'] = rate
    return dict(completed=True, updates=index+1, history=history,
        environment_steps=0, not_navigation_acceptance=True,
        physical_outcomes_used_only_to_select_training_examples=True,
        actor_inputs_include_truth_or_future_observations=False)
