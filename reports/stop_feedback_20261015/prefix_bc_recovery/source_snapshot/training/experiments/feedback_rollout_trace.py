"""Archive pre-update PPO samples separately from physical execution labels.

These labels are training diagnostics, not extra observations or commands.
The original sampled action/log-probability/mask are never replaced by an
executed emergency stop. Terminal bootstrap rewards remain as SB3 stored them.
"""
import json
from pathlib import Path

import numpy as np

from neural_navigation.decision_feedback import TRAIN_DIM, contract
from neural_navigation.masked_actions import COMMANDS, masks_from_observation


def save_rollout(path, buffer, diagnostics, *, collected_steps,
                 optimizer_epochs_before_update, feedback_schema):
    contract(feedback_schema)
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    observations = np.asarray(buffer.observations)
    if not buffer.full or observations.ndim != 3 or observations.shape[-1] != TRAIN_DIM:
        raise ValueError('Complete unflattened feedback rollout required')
    shape = observations.shape[:2]
    n = int(np.prod(shape))
    if len(diagnostics) != n:
        raise ValueError('Execution labels must match every original sampled state')
    actions = np.asarray(buffer.actions)
    if actions.shape != (*shape, 1) or not np.isfinite(actions).all():
        raise ValueError('Discrete original actions required')
    choices = actions[..., 0].astype(np.int64)
    if not np.array_equal(choices, actions[..., 0]) or np.any((choices < 0) | (choices >= 12)):
        raise ValueError('Twelve discrete choices required')
    masks = np.asarray(buffer.action_masks)
    if masks.shape != (*shape, 12) or not np.isin(masks, [0., 1.]).all():
        raise ValueError('Original twelve-choice masks required')
    expected = masks_from_observation(observations.reshape(n, TRAIN_DIM)[:, :282])
    if not np.array_equal(masks.reshape(n, 12), expected):
        raise ValueError('Archived mask differs from the sampled observation')
    if not masks.reshape(n, 12)[np.arange(n), choices.ravel()].all():
        raise ValueError('Illegal sampled action')
    arrays = dict(observations=observations, sampled_actions=actions, action_masks=masks)
    for source, name in [('log_probs', 'old_log_prob'), ('values', 'old_values'),
                         ('rewards', 'ppo_rewards'), ('returns', 'returns'),
                         ('advantages', 'advantages'), ('episode_starts', 'episode_starts')]:
        value = np.asarray(getattr(buffer, source))
        if value.shape != shape or not np.isfinite(value).all():
            raise ValueError('Invalid PPO rollout array: ' + source)
        arrays[name] = value
    if not np.isfinite(observations).all():
        raise ValueError('Finite observations required')
    dones = np.asarray([row['done'] for row in diagnostics], bool).reshape(shape)
    if not np.array_equal(arrays['episode_starts'][1:].astype(bool), dones[:-1]):
        raise ValueError('Execution labels cross an incorrect episode boundary')
    for i, row in enumerate(diagnostics):
        if (row['sampled_choice'] != choices.ravel()[i]
                or row['nonzero_legal_choices'] != expected[i, 1:].sum()):
            raise ValueError('Execution label changed the original action or choices')
        command = np.asarray(row['executed_command'], float)
        if command.shape != (2,) or not np.isfinite(command).all():
            raise ValueError('Finite actual command required')
        if (choices.ravel()[i] != 0 and row['executed_reason'] != 'estimated_goal_reached'
                and not np.allclose(command, COMMANDS[choices.ravel()[i]], rtol=0, atol=2e-8)):
            raise ValueError('Nonzero sampled action was changed by the guard')
    metadata = dict(schema='feedback_rollout_before_update_v1',
                    collected_steps=int(collected_steps),
                    optimizer_epochs_before_update=int(optimizer_epochs_before_update),
                    feedback_schema=feedback_schema,
                    execution_labels_are_policy_inputs=False,
                    sampled_actions_replaced_with_execution=False,
                    terminal_bootstrap='ppo_rewards preserve the rollout buffer after SB3 bootstrap',
                    scope='Training diagnostic samples; not independent navigation evaluation.')
    np.savez_compressed(path, **arrays,
                        execution_diagnostics_json=np.asarray([json.dumps(row) for row in diagnostics]).reshape(shape),
                        metadata_json=np.asarray(json.dumps(metadata)))
