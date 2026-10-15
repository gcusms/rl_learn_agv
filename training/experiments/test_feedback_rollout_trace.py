import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

from experiments.feedback_rollout_trace import save_rollout
from neural_navigation.decision_feedback import TRAIN_DIM, WAIT_SCHEMA
from neural_navigation.masked_actions import COMMANDS, masks_from_observation


def sample():
    obs = np.zeros((2, 1, TRAIN_DIM), np.float32)
    obs[..., 235:244] = 1.
    obs[..., 267:269] = 1.
    obs[1, 0, 216] = .25  # The second sampled state is distinguishable.
    scalar = np.zeros((2, 1), np.float32)
    buffer = SimpleNamespace(full=True, observations=obs, actions=np.array([[[1]], [[0]]]),
        action_masks=masks_from_observation(obs[:, 0, :282])[:, None, :],
        log_probs=np.full((2, 1), -2.3), values=scalar.copy(), rewards=scalar.copy(),
        returns=np.array([[1.], [-.5]]), advantages=np.array([[1.], [-.5]]),
        episode_starts=np.array([[1.], [0.]]))
    labels = [dict(sensor_timestamp=.2*i, sampled_choice=choice,
                   nonzero_legal_choices=11, executed_command=COMMANDS[choice].tolist(),
                   executed_reason='none', native_reward=0., true_geodesic_progress_m=0.,
                   case_json='{}', done=False) for i, choice in enumerate([1, 0])]
    return buffer, labels


class RolloutTraceTests(unittest.TestCase):
    def write(self, path, buffer, labels):
        save_rollout(path, buffer, labels, collected_steps=250,
                     optimizer_epochs_before_update=4, feedback_schema=WAIT_SCHEMA)

    def test_original_probabilities_actions_and_bootstrap_are_retained(self):
        buffer, labels = sample()
        # Goal handling may stop a sampled nonzero action. Keep that action and
        # its original probability for PPO instead of relabelling it as zero.
        labels[0]['executed_command'] = [0., 0.]
        labels[0]['executed_reason'] = 'estimated_goal_reached'
        buffer.rewards[1, 0] = .17  # Preserve a bootstrap-adjusted buffer reward.
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'trace.npz'
            self.write(path, buffer, labels)
            with np.load(path, allow_pickle=False) as trace:
                np.testing.assert_array_equal(trace['sampled_actions'], buffer.actions)
                np.testing.assert_array_equal(trace['old_log_prob'], buffer.log_probs)
                np.testing.assert_array_equal(trace['observations'], buffer.observations)
                self.assertEqual(trace['ppo_rewards'][1, 0], buffer.rewards[1, 0])
                actual = json.loads(trace['execution_diagnostics_json'][0, 0])
                self.assertEqual(actual['executed_command'], [0., 0.])
                metadata = json.loads(str(trace['metadata_json']))
                self.assertFalse(metadata['execution_labels_are_policy_inputs'])
                self.assertEqual(metadata['feedback_schema'], WAIT_SCHEMA)

    def test_mismatched_action_or_mask_is_rejected(self):
        for field in ['sampled_choice', 'mask']:
            buffer, labels = sample()
            if field == 'sampled_choice': labels[0]['sampled_choice'] = 0
            else: buffer.action_masks[0, 0, 1] = False
            with tempfile.TemporaryDirectory() as directory:
                with self.assertRaises(ValueError): self.write(Path(directory)/'trace.npz', buffer, labels)

    def test_execution_labels_cannot_cross_reset_boundary(self):
        buffer, labels = sample()
        labels[0]['done'] = True
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError): self.write(Path(directory)/'trace.npz', buffer, labels)
            buffer.episode_starts[1, 0] = 1.
            self.write(Path(directory)/'valid.npz', buffer, labels)

    def test_safety_override_is_not_silently_hidden(self):
        buffer, labels = sample()
        labels[0]['executed_command'] = [0., 0.]
        labels[0]['executed_reason'] = 'laser_braking_footprint'
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError): self.write(Path(directory)/'trace.npz', buffer, labels)


if __name__ == '__main__': unittest.main()
