"""Auxiliary sensor-feasibility replay, separate from on-policy PPO returns."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from experiments.feasibility_feedback_ppo import FeasibilityFeedbackPPO, feasibility_loss, FEEDBACK_REASONS
from neural_navigation.safety_context import OBSERVATION_DIM


class ReplayFeedbackPPO(FeasibilityFeedbackPPO):
    def __init__(self, *args, **kwargs):
        self.feedback_inputs = None
        self.feedback_generator = None
        self.last_replay_statistics = {}
        super().__init__(*args, **kwargs)

    def _excluded_save_params(self):
        return super()._excluded_save_params()+['feedback_inputs', 'feedback_generator']

    def load_feedback(self, directory, seed=92120):
        directory = Path(directory)
        meta = json.loads((directory/'status.json').read_text())
        path = directory/'sensor_feedback.npz'
        if (meta.get('status') != 'complete' or meta.get('kind') != 'training'
            or meta.get('ppo_rollout_eligible') is not False
            or meta.get('simulator_truth_in_samples') is not False
            or meta.get('evaluation_manifest_consumed') is not False
            or meta.get('collection_schema') != 'rejected_sensor_feedback_v2'):
            raise ValueError('Completed sensor-only training collection required')
        if hashlib.sha256(path.read_bytes()).hexdigest() != meta.get('sample_file_sha256'):
            raise ValueError('Feedback collection digest mismatch')
        with np.load(path, allow_pickle=False) as data:
            inputs = data['observations'].copy()
            episodes = data['episode_ids'].copy()
            command = data['commanded_velocity'].copy()
            proposed = data['proposed_actions'].copy()
            reasons = data['current_reasons'].copy()
        if inputs.shape != (meta['sample_count'], OBSERVATION_DIM) or not len(inputs) or not np.isfinite(inputs).all() or episodes.shape != (len(inputs),):
            raise ValueError('Nonempty finite captured input set required')
        if command.shape != (len(inputs), 2) or proposed.shape != command.shape or reasons.shape != (len(inputs),) or not np.isfinite(np.r_[command.ravel(), proposed.ravel()]).all():
            raise ValueError('Captured finite executed and proposed commands required')
        requested = proposed*np.array([.2, .5])
        requested[:, 0] = np.where(proposed[:, 0] >= 0, proposed[:, 0]*.2, proposed[:, 0]*.08)
        if not (np.isin(reasons, FEEDBACK_REASONS).all()
            and (abs(command[:, 0]) < .01).all() and (abs(command[:, 1]) < .05).all()
            and np.any(abs(requested-command) > 1e-8, axis=1).all()):
            raise ValueError('Every sample must be a currently rejected stationary geometry proposal')
        tensor = torch.tensor(inputs, dtype=torch.float32, device=self.device)
        _, fraction = feasibility_loss(tensor, torch.zeros((len(tensor), 2), device=self.device))
        if fraction.item() < 1.-1e-6:
            raise ValueError('Collection contains ineligible feedback observations')
        # Consecutive stalled frames are highly correlated. Thin each episode
        # deterministically and sample an episode uniformly before its frames.
        self.feedback_inputs = []
        for episode in np.unique(episodes):
            indices = np.flatnonzero(episodes == episode)[::5]
            self.feedback_inputs.append(tensor[indices])
        self.feedback_generator = np.random.default_rng(seed)
        return dict(sample_sha256=meta['sample_file_sha256'], source_frames=len(inputs),
            retained_frames=sum(len(x) for x in self.feedback_inputs),
            source_episodes=len(self.feedback_inputs), seed=seed,
            ppo_returns_or_advantages_replayed=False)

    def reference_loss(self, observations):
        on_policy_loss, fraction = super().reference_loss(observations)
        if not self.feedback_inputs:
            raise ValueError('Load validated sensor feedback before replay learning')
        count = min(64, len(observations))
        chosen = []
        for _ in range(count):
            episode = self.feedback_inputs[int(self.feedback_generator.integers(len(self.feedback_inputs)))]
            chosen.append(episode[int(self.feedback_generator.integers(len(episode)))])
        replay = torch.stack(chosen)
        mean = self.policy.get_distribution(replay).distribution.mean
        replay_loss, replay_fraction = feasibility_loss(replay, mean)
        self.last_replay_statistics = dict(loss=float(replay_loss.detach()),
            eligible_fraction=float(replay_fraction), sampled_frames=count)
        # Only the auxiliary feasibility objective consumes off-policy inputs.
        # PPO likelihood ratios, rewards and GAE remain the live rollout's.
        return on_policy_loss+.5*replay_loss, fraction
