"""Soft-prior trial with read-only actual recovery sample coverage diagnostics."""
import json
import runpy
from pathlib import Path
import numpy as np
import torch
import experiments.train_alignment_recovery as alignment
from experiments.recovery_sample_metrics import RecoverySampleMetrics


class MeasuredRecoveryProgress(alignment.AlignmentProgress):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.recovery_sample_metrics = RecoverySampleMetrics()

    def _on_step(self):
        # MaskablePPO calls callbacks before updating _last_obs; these are the
        # observations and masks used to sample this exact executed decision.
        observations = np.asarray(self.model._last_obs)
        masks = np.asarray(self.locals['action_masks'])
        actions = np.asarray(self.locals['actions']).reshape(-1)
        with torch.no_grad():
            probabilities = self.model.policy.get_distribution(
                torch.as_tensor(observations, device=self.model.device),
                action_masks=masks).distribution.probs.cpu().numpy()
        for observation, mask, action, probability, done in zip(
                observations, masks, actions, probabilities, self.locals['dones']):
            self.recovery_sample_metrics.record(observation.tolist(), mask.tolist(),
                                               int(action), probability.tolist())
            if done:
                self.recovery_sample_metrics.end_episode()
        result = super()._on_step()
        if self.num_timesteps % 250 == 0:
            self._save_metrics()
        return result

    def _save_metrics(self):
        target = Path(self.output)/'recovery_sample_metrics.json'
        target.write_text(json.dumps(self.recovery_sample_metrics.summary(), indent=2))

    def _on_training_end(self):
        super()._on_training_end()
        self._save_metrics()


if __name__ == '__main__':
    alignment.AlignmentProgress = MeasuredRecoveryProgress
    runpy.run_module('experiments.train_soft_prior_recovery', run_name='__main__')
