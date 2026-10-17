import hashlib
import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from stable_baselines3 import PPO
from experiments.replay_feedback_ppo import ReplayFeedbackPPO
from test_prior_anchored_ppo import TinyNavigation
from neural_navigation.safety_context import BASE_DIM, OBSERVATION_DIM, REASON_START, REASONS


class ReplayFeedbackTests(unittest.TestCase):
    def model(self):
        torch.set_num_threads(1)
        return ReplayFeedbackPPO('MlpPolicy', TinyNavigation(), anchor_strength=.1,
            device='cpu', n_steps=8, batch_size=8, n_epochs=1, seed=238,
            policy_kwargs=dict(net_arch=dict(pi=[8], vf=[8])))

    def collection(self, directory, count=10, **changes):
        obs = np.zeros((count, OBSERVATION_DIM), np.float32)
        obs[:, BASE_DIM] = 1.; obs[:, BASE_DIM+14] = 1.; obs[:, -1] = 1.
        obs[:, BASE_DIM+REASON_START+REASONS.index('laser_braking_footprint')] = 1.
        path = Path(directory)/'sensor_feedback.npz'
        np.savez_compressed(path, observations=obs, episode_ids=np.arange(count)%2,
            proposed_actions=np.tile([0., -1.], (count, 1)).astype(np.float32),
            commanded_velocity=np.zeros((count, 2), np.float32),
            current_reasons=np.full(count, 'laser_braking_footprint', dtype='U48'))
        meta = dict(status='complete', kind='training', ppo_rollout_eligible=False,
            simulator_truth_in_samples=False, evaluation_manifest_consumed=False,
            collection_schema='rejected_sensor_feedback_v2',
            sample_count=count, sample_file_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        meta.update(changes)
        (Path(directory)/'status.json').write_text(json.dumps(meta))

    def test_replay_has_gradient_when_live_rollout_feedback_is_exempt(self):
        model = self.model()
        with tempfile.TemporaryDirectory() as directory:
            self.collection(directory)
            info = model.load_feedback(directory)
            self.assertEqual(info['retained_frames'], 2)
            self.assertFalse(info['ppo_returns_or_advantages_replayed'])
            loss, fraction = model.reference_loss(torch.zeros((8, OBSERVATION_DIM)))
            self.assertEqual(fraction.item(), 0.)
            self.assertGreater(loss.item(), 0.)
            loss.backward()
            self.assertGreater(model.policy.action_net.bias.grad.abs().sum().item(), 0.)
            model.learn(8)
            self.assertEqual(model.last_replay_statistics['eligible_fraction'], 1.)
            model.save(directory+'/policy.zip')
            restored = PPO.load(directory+'/policy.zip', device='cpu')
            inputs = np.zeros(OBSERVATION_DIM, np.float32)
            np.testing.assert_array_equal(model.predict(inputs, deterministic=True)[0],
                restored.predict(inputs, deterministic=True)[0])
            self.assertFalse(hasattr(restored, 'feedback_inputs'))

    def test_incomplete_wrong_role_empty_or_corrupted_data_rejected(self):
        for changes in [dict(status='running'), dict(kind='evaluation'),
            dict(ppo_rollout_eligible=True), dict(simulator_truth_in_samples=True),
            dict(evaluation_manifest_consumed=True), dict(sample_file_sha256='bad')]:
            with tempfile.TemporaryDirectory() as directory:
                self.collection(directory, **changes)
                with self.assertRaises(ValueError): self.model().load_feedback(directory)
        with tempfile.TemporaryDirectory() as directory:
            self.collection(directory, count=0)
            with self.assertRaises(ValueError): self.model().load_feedback(directory)

    def test_fault_or_already_executed_motion_is_not_recovery_replay(self):
        for field, value in [('current_reasons', np.full(10, 'scan_unavailable', dtype='U48')),
            ('commanded_velocity', np.tile([0., -.5], (10, 1)).astype(np.float32))]:
            with tempfile.TemporaryDirectory() as directory:
                self.collection(directory)
                path = Path(directory)/'sensor_feedback.npz'
                with np.load(path, allow_pickle=False) as archive:
                    data = {key: archive[key].copy() for key in archive.files}
                data[field] = value; np.savez_compressed(path, **data)
                meta = json.loads((Path(directory)/'status.json').read_text())
                meta['sample_file_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
                (Path(directory)/'status.json').write_text(json.dumps(meta))
                with self.assertRaises(ValueError): self.model().load_feedback(directory)


if __name__ == '__main__':
    unittest.main()
