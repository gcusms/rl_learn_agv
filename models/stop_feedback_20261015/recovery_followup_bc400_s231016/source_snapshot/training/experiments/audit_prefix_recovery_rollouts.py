"""Audit warmed histories separately from the actual on-policy PPO samples."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from experiments.stop_recovery_policy import StopRecoveryPPO
from experiments.train_prefix_recovery import frozen_base_digest
from neural_navigation.decision_feedback import FEEDBACK_REASONS, REASON_SLICE, STAT_SLICE, WAIT_SCHEMA
from neural_navigation.masked_actions import masks_from_observation, COMMANDS


class History:
    def __init__(self):
        self.reason = 'none'
        self.idle = None
        self.choice_idle = None
        self.time = None

    def check(self, observation, timestamp):
        if self.time is not None and timestamp <= self.time:
            raise ValueError('History skipped reset or repeated sensor time')
        expected = np.zeros(len(FEEDBACK_REASONS), np.float32)
        expected[FEEDBACK_REASONS.index(self.reason)] = 1.
        actual = observation[300+REASON_SLICE.start:300+REASON_SLICE.stop]
        if not np.array_equal(expected, actual):
            raise ValueError('Executed reason is missing, acausal or crosses an episode reset')
        stats = observation[300+STAT_SLICE.start:300+STAT_SLICE.stop]
        for value, since in [(stats[2], self.idle), (stats[3], self.choice_idle)]:
            expected_timer = 0. if since is None else min((timestamp-since)/5., 1.)
            if abs(float(value)-expected_timer) > 2e-6:
                raise ValueError('Waiting timer does not follow its physical execution history')

    def record(self, timestamp, choice, masks, command, reason):
        if reason not in FEEDBACK_REASONS or not masks[choice]:
            raise ValueError('Uncontracted execution or illegal sampled action')
        if choice and reason != 'estimated_goal_reached':
            np.testing.assert_allclose(command, COMMANDS[choice], rtol=0, atol=2e-8)
        idle = np.max(np.abs(command)) < 1e-8 and reason != 'estimated_goal_reached'
        self.idle = (timestamp if self.idle is None else self.idle) if idle else None
        self.choice_idle = (timestamp if self.choice_idle is None else self.choice_idle) if idle and masks[1:].any() else None
        self.time = timestamp
        self.reason = reason


def initialize_history(root, relative_path, first_observation):
    history = History()
    if relative_path is None:
        return history, 0
    path = (root/relative_path).resolve()
    if not path.is_relative_to((root/'warmups').resolve()):
        raise ValueError('Initialization evidence must be inside the model run')
    with np.load(path, allow_pickle=False) as data:
        meta = json.loads(str(data['metadata_json']))
        source = Path(meta['source_trace'])
        if hashlib.sha256(source.read_bytes()).hexdigest() != meta['source_trace_sha256']:
            raise ValueError('Initialization physical source changed')
        with np.load(source, allow_pickle=False) as original:
            n = len(data['commands'])
            if np.count_nonzero(original['labels'][:, 0] < meta['source_release_s']-1e-8) != n:
                raise ValueError('Initialization does not match its source release time')
            for name in ['scans', 'odometry', 'measured_velocity', 'choices']:
                np.testing.assert_allclose(data[name], original[name][:n], rtol=0, atol=2e-8, equal_nan=True)
            np.testing.assert_allclose(data['physical_poses'], original['labels'][:n+1, 1:4], rtol=0, atol=2e-8)
            np.testing.assert_allclose(data['times'], original['labels'][:n+1, 0], rtol=0, atol=1e-8)
        if not np.array_equal(data['masks'], masks_from_observation(data['observations'][:-1, :282])):
            raise ValueError('Initialization masks do not match its actual observations')
        for i in range(n):
            history.check(data['observations'][i], float(data['times'][i]))
            history.record(float(data['times'][i]), int(data['choices'][i]), data['masks'][i],
                data['commands'][i], str(data['executed_reasons'][i]))
        history.check(data['observations'][-1], float(data['times'][-1]))
        np.testing.assert_array_equal(first_observation, data['observations'][-1])
    return history, n


def audit(root):
    root = root.resolve()
    config = json.loads((root/'config.json').read_text())
    history = History()
    expected_start = True
    sampled = warmup_frames = initialized_episodes = supported = moving = waiting = only_stop = 0
    exact = []
    longest_timer = 0.
    for path in sorted((root/'rollouts').glob('*.npz')):
        with np.load(path, allow_pickle=False) as data:
            meta = json.loads(str(data['metadata_json']))
            if meta['feedback_schema'] != WAIT_SCHEMA:
                raise ValueError('Only V2 initialized histories are supported')
            obs = data['observations'][:, 0]
            masks = data['action_masks'][:, 0].astype(bool)
            choices = data['sampled_actions'][:, 0, 0].astype(int)
            starts = data['episode_starts'][:, 0].astype(bool)
            if not np.array_equal(masks, masks_from_observation(obs[:, :282])):
                raise ValueError('PPO masks differ from actual policy inputs')
            labels = [json.loads(s) for s in data['execution_diagnostics_json'][:, 0]]
            for i, label in enumerate(labels):
                if bool(starts[i]) != expected_start:
                    raise ValueError('PPO reset marker differs from the actual episode boundary')
                if starts[i]:
                    history, n = initialize_history(root, label['initialization_trace'], obs[i])
                    warmup_frames += n
                    initialized_episodes += int(n > 0)
                t = float(label['sensor_timestamp'])
                history.check(obs[i], t)
                if int(choices[i]) != label['sampled_choice']:
                    raise ValueError('The stored policy action was substituted')
                history.record(t, int(choices[i]), masks[i], label['executed_command'], label['executed_reason'])
                expected_start = bool(label['done'])
                sampled += 1
                only_stop += int(not masks[i, 1:].any())
                longest_timer = max(longest_timer, float(obs[i, 300+STAT_SLICE.start+3])*5.)
                active = bool(label['recovery_actor_supported'])
                supported += int(active)
                moving += int(active and choices[i] != 0)
                waiting += int(active and choices[i] == 0)
            steps = meta['collected_steps']
            if steps != 250 and steps % 1000:
                continue
            checkpoint = root/('policy_initial.zip' if steps == 250 else f'policy_step_{steps}.zip')
            model = StopRecoveryPPO.load(checkpoint, device='cpu')
            if (model._n_updates != meta['optimizer_epochs_before_update']
                    or model.num_timesteps != (0 if steps == 250 else steps)):
                raise ValueError('Checkpoint is not the actual pre-update sampling policy')
            if frozen_base_digest(model.policy) != config['frozen_base_sha256']:
                raise ValueError('Frozen base changed in an actual training checkpoint')
            with torch.no_grad():
                x = torch.from_numpy(obs)
                distribution = model.policy.get_distribution(x, masks)
                recalculated = distribution.log_prob(torch.from_numpy(choices)).numpy()
                actual_support = model.policy.recovery_support(x).numpy()
            np.testing.assert_array_equal(actual_support, [bool(r['recovery_actor_supported']) for r in labels])
            error = float(np.max(abs(recalculated-data['old_log_prob'][:, 0])))
            if error > 3e-5:
                raise ValueError('Actual PPO old likelihood differs from its original sampling law')
            exact.append(dict(rollout=path.name, checkpoint=checkpoint.name,
                verified_samples=len(obs), maximum_log_probability_error=error,
                frozen_base_unchanged=True))
    if not sampled or not exact:
        raise ValueError('No complete recorded recovery rollout')
    return dict(completed=True, sampled_frames=sampled, causally_verified_sampled_frames=sampled,
        separately_verified_initialization_frames=warmup_frames, initialized_episodes=initialized_episodes,
        recovery_actor_supported_frames=supported, supported_motion_samples=moving,
        supported_wait_samples=waiting, only_stop_frames=only_stop,
        longest_observed_choice_wait_s=longest_timer, exact_sampling_checks=exact,
        initialization_is_counted_as_ppo_experience=False, deployment_approved=False,
        scope='Actual causal histories and pre-update sampling likelihoods; not navigation acceptance.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    result = audit(args.model_run)
    (args.output/'RESULTS.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
