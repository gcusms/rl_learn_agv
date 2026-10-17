"""Reach a training recovery state through its complete physical history.

Warmup is initialization, not PPO experience or an online steering controller.
Every warmup command must be legal and reproduce frozen sensors and geometry;
after reset returns, every command is sampled by the learning model alone.
"""
import hashlib
import json
from pathlib import Path

import gymnasium as gym
import numpy as np

from experiments.evaluate_feedback_recovery import load_prefix
from experiments.fixed_training_replay import reset_training_episode
from navrl.geometry import ROOT


class PrefixRecoveryTraining(gym.Wrapper):
    def __init__(self, env, specifications, output, budget):
        super().__init__(env)
        if not specifications:
            raise ValueError('Explicit frozen training prefixes required')
        self.specifications = specifications
        self.output = Path(output)
        self.budget = budget
        self.feedback_schema = env.feedback_schema
        self.reward_scale = env.reward_scale
        self.reset_index = 0
        self.prefix_index = 0
        self.warmup_count = 0
        self.current_warmup = None
        self.current_prefix_id = None
        self.initialization_frames = 0

    def action_masks(self):
        return self.env.action_masks()

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.reset_index = 0
            self.prefix_index = 0
            gym.Env.reset(self.env.env, seed=seed)
        focused = self.reset_index % 2 == 0 and not options
        self.reset_index += 1
        self.current_warmup = None
        self.current_prefix_id = None
        if not focused:
            return self.env.reset(seed=None, options=options)
        spec = self.specifications[self.prefix_index % len(self.specifications)]
        self.prefix_index += 1
        prefix, poses = load_prefix(spec)
        original = spec['case']
        if original.get('input_odometry', 'noisy_motion_feedback') != 'noisy_motion_feedback':
            raise ValueError('This curriculum requires the recorded measured-motion input')
        case = dict(scene=str(ROOT/original['scene']), seed=original['seed'], stage='robust',
            options=dict(start=original['start'], goal=original['goal'][:2]),
            interventions={k: v for k, v in original.items() if k not in
                ['id', 'category', 'scene', 'seed', 'start', 'goal', 'pilot_category', 'max_steps']},
            source_training_case_id='physical_prefix_'+spec['id'])
        base = self.env.env
        if base.max_steps != original['max_steps']:
            raise ValueError('Keep the source physical task horizon during initialization')
        curriculum_rng = base.np_random
        self.env.feedback.reset()
        self.env.prediction.reset()
        obs, info = reset_training_episode(base, case)
        base.np_random = curriculum_rng
        obs = self.env._augment(self.env.prediction.augment(obs, base.core))
        observations, commands, reasons, times, masks = [], [], [], [], []
        scans, odometry, velocities, physical_poses = [], [], [], []
        for i, timestamp in enumerate(prefix['times']):
            core = base.core
            if abs(core.elapsed-timestamp) > 1e-8:
                raise ValueError('Training warmup skipped a sensor frame')
            for name, value in [('scans', core.scan), ('odometry', core.odom),
                    ('measured_velocity', core.measured_velocity)]:
                np.testing.assert_allclose(value, prefix[name][i], rtol=0, atol=2e-8, equal_nan=True)
            np.testing.assert_allclose(base.active.truth, poses[i], rtol=0, atol=2e-8)
            scans.append(np.asarray(core.scan).copy())
            odometry.append(np.asarray(core.odom).copy())
            velocities.append(np.asarray(core.measured_velocity).copy())
            physical_poses.append(base.active.truth.copy())
            choice = int(prefix['choices'][i])
            mask = self.env.action_masks()
            if not mask[choice]:
                raise ValueError('Do not force an inadmissible training initialization command')
            observations.append(obs.copy())
            times.append(float(timestamp))
            masks.append(mask.copy())
            obs, _, terminated, truncated, info = self.env.step(choice)
            if terminated or truncated or info['collision_events'] or info['nonzero_choice_changed_by_guard']:
                raise ValueError('Training prefix did not safely reach its recorded release state')
            commands.append(np.asarray(info['commanded_velocity']).copy())
            reasons.append(info['decision_feedback_previous_reason'])
            self.initialization_frames += 1
            self.budget.tick()
        np.testing.assert_allclose(base.active.truth, poses[-1], rtol=0, atol=2e-8)
        if abs(base.core.elapsed-spec['release_s']) > 1e-8:
            raise ValueError('Training release time mismatch')
        observations.append(obs.copy())
        times.append(float(base.core.elapsed))
        physical_poses.append(base.active.truth.copy())
        self.warmup_count += 1
        folder = self.output/'warmups'
        folder.mkdir(exist_ok=True)
        path = folder/f'episode_{self.warmup_count:05d}.npz'
        metadata = dict(schema='physical_recovery_initialization_v1', prefix_id=spec['id'],
            source_trace=spec['trace'],
            source_trace_sha256=hashlib.sha256(Path(spec['trace']).read_bytes()).hexdigest(),
            source_release_s=spec['release_s'], case=case, sensor_and_physical_parity_verified=True,
            initialization_is_ppo_experience=False, initialized_frames=len(commands))
        np.savez_compressed(path, observations=np.asarray(observations), times=np.asarray(times),
            choices=prefix['choices'], masks=np.asarray(masks), commands=np.asarray(commands),
            scans=np.asarray(scans), odometry=np.asarray(odometry), measured_velocity=np.asarray(velocities),
            physical_poses=np.asarray(physical_poses),
            executed_reasons=np.asarray(reasons), metadata_json=np.asarray(json.dumps(metadata)))
        self.current_warmup = str(path.relative_to(self.output))
        self.current_prefix_id = spec['id']
        print(json.dumps(dict(physical_training_prefix_ready=spec['id'],
            sensor_seconds=float(base.core.elapsed), initialization_frames=len(commands),
            warmup_path=self.current_warmup)), flush=True)
        return obs, dict(info, physical_training_prefix=spec['id'], initialization_frames=len(commands))

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        return obs, reward, done, truncated, dict(info,
            recovery_initialization_trace=self.current_warmup,
            recovery_prefix_id=self.current_prefix_id)
