"""Rebuild causal observations under a shadow memory-evidence hypothesis.

Recorded commands keep the sensor replay on its original trajectory. The
saved network's independent proposals are diagnostics, never executed actions
or successful rollouts. Truth is used only by a separate local geometry audit.
"""
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import torch

from navrl.geometry import integrate
from neural_navigation.decision_feedback_runtime import StopFeedbackRuntime
from neural_navigation.experiment_budget import ExperimentBudget
from neural_navigation.masked_actions import ACTIONS, COMMANDS, masks_from_observation
from neural_navigation.motion_response import force_response_trajectory
from neural_navigation.safety_memory import NearFieldMemory
from neural_navigation.sensors import LidarSpec


ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
BASE = ROOT / 'reports/stop_feedback_20261015'
MODEL = ROOT / 'models/stop_feedback_20261015/actual95_transfer_bc400_s231016/bc_portable'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def physical_gap(pose, half, walls, humans, radii):
    """Exact circle/rectangle gap and conservative SAT wall separation."""
    c, s = np.cos(pose[2]), np.sin(pose[2])
    rotation = np.array([[c, -s], [s, c]])
    axes = np.vstack([rotation.T, np.eye(2)])
    centers = (walls[:, :2] + walls[:, 2:]) / 2
    extents = (walls[:, 2:] - walls[:, :2]) / 2
    robot_projection = np.abs(axes @ rotation) @ half
    wall_projection = np.abs(axes) @ extents.T
    separation = np.abs(axes @ (centers - pose[:2]).T) - robot_projection[:, None] - wall_projection
    wall_gap = float(separation.max(0).min())
    local = (humans - pose[:2]) @ rotation
    human_gap = float((np.linalg.norm(local - np.clip(local, -half, half), axis=1) - radii).min())
    return min(wall_gap, human_gap)


def sensor_fixtures(hardware):
    from types import SimpleNamespace
    lidar = LidarSpec(hardware['lidar'])
    config = dict(hardware['near_field_memory'], free_evidence_margin_m=.03,
                  free_evidence_frames=3, free_evidence_max_gap_s=.4)
    core = SimpleNamespace(elapsed=0., odom=np.zeros(3), estimate=np.zeros(3),
                           lidar=lidar, scan=np.full(lidar.beams, 2.))
    memory = NearFieldMemory(config)
    key = (20, 0)
    memory.entries[key] = (np.array([.8, 0.]), 0.)
    memory.update(core); memory.update(core)
    assert key in memory.entries
    core.elapsed = .2; memory.update(core)
    assert key in memory.entries
    core.elapsed = .4; core.scan[:] = np.nan; memory.update(core)
    core.scan[:] = 2.
    for t in [.6, .8]:
        core.elapsed = t; memory.update(core); assert key in memory.entries
    core.elapsed = 1.; memory.update(core); assert key not in memory.entries
    # A map-pose correction and elapsed time do not clear a rear measurement.
    rear_key = (-8, 0)
    memory.entries[rear_key] = (np.array([-.3, 0.]), 1.)
    core.estimate[:] = [4., 5., .7]
    for t in [2., 4., 20.]:
        core.elapsed = t; memory.update(core); assert rear_key in memory.entries
    # A nearer finite surface is occlusion, not positive free evidence.
    core.estimate[:] = 0; memory.entries[key] = (np.array([.8, 0.]), core.elapsed)
    core.scan[:] = .5
    for t in [20.2, 20.4, 20.6]:
        core.elapsed = t; memory.update(core); assert key in memory.entries
    return dict(duplicate_time_not_counted=True, unknown_resets_confirmation=True,
                three_distinct_free_frames_required=True, unseen_rear_retained=True,
                map_pose_correction_does_not_move_memory=True, occluding_surface_does_not_clear=True)


def main():
    if (OUT / 'PLAN.json').exists():
        raise FileExistsError('Do not overwrite a causal memory replay')
    torch.set_num_threads(1)
    trace_path = BASE / 'actual95_learned_validation_20261016/physical/w11_training_stall_t106.npz'
    case_path = BASE / 'actual_stop_validation_20261016/actual_prefix.json'
    case = json.loads(case_path.read_text())['case']
    scene_path = ROOT / 'training' / case['scene'] / 'physical_scene.json'
    scene = json.loads(scene_path.read_text())
    hardware = json.loads((MODEL / 'hardware.json').read_text())
    manifest = json.loads((MODEL / 'manifest.json').read_text())
    paths = [Path(__file__), trace_path, case_path, scene_path, MODEL / 'manifest.json']
    paths += [MODEL / name for name in manifest['files']]
    for folder in ['navrl', 'neural_navigation']:
        paths += list((ROOT / 'training' / folder).rglob('*.py'))
    inputs = {str(p.relative_to(ROOT)): digest(p) for p in paths}
    plan = dict(at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(), inputs=inputs,
        hypothesis='3cm free-ray evidence in three distinct contiguous scans, activated only at106s',
        activation_s=106., unchanged_collision_margin_m=.04, recorded_commands_used_for_replay=True,
        no_simulator_truth_in_runtime=True, physical_geometry_used_only_in_offline_evaluation=True,
        resource=dict(soft_cpu_cores=.15, hard_cpu_percent=20, memory_high_mib=2048, memory_max_mib=3072),
        new_physical_steps=0, new_ppo_steps=0, no_execution_admission=True, goal_achieved=False)
    (OUT / 'PLAN.json').write_text(json.dumps(plan, indent=2) + '\n')
    fixtures = sensor_fixtures(hardware)
    with np.load(trace_path, allow_pickle=False) as data:
        a = {k: data[k].copy() for k in ['labels', 'observations', 'scans', 'odometry',
                'measured_velocity', 'commands', 'choices', 'masks', 'human_positions']}
    runtime = StopFeedbackRuntime(MODEL, ROOT / 'training' / case['scene'] / 'decision_map.yaml',
                                  case['start'], case['goal'], 'shortcut')
    original = runtime.policy.policy
    proposals = []
    class ReplayCommands:
        def predict(self, observation):
            index = len(proposals)
            proposed = original.predict(observation)
            choice = int(np.abs(ACTIONS - proposed).max(1).argmin())
            mask = masks_from_observation(np.asarray(observation)[:282])
            assert np.max(np.abs(ACTIONS[choice] - proposed)) < 2e-6 and mask[choice]
            actual = int(a['choices'][index])
            assert mask[actual], 'A recorded initialization command became illegal'
            proposals.append(choice)
            return ACTIONS[actual].copy()
    runtime.policy.policy = ReplayCommands()
    budget = ExperimentBudget(.15, memory_mib=2048, directory=OUT)
    half = np.asarray(hardware['robot_footprint_m']) / 2
    walls = np.asarray(scene['walls'], float)
    radii = np.asarray([item['radius_m'] for item in scene['dynamic_obstacles']])
    observations, masks, local = [], [], []
    activated = False
    max_prefix_error = 0.
    max_pose_error = 0.
    for i, label in enumerate(a['labels']):
        now = float(label[0])
        if not activated and now >= 106. - 1e-8:
            assert runtime.guard.memory is not None
            runtime.guard.memory.config = dict(runtime.guard.memory.config, free_evidence_margin_m=.03,
                free_evidence_frames=3, free_evidence_max_gap_s=.4)
            runtime.guard.memory.free_confirmations.clear()
            activated = True
        result = runtime.step(a['scans'][i], a['odometry'][i], a['measured_velocity'][i], now,
                              capture_observation=True)
        obs = np.asarray(result['policy_observation'], np.float32)
        mask = masks_from_observation(obs[:282])
        np.testing.assert_allclose(result['command'], a['commands'][i], rtol=0, atol=2e-8)
        error = float(np.max(np.abs(runtime.core.estimate - label[4:7])))
        max_pose_error = max(max_pose_error, error)
        if error > 2e-8:
            raise ValueError('The replay estimate diverged from the recorded sensor-only state')
        if not activated:
            difference = float(np.max(np.abs(obs - a['observations'][i])))
            max_prefix_error = max(max_prefix_error, difference)
            if difference > 2e-6:
                raise ValueError('The unchanged prefix observation failed parity')
        observations.append(obs); masks.append(mask)
        if activated and i % 5 == 0:
            # Geometry belongs to evaluation only. Baseline pedestrian futures
            # are held/interpolated; they are not counterfactual reactive physics.
            novel = np.flatnonzero(mask & ~a['masks'][i])
            for choice in novel:
                command = COMMANDS[choice]
                times = np.linspace(0., .2, 41)
                nominal = np.array([integrate(np.zeros(3), *command, t) for t in times])
                response = force_response_trajectory(a['measured_velocity'][i], command, .2, samples=41)
                minima = {}
                for name, poses in [('nominal', nominal), ('finite_response', response)]:
                    gaps = []
                    c, s = np.cos(label[3]), np.sin(label[3])
                    rotation = np.array([[c, -s], [s, c]])
                    for j, pose in enumerate(poses):
                        world = np.r_[label[1:3] + pose[:2] @ rotation.T, label[3] + pose[2]]
                        other = a['human_positions'][min(i + 1, len(a['labels']) - 1)]
                        human = a['human_positions'][i] + (other - a['human_positions'][i]) * j / 40
                        gaps.append(physical_gap(world, half, walls, human, radii))
                    minima[name] = min(gaps)
                local.append(dict(time_s=now, choice=int(choice), independent_model_proposes_this=proposals[-1] == choice,
                    minimum_truth_gap_m=minima, minimum_samples=41, horizon_s=.2,
                    nonphysical_counterfactual_geometry_only=True))
        if i % 50 == 0:
            (OUT / 'STATUS.json').write_text(json.dumps(dict(status='sensor_replay', frames=i + 1,
                total_frames=len(a['labels']), sensor_time_s=now, activated=activated), indent=2) + '\n')
        budget.tick()
    observations, masks, proposals = np.asarray(observations), np.asarray(masks), np.asarray(proposals)
    selected = a['labels'][:, 0] >= 106. - 1e-8
    np.savez_compressed(OUT / 'replay.npz', observations=observations, masks=masks,
                        independent_model_proposals=proposals, recorded_choices=a['choices'],
                        recorded_commands=a['commands'], times=a['labels'][:, 0])
    for name, expected in inputs.items():
        assert digest(ROOT / name) == expected
    count = int(np.count_nonzero(proposals[selected]))
    report = dict(completed=True, at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(), inputs=inputs,
        fixtures=fixtures, frozen_prefix_max_observation_difference=max_prefix_error,
        all_frames_max_sensor_pose_difference=max_pose_error,
        post106_frames=int(selected.sum()), independent_motion_proposals=count,
        independent_motion_choice_counts={str(int(c)): int(n) for c, n in zip(*np.unique(proposals[selected], return_counts=True))},
        frames_with_new_geometry_permissions=int((masks[selected] & ~a['masks'][selected]).any(1).sum()),
        local_geometry_checks=local, replay_sha256=digest(OUT / 'replay.npz'), resource=budget.statistics(),
        new_physical_steps=0, new_optimizer_updates=0, new_ppo_steps=0,
        current_runtime_files_or_published_models_changed=False, execution_admitted=False,
        limitations=['Recorded commands were replayed even if the independent network proposed motion.',
            'Truth was used only for local geometry evaluation, never observations or command choice.',
            'Local0.2s geometric checks and baseline pedestrian interpolation do not establish full safety or escape.',
            'Memory thresholds remain uncalibrated for real lidar; no relaxation of body/4cm collision margin.'],
        goal_achieved=False, deployment_approved=False)
    (OUT / 'RESULTS.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(dict(completed=True, independent_motion_proposals=count,
        frames_with_new_geometry_permissions=report['frames_with_new_geometry_permissions'],
        local_checks=len(local), resource=report['resource']), indent=2), flush=True)


if __name__ == '__main__':
    os.chdir(ROOT)
    try:
        main()
    except Exception as error:
        (OUT / 'FAILURE.json').write_text(json.dumps(dict(error=repr(error), automatically_restarted=False,
            new_physical_steps=0, goal_achieved=False), indent=2) + '\n')
        raise
