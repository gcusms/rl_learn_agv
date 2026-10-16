"""Verify a finished700-frame replay and recover its failed JSON report.

The original replay completed and saved its arrays before JSON serialization
failed on a NumPy boolean. Preserve that failure and every pinned original
input. Recompute local geometry and all exported decisions from saved arrays;
do not rerun the sensor runtime, physics, or any optimizer.
"""
from datetime import datetime
import hashlib
import importlib.util
import json
import os
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import torch

from navrl.geometry import integrate
from neural_navigation.experiment_budget import ExperimentBudget
from neural_navigation.masked_actions import ACTIONS, COMMANDS, masks_from_observation
from neural_navigation.motion_response import force_response_trajectory


ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
BASE = ROOT / 'reports/stop_feedback_20261015'
SOURCE = BASE / 'actual106_memory_context_20261016'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    if (OUT / 'RESULTS.json').exists():
        raise FileExistsError('Do not overwrite verified replay evidence')
    torch.set_num_threads(1)
    plan = json.loads((SOURCE / 'PLAN.json').read_text())
    failure = json.loads((SOURCE / 'FAILURE.json').read_text())
    assert failure['error'] == "TypeError('Object of type bool_ is not JSON serializable')"
    for name, expected in plan['inputs'].items():
        assert digest(ROOT / name) == expected, name
    spec = importlib.util.spec_from_file_location('original_memory_context_audit', SOURCE / 'replay.py')
    old = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(old)
    hardware = json.loads((old.MODEL / 'hardware.json').read_text())
    fixtures = old.sensor_fixtures(hardware)
    trace_path = BASE / 'actual95_learned_validation_20261016/physical/w11_training_stall_t106.npz'
    with np.load(SOURCE / 'replay.npz', allow_pickle=False) as z:
        replay = {key: z[key].copy() for key in z.files}
    with np.load(trace_path, allow_pickle=False) as z:
        source = {key: z[key].copy() for key in ['labels', 'observations', 'masks', 'choices',
                                               'commands', 'human_positions', 'measured_velocity']}
    n = len(source['choices'])
    assert n == 700 and replay['observations'].shape == (700, 439)
    np.testing.assert_array_equal(replay['times'], source['labels'][:, 0])
    np.testing.assert_array_equal(replay['recorded_choices'], source['choices'])
    np.testing.assert_array_equal(replay['recorded_commands'], source['commands'])
    np.testing.assert_array_equal(replay['masks'], masks_from_observation(replay['observations'][:, :282]))
    np.testing.assert_array_equal(replay['observations'][:, :235], source['observations'][:, :235])
    before = replay['times'] < 106. - 1e-8
    np.testing.assert_array_equal(replay['observations'][before], source['observations'][before])
    actor = torch.jit.load(str(old.MODEL / 'actor.pt'), map_location='cpu').eval()
    budget = ExperimentBudget(.15, memory_mib=2048, directory=OUT)
    for i, obs in enumerate(replay['observations']):
        with torch.no_grad():
            command = actor(torch.from_numpy(obs[None])).cpu().numpy()[0]
        choice = int(replay['independent_model_proposals'][i])
        np.testing.assert_array_equal(command, ACTIONS[choice])
        assert replay['masks'][i, choice]
        budget.tick()
    case = json.loads((BASE / 'actual_stop_validation_20261016/actual_prefix.json').read_text())['case']
    scene = json.loads((ROOT / 'training' / case['scene'] / 'physical_scene.json').read_text())
    half = np.asarray(hardware['robot_footprint_m']) / 2
    walls = np.asarray(scene['walls'], float)
    radii = np.asarray([item['radius_m'] for item in scene['dynamic_obstacles']])
    local = []
    for i, label in enumerate(source['labels']):
        if label[0] < 106. - 1e-8 or i % 5:
            continue
        for value in np.flatnonzero(replay['masks'][i] & ~source['masks'][i]):
            choice = int(value)
            command = COMMANDS[choice]
            times = np.linspace(0., .2, 41)
            nominal = np.array([integrate(np.zeros(3), *command, t) for t in times])
            response = force_response_trajectory(source['measured_velocity'][i], command, .2, samples=41)
            minima = {}
            c, s = np.cos(label[3]), np.sin(label[3])
            rotation = np.array([[c, -s], [s, c]])
            for name, poses in [('nominal', nominal), ('finite_response', response)]:
                gaps = []
                for j, pose in enumerate(poses):
                    world = np.r_[label[1:3] + pose[:2] @ rotation.T, label[3] + pose[2]]
                    other = source['human_positions'][min(i + 1, n - 1)]
                    humans = source['human_positions'][i] + (other - source['human_positions'][i]) * j / 40
                    gaps.append(old.physical_gap(world, half, walls, humans, radii))
                minima[name] = float(min(gaps))
            local.append(dict(time_s=float(label[0]), choice=choice,
                independent_model_proposes_this=bool(replay['independent_model_proposals'][i] == choice),
                minimum_truth_gap_m=minima, samples=41, horizon_s=.2,
                nonphysical_counterfactual_geometry_only=True))
        budget.tick()
    selected = ~before
    proposals = replay['independent_model_proposals']
    report = dict(completed=True, at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
        source_plan_sha256=digest(SOURCE / 'PLAN.json'), source_failure=failure,
        source_replay=str((SOURCE / 'replay.npz').relative_to(ROOT)), source_replay_sha256=digest(SOURCE / 'replay.npz'),
        original_pinned_inputs_verified=len(plan['inputs']), original_runtime_replay_restarted=False,
        original_json_failure_preserved=True, fixtures=fixtures,
        frozen_prefix_frames=int(before.sum()), frozen_prefix_observations_bit_identical=True,
        all700_base235_features_bit_identical=True, all700_saved_actor_proposals_exactly_reproduced=True,
        recorded_choices_and_commands_bit_identical=True,
        original_pose_and_execution_assertions_passed_before_array_archival=True,
        original_exact_maximum_pose_difference_not_retained=True,
        post106_frames=int(selected.sum()), independent_motion_proposals=int(np.count_nonzero(proposals[selected])),
        independent_motion_choice_counts={str(int(c)): int(v) for c, v in zip(*np.unique(proposals[selected], return_counts=True))},
        frames_with_new_geometry_permissions=int((replay['masks'][selected] & ~source['masks'][selected]).any(1).sum()),
        local_geometry_checks=local, resource=budget.statistics(),
        new_physical_steps=0, new_optimizer_updates=0, new_ppo_steps=0,
        physical_runtime_hypothesis_admitted=False, goal_achieved=False, deployment_approved=False,
        limitations=['Recorded commands, not independent model proposals, generated this replay history.',
            'Truth only audits local geometry; it is not a policy or localization input.',
            '0.2s geometry with interpolated baseline pedestrian motion is not complete reactive physics or escape.',
            'Repaired JSON serialization only; no model, mask or source runtime modification.'])
    for name, expected in plan['inputs'].items():
        assert digest(ROOT / name) == expected, name
    (OUT / 'RESULTS.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(dict(completed=True, independent_motion_proposals=report['independent_motion_proposals'],
        frames_with_new_geometry_permissions=report['frames_with_new_geometry_permissions'], local_checks=len(local),
        minimum_local_geometry_gap_m=min([min(row['minimum_truth_gap_m'].values()) for row in local], default=None),
        resource=report['resource']), indent=2), flush=True)


if __name__ == '__main__':
    os.chdir(ROOT)
    try:
        main()
    except Exception as error:
        (OUT / 'FAILURE.json').write_text(json.dumps(dict(error=repr(error),
            new_physical_steps=0, automatically_restarted=False), indent=2) + '\n')
        raise
