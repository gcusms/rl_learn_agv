"""Geometry-only shadow checks on recorded inputs; no commands or simulation."""
from pathlib import Path
from types import SimpleNamespace
import copy
import hashlib
import json
import os

import numpy as np
from navrl.map import StaticMap
from neural_navigation.decision_feedback import GROUP_DIM, STAT_SLICE
from neural_navigation.experiment_budget import ExperimentBudget
from neural_navigation.masked_actions import COMMANDS
from neural_navigation.runtime import EmergencyStop
from neural_navigation.sensors import LidarSpec


ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent


def main():
    os.chdir(ROOT)
    plan = json.loads((OUT/'PLAN.json').read_text())
    for name, expected in plan['inputs'].items():
        assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == expected, name
    budget = ExperimentBudget(.05, memory_mib=2000, directory=OUT)
    hardware = json.loads(Path(plan['hardware']).read_text())
    static = StaticMap(plan['map'], radius=.3)
    lidar = LidarSpec(hardware['lidar'])
    rows = []
    for source in plan['traces']:
        with np.load(source['trace'], allow_pickle=False) as data:
            arrays = {key: data[key].copy() for key in ['labels', 'observations', 'masks',
                'commands', 'scans', 'odometry', 'measured_velocity', 'route_diagnostics']}
        guard = EmergencyStop(hardware['robot_footprint_m'], memory_config=hardware['near_field_memory'])
        # The predictive constraint is deliberately not reconstructed or bypassed:
        # the recorded rejection group preserves its veto in shadow inferences.
        shadow = copy.copy(guard)
        shadow.clearance_increasing_rotation = True
        selected_rows = []
        for i, label in enumerate(arrays['labels']):
            now = float(label[0])
            obs = arrays['observations'][i]
            core = SimpleNamespace(scan=arrays['scans'][i], odom=arrays['odometry'][i],
                measured_velocity=arrays['measured_velocity'][i], elapsed=now, estimate=label[4:7],
                static_map=static, lidar=lidar,
                localizer=SimpleNamespace(std=arrays['route_diagnostics'][i, -3:]))
            stats = obs[303+STAT_SLICE.start:303+STAT_SLICE.stop]
            valid = np.isfinite(core.scan) & (core.scan >= lidar.range_min) & (core.scan < lidar.range_max)
            usable = bool(stats[7] > .5 and valid.any()
                and max(core.localizer.std[:2]) <= .2 and core.localizer.std[2] <= .25)
            if usable:
                remembered, uncertainty = guard.memory.update(core)
            if not any(start-1e-8 <= now < end-1e-8 for start, end in source['intervals_s']):
                budget.tick()
                continue
            if not usable:
                selected_rows.append(dict(time_s=now, skipped='Sensing/localization not suitable for geometry reconstruction'))
                continue
            angle = lidar.angles[valid]+lidar.mount[3]
            current = core.scan[valid, None]*np.c_[np.cos(angle), np.sin(angle)]+lidar.mount[:2]
            points = np.vstack([current, remembered])
            margins = np.r_[np.full(len(current), .04), .04+uncertainty]
            groups = obs[303:303+GROUP_DIM].reshape(12, -1).argmax(-1)
            baseline, changed = [], []
            for choice, command in enumerate(COMMANDS):
                before = guard._trajectory_check(core, command, points, margins, 0., allow_escape=True)
                after = shadow._trajectory_check(core, command, points, margins, 0., allow_escape=True)
                baseline.append(before)
                if before is not None and after is None:
                    changed.append(dict(choice=choice, command=command.tolist(),
                        recorded_rejection_group=int(groups[choice]),
                        original_mask_allowed=bool(arrays['masks'][i, choice]),
                        geometry_before=before, geometry_after=after,
                        prediction_was_already_passed=bool(groups[choice] in (3, 4))))
            # A reconstruction that blocks a recorded accepted command is
            # not trustworthy as a claim about the deployed guard.
            inconsistent = [j for j in range(1, 12)
                if arrays['masks'][i, j] and baseline[j] is not None]
            if inconsistent:
                raise ValueError('Geometry reconstruction rejects recorded legal actions at '+str(now)+': '+str(inconsistent))
            laser_group_consistency = [j for j in range(1, 12)
                if groups[j] == 4 and baseline[j] is None]
            gaps = np.linalg.norm(points-np.clip(points, -guard.half, guard.half), axis=1)
            nearest = []
            for point_index in np.argsort(gaps)[:6]:
                point_index = int(point_index)
                nearest.append(dict(body_point=points[point_index].tolist(),
                    source='current_laser' if point_index < len(current) else 'remembered_endpoint',
                    body_gap_m=float(gaps[point_index]), required_margin_m=float(margins[point_index])))
            selected_rows.append(dict(time_s=now, recorded_legal_actions=np.flatnonzero(arrays['masks'][i]).tolist(),
                recorded_command=arrays['commands'][i].tolist(), geometry_reasons=baseline,
                rotation_shadow_new_geometry_permissions=changed,
                recorded_accepted_actions_all_reproduced=True,
                recorded_laser_rejections_not_reproduced=laser_group_consistency,
                nearest_points=nearest, memory_points=len(remembered)))
            budget.tick()
        rows.append(dict(id=source['id'], trace=source['trace'], frames=selected_rows))
    for name, expected in plan['inputs'].items():
        assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == expected, name
    result = dict(completed=True, geometry_only=True, no_new_physical_steps=True,
        executed_commands_or_models_changed=False, rows=rows, resource=budget.statistics(),
        limitations=['Only supplied sensor, odometry and recorded map-pose streams reconstruct memory and geometry.',
            'Prediction is not re-run; an existing dynamic veto is never treated as removed.',
            'Recorded accepted actions must remain geometry-clear; any inconsistency fails the audit.',
            'Geometry acceptance is not a safe complete maneuver or a demonstration.',
            'The rotation option is a shadow; no hardware config or mask was changed.'],
        goal_achieved=False, deployment_approved=False)
    (OUT/'RESULTS.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(completed=True, summaries=[dict(id=r['id'],
        frames=len(r['frames']), newly_geometry_clear_frames=sum(bool(f.get('rotation_shadow_new_geometry_permissions')) for f in r['frames']),
        inconsistent_laser_group_frames=sum(bool(f.get('recorded_laser_rejections_not_reproduced')) for f in r['frames'])) for r in rows],
        resource=result['resource']), indent=2), flush=True)


if __name__ == '__main__':
    main()
