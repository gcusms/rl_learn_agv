"""Inspect current-ray and retained-point vetoes using recorded sensors only.

The optional multi-frame free-evidence settings are shadows of the existing
memory implementation. They do not alter a runtime, mask, model or command.
"""
from datetime import datetime
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import numpy as np

from navrl.map import StaticMap
from neural_navigation.decision_feedback import GROUP_DIM, STAT_SLICE
from neural_navigation.experiment_budget import ExperimentBudget
from neural_navigation.masked_actions import COMMANDS
from neural_navigation.runtime import EmergencyStop
from neural_navigation.safety_memory import NearFieldMemory
from neural_navigation.sensors import LidarSpec


ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
BASE = ROOT / 'reports/stop_feedback_20261015'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def free_evidence(core, points):
    mount = core.lidar.mount
    c, s = np.cos(mount[3]), np.sin(mount[3])
    local = (points - mount[:2]) @ np.array([[c, -s], [s, c]])
    distance = np.linalg.norm(local, axis=1)
    bearing = np.arctan2(local[:, 1], local[:, 0])
    indices = np.clip(np.searchsorted(core.lidar.angles, bearing), 0, len(core.scan) - 1)
    valid = np.isfinite(core.scan) & (core.scan >= core.lidar.range_min) & (core.scan < core.lidar.range_max)
    ranges = np.where(valid, core.scan, 0.)
    observed = np.minimum.reduce([ranges[np.clip(indices + k, 0, len(ranges) - 1)] for k in [-1, 0, 1]])
    supported = (bearing >= core.lidar.angle_min) & (bearing <= core.lidar.angle_max)
    return observed - distance, supported


def main():
    if (OUT / 'RESULTS.json').exists():
        raise FileExistsError('Do not overwrite the memory-replay audit')
    hardware_path = ROOT / 'models/stop_feedback_20261015/early_transfer_bc800_s231015/bc_portable/hardware.json'
    trace_path = BASE / 'actual95_learned_validation_20261016/physical/w11_training_stall_t106.npz'
    map_path = ROOT / 'training/assets/room_navigation_scripted/decision_map.yaml'
    paths = [Path(__file__), hardware_path, trace_path, map_path, map_path.with_suffix('.pgm'),
        ROOT / 'training/neural_navigation/safety_memory.py', ROOT / 'training/neural_navigation/runtime.py',
        ROOT / 'training/neural_navigation/static_footprint.py']
    inputs = {str(p.relative_to(ROOT)): digest(p) for p in paths}
    hardware = json.loads(hardware_path.read_text())
    static = StaticMap(map_path, radius=.3)
    lidar = LidarSpec(hardware['lidar'])
    configurations = {'original': hardware['near_field_memory']}
    for margin in [.03, .04]:
        configurations['confirmed_free_' + str(margin)] = dict(hardware['near_field_memory'],
            free_evidence_margin_m=margin, free_evidence_frames=3, free_evidence_max_gap_s=.4)
    memories = {name: NearFieldMemory(c) for name, c in configurations.items()}
    guard = EmergencyStop(hardware['robot_footprint_m'], memory_config=hardware['near_field_memory'])
    budget = ExperimentBudget(.05, memory_mib=2000, directory=OUT)
    with np.load(trace_path, allow_pickle=False) as z:
        a = {k: z[k].copy() for k in ['labels', 'observations', 'scans', 'odometry',
            'measured_velocity', 'route_diagnostics', 'masks', 'commands']}
    selected = [int(np.argmin(abs(a['labels'][:, 0] - t))) for t in [106., 110., 120., 130., 139.8]]
    rows = []
    for i, label in enumerate(a['labels']):
        now = float(label[0])
        core = SimpleNamespace(scan=a['scans'][i], odom=a['odometry'][i],
            measured_velocity=a['measured_velocity'][i], elapsed=now, estimate=label[4:7],
            static_map=static, lidar=lidar,
            localizer=SimpleNamespace(std=a['route_diagnostics'][i, -3:]))
        raw = a['observations'][i]
        stats = raw[303 + STAT_SLICE.start:303 + STAT_SLICE.stop]
        valid = np.isfinite(core.scan) & (core.scan >= lidar.range_min) & (core.scan < lidar.range_max)
        usable = bool(stats[7] > .5 and valid.any()
            and max(core.localizer.std[:2]) <= .2 and core.localizer.std[2] <= .25)
        remembered = {name: memory.update(core) for name, memory in memories.items()} if usable else {}
        if i not in selected:
            budget.tick()
            continue
        if not usable or not np.isclose(stats[1], 0):
            raise ValueError('Selected memory diagnosis requires valid, current scans and localization')
        angles = lidar.angles[valid] + lidar.mount[3]
        current = core.scan[valid, None] * np.c_[np.cos(angles), np.sin(angles)] + lidar.mount[:2]
        current_reasons = [guard._trajectory_check(core, command, current,
            np.full(len(current), .04), 0., allow_escape=True) for command in COMMANDS]
        groups = raw[303:303 + GROUP_DIM].reshape(12, -1).argmax(-1)
        variants = {}
        for name, (points, uncertainty) in remembered.items():
            combined = np.vstack([current, points])
            margins = np.r_[np.full(len(current), .04), .04 + uncertainty]
            reasons = [guard._trajectory_check(core, command, combined, margins, 0., allow_escape=True)
                for command in COMMANDS]
            if name == 'original':
                if any(a['masks'][i, j] and reasons[j] is not None for j in range(1, 12)):
                    raise ValueError('Original memory reconstruction vetoes a recorded legal action')
                if any(groups[j] == 4 and reasons[j] is None for j in range(1, 12)):
                    raise ValueError('Original memory reconstruction cannot reproduce a laser veto')
            gaps = np.linalg.norm(points - np.clip(points, -guard.half, guard.half), axis=1)
            slack, supported = free_evidence(core, points)
            critical = gaps < .04 + uncertainty
            nearest = []
            for index in np.argsort(gaps)[:6]:
                index = int(index)
                nearest.append(dict(body_point=points[index].tolist(), gap_m=float(gaps[index]),
                    required_margin_m=float(.04 + uncertainty[index]),
                    current_three_neighbor_free_slack_m=float(slack[index]),
                    visible_in_current_scan=bool(supported[index])))
            new_permissions = [j for j in range(1, 12) if not a['masks'][i, j]
                and groups[j] in (3, 4) and reasons[j] is None]
            variants[name] = dict(memory_points=len(points), critical_memory_points=int(critical.sum()),
                geometry_reasons=reasons, geometry_clear_choices=[j for j in range(1, 12) if reasons[j] is None],
                new_geometry_permissions_preserving_recorded_non_geometry_vetoes=new_permissions,
                critical_visible_points_with_slack_gt3cm=int((critical & supported & (slack > .03)).sum()),
                critical_visible_points_with_slack_gt4cm=int((critical & supported & (slack > .04)).sum()),
                nearest_remembered_points=nearest)
        rows.append(dict(time_s=now, original_legal_choices=np.flatnonzero(a['masks'][i]).tolist(),
            current_ray_only_geometry_clear_choices=[j for j in range(1, 12) if current_reasons[j] is None],
            current_ray_only_geometry_reasons=current_reasons,
            original_geometry_acceptance_and_laser_veto_parity_passed=True, variants=variants))
        budget.tick()
    for name, expected in inputs.items():
        assert digest(ROOT / name) == expected
    report = dict(completed=True, at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(), inputs=inputs,
        configurations=configurations, rows=rows, resource=budget.statistics(),
        new_physical_steps=0, new_optimizer_updates=0, new_ppo_steps=0,
        current_runtime_or_running_job_changed=False, masks_or_guard_relaxed=False,
        shadow_hypotheses_not_admitted_for_execution=True,
        limitations=['Only recorded sensor/odometry/map-pose streams are used.',
            'Memory-clearing variants are geometry-only shadows; accepted geometry is not a safe full escape.',
            'Recorded dynamic/perception/localization/projection vetoes are not assumed removed.',
            'Multi-frame free evidence must be separately checked against true hazards and complete physical tasks before any runtime experiment.',
            'Current-ray-only geometry is attribution, never a permission to drop rear memory.'],
        goal_achieved=False, deployment_approved=False)
    (OUT / 'RESULTS.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(dict(completed=True, frames=[dict(time_s=r['time_s'],
        original=r['original_legal_choices'], current_ray_only=r['current_ray_only_geometry_clear_choices'],
        shadows={k:v['new_geometry_permissions_preserving_recorded_non_geometry_vetoes'] for k,v in r['variants'].items()})
        for r in rows], resource=report['resource']), indent=2), flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        (OUT / 'FAILURE.json').write_text(json.dumps(dict(error=repr(error),
            new_physical_steps=0, goal_achieved=False), indent=2) + '\n')
        raise
