"""Freeze and diagnose a completed paired loss without new physics or training.

The two completed rollouts remain held-out evaluation data. A reconstructed
laser-geometry check is explanatory only; it never authorizes a command and
does not turn a fixed-pose alternative into a successful counterfactual.
"""
import copy
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / 'reports/stop_feedback_20261015'
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE / 'actual106_resolution_audit_20261017'))
from audit import EARLY, geometry
from neural_navigation.decision_feedback import FEEDBACK_REASONS, GROUPS
from neural_navigation.experiment_budget import ExperimentBudget
from neural_navigation.masked_actions import COMMANDS
from neural_navigation.safety_memory import NearFieldMemory
from neural_navigation.sensors import LidarSpec


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    target = HERE / 'WHOLE_MISSION_FROZEN_REGRESSION_AUDIT.json'
    assert not target.exists(), 'Preserve the completed audit'
    study = BASE / 'native625_local_speedup_retention_20261017'
    captured = (study / 'RESULTS.json').read_bytes()
    report = json.loads(captured)
    case_id = 'pre_ppo_holdout_dynamic_01'
    rows = {r['variant']: r for r in report['frozen_rows'] if r['case_id'] == case_id}
    assert set(rows) == {'parent', 'candidate'}
    assert rows['parent']['safe_success'] and rows['candidate']['timeout']
    assert all(not any(r[k] for k in ['collisions', 'false_arrival', 'physical_guards'])
               for r in rows.values())
    budget = ExperimentBudget(.1, memory_mib=550, directory=HERE)
    files = [Path(__file__), study / 'PLAN.json', study / 'DEPENDENCIES.json',
             BASE / 'actual106_resolution_audit_20261017/audit.py',
             BASE / 'native625_frozen_holdout_20261016/PLAN.json',
             BASE / 'native625_frozen_holdout_20261016/CASES.json']
    traces = {}
    for variant, row in rows.items():
        path = ROOT / row['trace_path']
        assert digest(path) == row['trace_sha256']
        files.append(path)
        with np.load(path, allow_pickle=False) as saved:
            traces[variant] = {key: saved[key].copy() for key in saved.files}
    old, new = traces['parent'], traces['candidate']
    common = min(len(old['choices']), len(new['choices']))
    changed = np.flatnonzero(old['choices'][:common] != new['choices'][:common])
    assert len(changed)
    first = int(changed[0])
    exact = {key: bool(np.array_equal(old[key][:first + 1], new[key][:first + 1], equal_nan=True))
             for key in ['observations', 'masks', 'scans', 'odometry', 'measured_velocity',
                         'labels', 'human_positions']}
    exact.update(previous_choices=bool(np.array_equal(old['choices'][:first], new['choices'][:first])),
                 previous_commands=bool(np.array_equal(old['commands'][:first], new['commands'][:first])))
    assert all(exact.values()), exact
    for variant, row in rows.items():
        assert traces[variant]['masks'][first, traces[variant]['choices'][first]]
    old_bundle, new_bundle = [ROOT / rows[v]['bundle'] for v in ['parent', 'candidate']]
    unchanged = ['hardware.json', 'localizer.pt', 'router.pt', 'parent_actor.pt', 'previous_risk_actor.pt',
                 'localization_kernel.so', 'localization_contract.json', 'uncertainty_contract.json',
                 'docking_contract.json', 'decision_feedback.json', 'release_observation.json',
                 'long_risk_observation.json', 'goal_pose_observation.json', 'quality_observation.json']
    for name in unchanged:
        assert digest(old_bundle / name) == digest(new_bundle / name), name
        files += [old_bundle / name, new_bundle / name]
    files += [old_bundle / 'actor.pt', new_bundle / 'actor.pt']
    hardware = json.loads((old_bundle / 'hardware.json').read_text())
    assert hardware['near_field_memory'].get('motion_response_schema') is None
    assert hardware['near_field_memory']['static_footprint_margin_m'] == .04
    lidar = LidarSpec(hardware['lidar'])
    half = np.asarray(hardware['robot_footprint_m']) / 2
    configs = dict(original=hardware['near_field_memory'],
                   three_frame_evidence=dict(hardware['near_field_memory'],
                       free_evidence_margin_m=.03, free_evidence_frames=3, free_evidence_max_gap_s=.4))
    memories = {k: NearFieldMemory(v) for k, v in configs.items()}
    legal = new['masks'][:, 1:].any(1)
    suffix = len(legal)
    while suffix and not legal[suffix - 1]:
        suffix -= 1
    assert suffix < len(legal)
    selected = sorted(set([150, 200, 250, 300, 400, 490]
                          + np.linspace(suffix, len(legal) - 2, 12).astype(int).tolist()))
    reconstructions = []
    matched = checked = 0
    for frame in range(max(selected) + 1):
        now = float(new['labels'][frame, 0])
        core = SimpleNamespace(elapsed=now, odom=new['odometry'][frame], scan=new['scans'][frame], lidar=lidar)
        reason = FEEDBACK_REASONS[int(new['observations'][frame + 1, 399:419].argmax())]
        for name, memory in memories.items():
            if frame in selected:
                shadow = copy.deepcopy(memory)
                points, uncertainty = shadow.update(core)
                seen = np.array([entry[1] for entry in shadow.entries.values()])
                valid = np.isfinite(core.scan) & (core.scan >= lidar.range_min) & (core.scan < lidar.range_max)
                angle = lidar.angles[valid] + lidar.mount[3]
                current = core.scan[valid, None] * np.c_[np.cos(angle), np.sin(angle)] + lidar.mount[:2]
                probes = []
                groups = new['observations'][frame, 303:399].reshape(12, 8).argmax(1)
                for action in [4, 5]:
                    result = geometry(np.vstack([current, points]),
                                      np.r_[np.full(len(current), .04), .04 + uncertainty],
                                      new['measured_velocity'][frame], COMMANDS[action], half,
                                      len(current), seen, now, configs[name])
                    raw = geometry(current, np.full(len(current), .04),
                                   new['measured_velocity'][frame], COMMANDS[action], half,
                                   len(current), np.empty(0), now, configs[name])
                    if name == 'original' and groups[action] in [0, 4]:
                        checked += 1
                        matched += int(bool(new['masks'][frame, action]) == result['laser_geometry_clear'])
                        assert bool(new['masks'][frame, action]) == result['laser_geometry_clear'], (frame, action)
                    probes.append(dict(action=action, recorded_group=GROUPS[int(groups[action])],
                                       recorded_legal=bool(new['masks'][frame, action]),
                                       raw_returns_geometry_clear=raw['laser_geometry_clear'], **result))
                reconstructions.append(dict(frame=frame, time_s=now, variant=name,
                    legal_recorded_choices=np.flatnonzero(new['masks'][frame]).tolist(),
                    remembered_points=len(points), probes=probes))
                if reason not in EARLY:
                    memories[name] = shadow
            elif reason not in EARLY:
                memory.update(core)
            if reason == 'odometry_jump':
                memories[name].reset()
        if frame % 25 == 0:
            budget.tick()
    assert checked and checked == matched
    suffix_groups = new['observations'][suffix:, 303:399].reshape(-1, 12, 8).argmax(2)
    reason_counts = {GROUPS[i]: int(np.count_nonzero(suffix_groups[:, 1:] == i)) for i in range(8)}
    for name in ['decision_feedback.py', 'safety_context.py', 'safety_memory.py', 'sensors.py',
                 'masked_actions.py', 'motion_response.py', 'experiment_budget.py', 'runtime.py']:
        files.append(ROOT / 'training/neural_navigation' / name)
    files.append(ROOT / 'training/navrl/geometry.py')
    budget.tick()
    result = dict(completed=True, at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(), case_id=case_id,
        completed_paired_rows=rows, mutable_study_report_sha256_at_capture=hashlib.sha256(captured).hexdigest(),
        first_choice_difference=dict(frame=first, sensor_time_s=float(old['labels'][first, 0]),
            parent_choice=int(old['choices'][first]), candidate_choice=int(new['choices'][first]),
            parent_command=old['commands'][first].tolist(), candidate_command=new['commands'][first].tolist(),
            legal_choices=np.flatnonzero(old['masks'][first]).tolist(), exact_common_inputs=exact),
        final_only_stop_suffix=dict(start_frame=suffix, start_time_s=float(new['labels'][suffix, 0]),
            frames=len(legal)-suffix, every_nonstop_action_vetoed=True, rejection_groups=reason_counts,
            predicted_moving_tracks_max=float(new['constraint_diagnostics'][suffix:, 0].max()),
            predicted_veto_count_max=float(new['constraint_diagnostics'][suffix:, 1].max()),
            physical_xy_displacement_m=float(np.linalg.norm(new['labels'][-1, 1:3]-new['labels'][suffix, 1:3])),
            maximum_localization_error_m=float(np.linalg.norm(new['labels'][suffix:, 1:3]-new['labels'][suffix:, 4:6], axis=1).max()),
            maximum_route_endpoint_error_m=float(np.linalg.norm(new['route_diagnostics'][suffix:, 2:4]-[7.125, 5.725], axis=1).max())),
        current_geometry_checks=checked, current_geometry_matches=matched,
        geometry_samples=reconstructions,
        unchanged_component_files=unchanged,
        actor_sha256={v: digest(ROOT / r['bundle'] / 'actor.pt') for v, r in rows.items()},
        original_failed_106s_recovery_unresolved=True, not_a_monotonic_model_improvement=True,
        observed_loss_blocks_automatic_replacement=True,
        no_heldout_trace_admitted_to_training=True, no_new_physics=True, new_ppo_steps=0,
        new_optimizer_updates=0, runtime_hardware_and_guard_unchanged=True,
        no_closed_loop_success_or_full_guard_claim_for_alternative_memory=True,
        goal_achieved=False, deployment_approved=False,
        resource=budget.statistics(), inputs={str(p.relative_to(ROOT)): digest(p) for p in set(files)})
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({k: result[k] for k in ['case_id', 'first_choice_difference',
        'final_only_stop_suffix', 'current_geometry_checks', 'current_geometry_matches', 'resource']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
