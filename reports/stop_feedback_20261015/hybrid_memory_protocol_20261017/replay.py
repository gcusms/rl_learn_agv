"""Fixed-recorded-trajectory evidence replay, not closed-loop navigation.

Original/three-frame/hybrid memory receive exactly the same measured scans,
input odometry and actual execution-commit schedule. Pure-turn laser geometry
uses the runtime's original constant-command nine-sample trajectory.
"""
import copy
from datetime import datetime
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import numpy as np

from hybrid_memory import HybridFreeEvidenceMemory, contract
from neural_navigation.decision_feedback import FEEDBACK_REASONS
from neural_navigation.experiment_budget import ExperimentBudget
from neural_navigation.safety_memory import NearFieldMemory
from neural_navigation.sensors import LidarSpec


ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
BASE = HERE.parent
EARLY = {'nonfinite_input','stale_or_future_scan','scan_unavailable','odometry_jump',
         'localization_uncertain','predicted_dynamic_contact','predictive_command_contract'}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def turn_geometry(current, remembered, uncertainty, half, yaw):
    points = np.vstack([current,remembered])
    margins = np.r_[np.full(len(current),.04),.04+uncertainty]
    for angle in np.linspace(0.,yaw*.2,9):
        c,s = np.cos(angle),np.sin(angle)
        relative = points @ np.array([[c,-s],[s,c]])
        gaps = np.linalg.norm(relative-np.clip(relative,-half,half),axis=1)
        if np.any(gaps < margins-1e-4):
            return False
    return True


def main():
    assert not (HERE/'REPLAY.json').exists()
    hardware_path = ROOT/'models/stop_feedback_20261015/guard_aligned_migration_s231017/portable/hardware.json'
    hardware = json.loads(hardware_path.read_text())
    old_path = BASE/'normal_suite_wait_validation_v2_20261016/RESULTS.json'
    strict_path = BASE/'memory_from_start_20261017/RESULTS.json'
    original,strict = json.loads(old_path.read_text()),json.loads(strict_path.read_text())
    specs = [(r,'original') for r in original['groups']['hard3']['rows']
             if r['case_id'] in ('legal_wait_parent_t90','w11_training_stall_t106')]
    specs += [(r,'three_frame') for r in strict['rows']]
    assert len(specs) == 4
    paths = [Path(__file__),HERE/'hybrid_memory.py',hardware_path,old_path,strict_path,
        ROOT/'training/neural_navigation/safety_memory.py',ROOT/'training/neural_navigation/runtime.py',
        ROOT/'training/neural_navigation/safety_context.py',ROOT/'training/neural_navigation/sensors.py']
    spec = LidarSpec(hardware['lidar'])
    half = np.asarray(hardware['robot_footprint_m'])/2
    config = hardware['near_field_memory']
    assert config.get('motion_response_schema') is None
    assert not config.get('clearance_increasing_rotation',False)
    budget = ExperimentBudget(.1,memory_mib=500,directory=HERE)
    results = []
    for row,source_kind in specs:
        path = ROOT/row['trace_path']
        assert digest(path) == row['trace_sha256']
        paths.append(path)
        with np.load(path,allow_pickle=False) as saved:
            z = {k:saved[k].copy() for k in ['labels','scans','odometry','observations','masks']}
        memories = dict(original=NearFieldMemory(config),
            three_frame=NearFieldMemory(dict(config,free_evidence_margin_m=.03,
                free_evidence_frames=3,free_evidence_max_gap_s=.4)),
            hybrid=HybridFreeEvidenceMemory(config))
        counts = dict(source_accepted_turn_checks=0,source_laser_rejected_turn_checks=0,
            source_laser_geometry_mismatches=0,three_frame_blocks_where_original_clear=0,
            hybrid_restores_those_original_clear_checks=0,hybrid_additional_clear_checks=0)
        peak_points = dict.fromkeys(memories,0)
        geometry = {name:0 for name in memories}
        examples = []
        for i,now in enumerate(z['labels'][:,0]):
            core = SimpleNamespace(elapsed=float(now),odom=z['odometry'][i],scan=z['scans'][i],lidar=spec)
            reason = (FEEDBACK_REASONS[int(z['observations'][i+1,399:419].argmax())]
                      if i+1<len(z['labels']) else None)
            remembered = {}
            for name,memory in memories.items():
                shadow = copy.deepcopy(memory)
                remembered[name] = shadow.update(core)
                if reason == 'odometry_jump':
                    memory.reset()
                elif reason is not None and reason not in EARLY:
                    memories[name] = shadow
                peak_points[name] = max(peak_points[name],len(shadow.entries))
            if now < row['release_sensor_timestamp_s']-1e-8:
                if i%20==0:
                    budget.tick()
                continue
            valid = np.isfinite(core.scan)&(core.scan>=spec.range_min)&(core.scan<spec.range_max)
            angles = spec.angles[valid]+spec.mount[3]
            current = core.scan[valid,None]*np.c_[np.cos(angles),np.sin(angles)]+spec.mount[:2]
            groups = z['observations'][i,303:399].reshape(12,8).argmax(1)
            for action,yaw in [(4,.5),(5,-.5)]:
                clear = {name:turn_geometry(current,*values,half,yaw) for name,values in remembered.items()}
                for name,is_clear in clear.items():
                    geometry[name] += int(not is_clear)
                if z['masks'][i,action]:
                    counts['source_accepted_turn_checks'] += 1
                    counts['source_laser_geometry_mismatches'] += int(not clear[source_kind])
                elif groups[action] == 4:
                    counts['source_laser_rejected_turn_checks'] += 1
                    counts['source_laser_geometry_mismatches'] += int(clear[source_kind])
                added = clear['original'] and not clear['three_frame']
                restored = added and clear['hybrid']
                counts['three_frame_blocks_where_original_clear'] += int(added)
                counts['hybrid_restores_those_original_clear_checks'] += int(restored)
                counts['hybrid_additional_clear_checks'] += int(not clear['original'] and clear['hybrid'])
                if (restored or (not clear['original'] and clear['hybrid'])) and len(examples)<8:
                    examples.append(dict(time_s=float(now),action=action,clear=clear,
                        memory_point_counts={name:len(value[0]) for name,value in remembered.items()}))
            if i%20==0:
                (HERE/'REPLAY_STATUS.json').write_text(json.dumps(dict(case_id=row['case_id'],
                    source=source_kind,frame=i+1,total=len(z['labels'])))+'\n')
                budget.tick()
        result = dict(case_id=row['case_id'],source_memory=source_kind,source_safe_success=row['safe_success'],
            source_trace=str(path.relative_to(ROOT)),source_release_s=row['release_sensor_timestamp_s'],
            recorded_frames=len(z['labels']),geometry_rejections=geometry,counts=counts,
            peak_memory_points=peak_points,examples=examples)
        results.append(result)
        print(json.dumps(result,ensure_ascii=False),flush=True)
    budget.tick()
    result = dict(completed=True,at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
        contract=contract(),rows=results,
        exact_source_accepted_and_laser_rejected_turn_geometry_reproduced=all(
            r['counts']['source_laser_geometry_mismatches']==0 for r in results),
        same_recorded_sensor_and_execution_commit_schedule=True,
        no_truth_or_map_used_to_clear_memory=True,nominal_runtime_turn_trajectory_used=True,
        memory_from_initialization_not_timed_handover=True,
        pure_turn_laser_geometry_only_not_all_guard_or_closed_loop=True,
        no_new_physics=True,new_ppo_steps=0,new_optimizer_updates=0,
        inputs={str(p.relative_to(ROOT)):digest(p) for p in paths},resource=budget.statistics(),
        goal_achieved=False,deployment_approved=False)
    (HERE/'REPLAY.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    (HERE/'REPLAY_STATUS.json').write_text(json.dumps(dict(status='complete',
        source_geometry_reproduced=result['exact_source_accepted_and_laser_rejected_turn_geometry_reproduced']))+'\n')
    assert result['exact_source_accepted_and_laser_rejected_turn_geometry_reproduced']


if __name__ == '__main__':
    main()
