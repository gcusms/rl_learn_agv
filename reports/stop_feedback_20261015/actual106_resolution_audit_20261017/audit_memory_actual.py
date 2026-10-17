"""Correct an earlier memory-only diagnostic's trajectory-model assumption.

Physical validation results are unchanged. Recompute the same recorded vetoes
using the exact geometry trajectory selected by the source hardware contract.
The finite-response model remains in dynamic prediction, not this guard path.
"""
import copy
from datetime import datetime
import json
from pathlib import Path
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import numpy as np

from audit import ROOT, OUT, BASE, EARLY, digest, geometry, write
from neural_navigation.decision_feedback import FEEDBACK_REASONS
from neural_navigation.experiment_budget import ExperimentBudget
from neural_navigation.safety_memory import NearFieldMemory
from neural_navigation.sensors import LidarSpec


def main():
    output = OUT / 'MEMORY_GEOMETRY_CORRECTION.json'
    assert not output.exists()
    old_path = BASE / 'native625_learning_diagnostic_20261017/MEMORY_GEOMETRY_AUDIT.json'
    old = json.loads(old_path.read_text())
    hardware_path = ROOT / 'models/stop_feedback_20261015/normal_suite_wait_pair_v2_s231016/wait_continuation/portable/hardware.json'
    hardware = json.loads(hardware_path.read_text())
    assert hardware['near_field_memory'].get('motion_response_schema') is None
    lidar = LidarSpec(hardware['lidar'])
    half = np.asarray(hardware['robot_footprint_m'])/2
    configs = dict(original=hardware['near_field_memory'],
        finite_free_3cm_3frames=dict(hardware['near_field_memory'], free_evidence_margin_m=.03,
            free_evidence_frames=3, free_evidence_max_gap_s=.4))
    prefix_path = BASE / 'context_broad_diagnostic_20261016/physical/hard3_w11_training_stall_t106.npz'
    snapshot_path = BASE / 'native625_learning_diagnostic_20261017/FIRST3_GEOMETRY_ROWS.json'
    rows = json.loads(snapshot_path.read_text())
    paths = [Path(__file__), OUT/'audit.py', old_path, hardware_path, prefix_path, snapshot_path,
        ROOT/'training/neural_navigation/safety_memory.py', ROOT/'training/neural_navigation/runtime.py',
        ROOT/'training/neural_navigation/motion_response.py', ROOT/'training/navrl/geometry.py']
    paths += [ROOT/row['rollout_path'] for row in rows]
    hashes = {str(path.relative_to(ROOT)): digest(path) for path in paths}
    budget = ExperimentBudget(.1, memory_mib=550, directory=OUT)
    with np.load(prefix_path, allow_pickle=False) as saved:
        prefix = {key:saved[key].copy() for key in ['labels','odometry','scans','observations']}
    initial = {name:NearFieldMemory(config) for name,config in configs.items()}
    for index in range(450):
        core = SimpleNamespace(elapsed=float(prefix['labels'][index,0]),
            odom=prefix['odometry'][index], scan=prefix['scans'][index], lidar=lidar)
        reason = FEEDBACK_REASONS[int(prefix['observations'][index+1,399:419].argmax())]
        for memory in initial.values():
            if reason == 'odometry_jump':memory.reset()
            if reason not in EARLY:memory.update(core)
        if index % 25 == 0:budget.tick()
    results = []
    for row in rows:
        path = ROOT/row['rollout_path']
        assert digest(path) == row['rollout_sha256']
        with np.load(path,allow_pickle=False) as saved:
            data = {key:saved[key].copy() for key in saved.files}
        memories = copy.deepcopy(initial)
        counts = {name:dict(reconstructed_rejections=0,current_raw_clear_rejections=0) for name in configs}
        tested = missed = raw_clear = 0
        for index, scan in enumerate(data['raw_scans']):
            core = SimpleNamespace(elapsed=float(data['labels'][index,0]),
                odom=data['input_odometry'][index],scan=scan,lidar=lidar)
            reason = str(data['stop_reasons'][index])
            remembered = {}
            shadows = {}
            for name,memory in memories.items():
                shadow = copy.deepcopy(memory)
                points,uncertainty = shadow.update(core)
                seen = np.array([entry[1] for entry in shadow.entries.values()])
                assert len(points) == len(seen)
                remembered[name] = (points,uncertainty,seen)
                shadows[name] = shadow
            if not data['masks'][index,1:].any():
                valid = np.isfinite(scan)&(scan>=lidar.range_min)&(scan<lidar.range_max)
                angle = lidar.angles[valid]+lidar.mount[3]
                current = scan[valid,None]*np.c_[np.cos(angle),np.sin(angle)]+lidar.mount[:2]
                velocity = data['input_measured_velocity'][index]
                for action,command in [(4,np.array([0.,.5])),(5,np.array([0.,-.5]))]:
                    if data['per_action_guard_reasons'][index,action] != 'laser_braking_footprint':continue
                    tested += 1
                    raw = geometry(current,np.full(len(current),.04),velocity,command,half,
                        len(current),np.empty(0),core.elapsed,configs['original'])
                    raw_is_clear = raw['laser_geometry_clear']
                    raw_clear += int(raw_is_clear)
                    for name,(points,uncertainty,seen) in remembered.items():
                        combined = geometry(np.vstack([current,points]),
                            np.r_[np.full(len(current),.04),.04+uncertainty],velocity,command,
                            half,len(current),seen,core.elapsed,configs[name])
                        blocked = not combined['laser_geometry_clear']
                        counts[name]['reconstructed_rejections'] += int(blocked)
                        counts[name]['current_raw_clear_rejections'] += int(blocked and raw_is_clear)
                        if name == 'original':missed += int(not blocked)
            for name in memories:
                if reason not in EARLY:memories[name] = shadows[name]
                if reason == 'odometry_jump':memories[name].reset()
            if index % 20 == 0:budget.tick()
        assert missed == 0, (row['episode'],missed)
        previous = next(item for item in old['episodes'] if item['episode'] == row['episode'])
        results.append(dict(episode=row['episode'],actual_laser_rejected_turns=tested,
            original_reconstruction_misses=missed,current_returns_alone_clear=raw_clear,
            variants=counts,previous_finite_response_diagnostic=previous['variants']))
    for path,expected in hashes.items():assert digest(ROOT/path) == expected,path
    budget.tick()
    result = dict(completed=True,at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),episodes=results,
        guard_geometry_trajectory='nominal constant command selected by absent motion_response_schema',
        hardware_predictive_constraint_still_uses_finite_response=True,
        original474checks_matched=sum(row['actual_laser_rejected_turns'] for row in results)==474,
        old_report_preserved=True,correction_scope='Offline trajectory approximation and its derived counts only',
        old_closed_loop_physical_outcomes_not_reinterpreted=True,
        inputs=hashes,resource=budget.statistics(),new_physical_steps=0,new_ppo_steps=0,
        no_optimizer_updates=True,no_runtime_hardware_model_or_permission_change=True,
        no_closed_loop_success_or_full_safety_proof=True,goal_achieved=False,deployment_approved=False)
    write('MEMORY_GEOMETRY_CORRECTION.json',result)
    print(json.dumps({key:value for key,value in result.items() if key!='inputs'},ensure_ascii=False),flush=True)


if __name__ == '__main__':
    try:main()
    except Exception as error:
        write('MEMORY_CORRECTION_FAILURE.json',dict(error=repr(error),automatically_restarted=False))
        raise
