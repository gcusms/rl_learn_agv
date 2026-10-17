"""Check retained-point inclusion on actual complete sensor histories.

No-cap inclusion prevents adding an obstruction on the same replayed state; it
does not certify that extra clearing is safe or preserve closed-loop choices.
"""
import copy
from datetime import datetime
import json
from pathlib import Path
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import numpy as np

from replay import ROOT,HERE,BASE,EARLY,digest
from hybrid_memory import HybridFreeEvidenceMemory
from neural_navigation.decision_feedback import FEEDBACK_REASONS
from neural_navigation.safety_memory import NearFieldMemory
from neural_navigation.sensors import LidarSpec
from neural_navigation.experiment_budget import ExperimentBudget


def main():
    output=HERE/'SUBSET_REPLAY.json';assert not output.exists()
    source_path=HERE/'REPLAY.json';source=json.loads(source_path.read_text())
    assert source['completed'] and source['exact_source_accepted_and_laser_rejected_turn_geometry_reproduced']
    hardware_path=ROOT/'models/stop_feedback_20261015/guard_aligned_migration_s231017/portable/hardware.json'
    hardware=json.loads(hardware_path.read_text());config=hardware['near_field_memory'];lidar=LidarSpec(hardware['lidar'])
    paths=[Path(__file__),HERE/'replay.py',HERE/'hybrid_memory.py',source_path,hardware_path,
        ROOT/'training/neural_navigation/safety_memory.py']
    budget=ExperimentBudget(.1,550,HERE);rows=[]
    for row in source['rows']:
        path=ROOT/row['source_trace'];assert digest(path)==source['inputs'][str(path.relative_to(ROOT))];paths.append(path)
        original,hybrid=NearFieldMemory(config),HybridFreeEvidenceMemory(config)
        peak=extra_cleared=point_comparisons=commits=0
        with np.load(path,allow_pickle=False) as z:
            for i,stamp in enumerate(z['labels'][:,0]):
                core=SimpleNamespace(elapsed=float(stamp),odom=z['odometry'][i],scan=z['scans'][i],lidar=lidar)
                a,b=copy.deepcopy(original),copy.deepcopy(hybrid);a.update(core);b.update(core)
                assert len(a.entries)<config['max_points'],'Inclusion proof excludes capacity eviction'
                assert set(b.entries)<=set(a.entries)
                for key,(point,measured_at) in b.entries.items():
                    np.testing.assert_array_equal(point,a.entries[key][0]);assert measured_at==a.entries[key][1]
                    point_comparisons+=1
                extra_cleared+=len(a.entries)-len(b.entries);peak=max(peak,len(a.entries))
                reason=(FEEDBACK_REASONS[int(z['observations'][i+1,399:419].argmax())] if i+1<len(z['labels']) else None)
                if reason=='odometry_jump':original.reset();hybrid.reset()
                elif reason is not None and reason not in EARLY:original,hybrid=a,b;commits+=1
                if i%20==0:budget.tick()
        rows.append(dict(case_id=row['case_id'],source_memory=row['source_memory'],frames=len(z['labels']) if False else row['recorded_frames'],
            maximum_original_points=peak,unchanged_retained_point_and_measurement_time_checks=point_comparisons,
            aggregate_fewer_retained_point_counts=extra_cleared,actual_execution_commits=commits,
            hybrid_subset_of_original_at_every_probe=True))
    budget.tick()
    result=dict(completed=True,at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),rows=rows,
        checked_frames=sum(r['frames'] for r in rows),same_recorded_scans_odometry_and_commit_schedule=True,
        capacity_never_active=True,all_retained_points_and_uncertainty_timestamps_match_original=True,
        does_not_certify_extra_clearing_or_closed_loop_success=True,original4cm_margin_unchanged=True,
        new_physical_episodes=0,new_ppo_steps=0,new_optimizer_updates=0,resource=budget.statistics(),
        inputs={str(p.relative_to(ROOT)):digest(p) for p in paths},goal_achieved=False,deployment_approved=False)
    output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(frames=result['checked_frames'],rows=rows,resource=result['resource']),ensure_ascii=False),flush=True)


if __name__=='__main__':main()
