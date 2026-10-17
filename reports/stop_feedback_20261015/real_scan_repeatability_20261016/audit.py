"""Measure repeated recorded scans; do not label command-zero as true standstill.

No independent pose reference exists in this record. Low reported motion is
only a selection rule, so statistics also contain scene motion and input error.
They are neither a fitted lidar-noise model nor approval of a memory threshold.
"""
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

from neural_navigation.experiment_budget import ExperimentBudget


ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
SOURCE = ROOT / 'training/assets/real_recording_20261014/record/decisions.jsonl'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def quantiles(values):
    a = np.asarray(values, float)
    return None if not a.size else dict(count=int(a.size),
        p50=float(np.percentile(a, 50)),p90=float(np.percentile(a,90)),
        p95=float(np.percentile(a,95)),p99=float(np.percentile(a,99)),maximum=float(a.max()))


def main():
    if (OUT/'PLAN.json').exists(): raise FileExistsError('Fresh read-only audit required')
    inputs={str(p.relative_to(ROOT)):digest(p) for p in [SOURCE,Path(__file__)]}
    plan=dict(inputs=inputs,low_reported_motion=dict(linear_m_s=.005,angular_rad_s=.02,
        maximum_pair_translation_m=.002,maximum_pair_yaw_rad=.004,maximum_scan_interval_s=.4),
        range_analysis_m=[.1,10.],no_independent_reference_pose=True,
        no_model_or_memory_threshold_change=True,new_physical_steps=0,new_ppo_steps=0)
    (OUT/'PLAN.json').write_text(json.dumps(plan,indent=2)+'\n')
    budget=ExperimentBudget(.15,memory_mib=512,directory=OUT)
    records=[]
    with SOURCE.open() as f:
        for line in f: records.append(json.loads(line)['data'])
    differences,near_differences,triplets,invalid,gaps=[],[],[],[],[]
    selected=[]
    for i,r in enumerate(records):
        scan=np.asarray(r['inputs']['scan'],float)
        valid=np.isfinite(scan)&(scan>=.1)&(scan<10.)
        invalid.append(1.-float(valid.mean()))
        if not i: continue
        previous=records[i-1]
        stamp=int(r['scan_stamp_ns']);oldstamp=int(previous['scan_stamp_ns'])
        dt=(stamp-oldstamp)*1e-9
        if (r['mission_id']!=previous['mission_id'] or not 0.<dt<=.4):continue
        odom=np.asarray(r['inputs']['odom'],float);oldodom=np.asarray(previous['inputs']['odom'],float)
        velocity=np.asarray(r['inputs']['velocity'],float);oldvelocity=np.asarray(previous['inputs']['velocity'],float)
        dyaw=abs(float(np.arctan2(np.sin(odom[2]-oldodom[2]),np.cos(odom[2]-oldodom[2]))))
        if (max(abs(velocity[0]),abs(oldvelocity[0]))>.005
            or max(abs(velocity[-1]),abs(oldvelocity[-1]))>.02
            or np.linalg.norm(odom[:2]-oldodom[:2])>.002 or dyaw>.004):continue
        before=np.asarray(previous['inputs']['scan'],float)
        if before.shape!=scan.shape:continue
        common=valid&np.isfinite(before)&(before>=.1)&(before<10.)
        delta=np.abs(scan-before)
        differences.extend(delta[common])
        near_differences.extend(delta[common&(scan<1.5)&(before<1.5)])
        selected.append(i);gaps.append(dt)
        # Same-ray neighboring finite scans: retain actual angular gaps as unknown.
        a=np.where(valid,scan,np.nan)
        oldvalid=np.isfinite(before)&(before>=.1)&(before<10.)
        b=np.where(oldvalid,before,np.nan)
        if len(a)>=3:
            now=np.stack([a[:-2],a[1:-1],a[2:]])
            was=np.stack([b[:-2],b[1:-1],b[2:]])
            complete=np.isfinite(now).all(0)&np.isfinite(was).all(0)
            triplets.extend(np.abs(now[:,complete].min(0)-was[:,complete].min(0)))
        budget.tick()
    result=dict(completed=True,at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
        records=len(records),low_reported_motion_adjacent_pairs=len(selected),
        selected_second_record_indices=selected,scan_interval_s=quantiles(gaps),
        absolute_range_difference_m=quantiles(differences),
        near_1_5m_absolute_range_difference_m=quantiles(near_differences),
        three_neighbor_minimum_difference_m=quantiles(triplets),
        invalid_fraction_using_analysis_range_0_1_to_10_m=quantiles(invalid),
        no_estimated_sigma_or_free_memory_threshold_admitted=True,
        new_physical_steps=0,new_optimizer_updates=0,new_ppo_steps=0,
        resource=budget.statistics(),deployment_approved=False,goal_achieved=False,
        limitations=['Low reported motion and almost unchanged supplied odometry do not prove physical standstill.',
            'Difference contains lidar noise, scene motion, angular discretization and motion/feedback errors.',
            'No independent reference; no real-hardware4cm safety or Gaussian independence claim.',
            'Range10m is an analysis threshold, not evidence that every Inf is an invalid return.',
            'Repeated scans and routes are development data, not held-out hardware validation.'])
    for name,expected in inputs.items():assert digest(ROOT/name)==expected,name
    (OUT/'RESULTS.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='selected_second_record_indices'}),flush=True)


if __name__=='__main__':
    os.chdir(ROOT)
    try:main()
    except Exception as error:
        (OUT/'FAILURE.json').write_text(json.dumps(dict(error=repr(error),automatically_restarted=False))+'\n')
        raise
