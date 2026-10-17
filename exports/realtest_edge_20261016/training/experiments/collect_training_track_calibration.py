"""Capture frozen-policy sensor tracks on training tasks, never eval tasks."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from sb3_contrib import MaskablePPO
from experiments.masked_navigation import MaskedNavigationCurriculum
from experiments.fixed_training_replay import reset_training_episode
from experiments.calibrate_track_residuals import audit


def select_tasks(records,count):
    tasks=[]
    for record in records:
        if 'worker' not in record or 'training_case_json' not in record:continue
        case=json.loads(record['training_case_json'])
        if case.get('stage')!='robust' or case.get('focused_replay_origin'):continue
        if case['interventions'].get('obstacle') not in ('crossing','sudden_block'):continue
        tasks.append((record['step'],case))
        if len(tasks)==count:break
    if len(tasks)!=count:raise ValueError('Not enough chronological randomized training tasks')
    return tasks


def main():
    p=argparse.ArgumentParser();p.add_argument('--policy',type=Path,required=True)
    p.add_argument('--training-log',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--count',type=int,default=6)
    a=p.parse_args()
    if a.output.exists() or not 1<=a.count<=12:p.error('Fresh output and count 1..12 required')
    tasks=select_tasks([json.loads(x) for x in a.training_log.read_text().splitlines()],a.count)
    a.output.mkdir(parents=True);torch.set_num_threads(1)
    digest=hashlib.sha256(a.policy.read_bytes()).hexdigest();model=MaskablePPO.load(a.policy,device='cpu')
    config=dict(policy=str(a.policy.resolve()),policy_sha256=digest,
        training_log=str(a.training_log.resolve()),training_log_sha256=hashlib.sha256(a.training_log.read_bytes()).hexdigest(),
        selected_steps=[s for s,_ in tasks],selection='first chronological non-focused randomized dynamic training tasks, regardless of outcome',
        training=False,actor_inputs='native sensor core',truth_used_for_tracking=False)
    (a.output/'config.json').write_text(json.dumps(config,indent=2));results=[]
    status=dict(status='running',completed_cases=0,started_unix=time.time())
    def save():(a.output/'status.json').write_text(json.dumps(status,indent=2))
    save()
    try:
        for step,case in tasks:
            env=MaskedNavigationCurriculum(a.policy.parent,stage='robust',motion_profiles=True,footprint_clearance_cost=1.)
            rows=[]
            try:
                obs,_=reset_training_episode(env,case)
                for _ in range(env.max_steps):
                    choice=int(model.predict(obs,action_masks=env.action_masks(),deterministic=True)[0])
                    predictor=env.core.laser_fusion.predictor
                    tracks=[{k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in t.items()} for t in predictor.tracks]
                    rows.append(dict(timestamp=env.active.elapsed,estimate=env.core.estimate.tolist(),
                        policy_observation=obs.tolist(),sensor_diagnostic=dict(tracks=tracks)))
                    obs,_,done,truncated,info=env.step(choice)
                    if done or truncated:break
                result=dict(step=step,safe_success=bool(info['verified_mission_success']),
                    collisions=int(info['collision_events']),timeout=bool(truncated),frames=len(rows))
                trace=a.output/f'training_{step}.json'
                trace.write_text(json.dumps(dict(kind='training_sensor_calibration',case=case,samples=rows,result=result),allow_nan=False))
                calibration=audit(trace);calibration['recorded_outcome']=result
                results.append(calibration)
                status['completed_cases']=len(results);save()
                print(json.dumps(dict(step=step,result=result,calibration=calibration['all_pairs'])),flush=True)
            finally:env.close()
        if hashlib.sha256(a.policy.read_bytes()).hexdigest()!=digest:raise ValueError('Frozen actor changed')
        (a.output/'CALIBRATION.json').write_text(json.dumps(dict(results=results,config=config,
            limitation='Reused training tasks, not independent navigation evaluation. Sensor centroid consistency is not collision coverage. No model updated.'),indent=2))
        status.update(status='complete',finished_unix=time.time());save()
    except Exception as error:
        status.update(status='failed',error=str(error));save();raise


if __name__=='__main__':main()
