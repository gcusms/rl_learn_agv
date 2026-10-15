"""Diagnostic sensor controller: tests reachability, never scores learned success.

This deliberately simple controller is NOT a trained model or deployable
fallback. It selects among exactly the same sensor-admitted RL actions to
check whether the repaired environment offers a usable action sequence.
"""
import argparse
import copy
import json
import time
from pathlib import Path
import numpy as np
import torch
from prototypes.narrow_training_env import NarrowTrainingEnv
from prototypes.narrow_observation import COMMANDS
from prototypes.train_narrow_long import CASES


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--case',choices=[c['id'] for c in CASES],default='short_exit')
    p.add_argument('--max-steps',type=int,default=250)
    p.add_argument('--audit',action='store_true')
    p.add_argument('--cruise-speed',type=float,default=.05)
    p.add_argument('--center-corridor',action='store_true')
    args=p.parse_args()
    if not (.005<args.cruise_speed<=.2):p.error('Cruise speed must remain within physical forward limits')
    if args.output.exists():p.error('Fresh output required')
    args.output.parent.mkdir(parents=True,exist_ok=True);torch.set_num_threads(1)
    case=next(c for c in CASES if c['id']==args.case)
    env=NarrowTrainingEnv('training/runs/narrow_long_50000_v2_20261013/scenes/gap_150',
        'models/recovery_20261012/portable',seed=case['seed'],max_steps=args.max_steps)
    started=time.time();rows=[]
    try:
        obs,_=env.reset(options=dict(start=case['start'],goal=case['goal'],case_seed=case['seed']))
        for index in range(args.max_steps):
            mask=env.action_masks();target=obs[216:218]
            if np.linalg.norm(target)<.025:target=obs[218:220]
            angle=float(np.arctan2(target[1],target[0]));desired_w=float(np.clip(2*angle,-.2,.2))
            if args.center_corridor and obs[308]>.5 and obs[309]*2<.8:
                desired_w=float(np.clip(2*(obs[311]*.35)+5*(obs[310]*.25),-.2,.2))
            desired_v=args.cruise_speed if abs(angle)<.3 else 0.
            angular_weight=4. if args.center_corridor else 1.
            cost=np.abs(COMMANDS[:,0]-desired_v)/.2+angular_weight*np.abs(COMMANDS[:,1]-desired_w)/.5
            if args.center_corridor:cost[0]+=.3
            choice=int(np.argmin(np.where(mask,cost,np.inf)))
            audit=None
            if args.audit and index in [0,20,40,60,79]:
                from prototypes.diagnose_narrow_memory import point_summary
                core=env.runtime.core;guard=env.runtime.guard
                valid=np.isfinite(core.scan)&(core.scan>=core.lidar.range_min)&(core.scan<core.lidar.range_max)
                angles=core.lidar.angles[valid]+core.lidar.mount[3]
                fresh=core.scan[valid,None]*np.c_[np.cos(angles),np.sin(angles)]+core.lidar.mount[:2]
                remembered,uncertainty=copy.deepcopy(guard.memory).update(core)
                audit={}
                for name,points,margins in [('fresh',fresh,np.full(len(fresh),.04)),('remembered',remembered,.04+uncertainty),
                    ('combined',np.vstack([fresh,remembered]),np.r_[np.full(len(fresh),.04),.04+uncertainty])]:
                    audit[name]=point_summary(points,margins,guard.half)
                    audit[name]['forward_reason']=guard._trajectory_check(core,np.array([.05,0.]),points,margins,0.,True)
                audit['legal_choices']=np.flatnonzero(mask).tolist();audit['route']=core.predicted_path.tolist()
                audit['map_agreement']=float(core.localizer.map_agreement);audit['unmatched_returns']=int(core.unmatched_returns)
                audit['native_forward_reason']=copy.deepcopy(guard).check(core,[.05,0.],core.elapsed,core.elapsed)[1]
            obs,reward,done,truncated,info=env.step(choice)
            rows.append(dict(choice=choice,command=info['commanded_velocity'],pose=env.runtime.core.estimate.tolist(),audit=audit))
            if done or truncated:break
        result=dict(case=case,result=info,frames=len(rows),samples=rows,timeout=bool(truncated),
            diagnostic_controller=True,learned_success=False,truth_used_for_control=False,
            cruise_speed_m_s=args.cruise_speed,
            corridor_centering_diagnostic=args.center_corridor,
            started_unix=started,finished_unix=time.time())
        args.output.write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='samples'}),flush=True)
    finally:env.close()


if __name__=='__main__':main()
