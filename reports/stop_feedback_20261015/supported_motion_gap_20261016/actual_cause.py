"""Decode the genuine106s stop from recorded sensor and physical evidence."""
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT=Path(__file__).resolve().parents[3]
OUT=Path(__file__).resolve().parent
source=ROOT/'reports/stop_feedback_20261015/w11_retained2000_w11_20261016'
report=json.loads((source/'RESULTS.json').read_text())
row=next(r for r in report['rows'] if r['case_id']=='w11_training_stall_t106')
path=source/'w11_training_stall_t106.npz'
if not report['completed'] or hashlib.sha256(path.read_bytes()).hexdigest()!=row['trace_sha256']:
    raise ValueError('A fixed complete actual failure is required')
with np.load(path,allow_pickle=False) as data:
    indices=np.flatnonzero(data['model_controlled'])
    first=int(indices[0]);obs=data['observations'][indices];labels=data['labels'][indices]
    groups=obs[:,303:399].reshape(-1,12,8).argmax(-1)
    xy=np.linalg.norm(labels[:,1:3]-labels[:,4:6],axis=-1)
    yaw=np.arctan2(np.sin(labels[:,3]-labels[:,6]),np.cos(labels[:,3]-labels[:,6]))
    masks=data['masks'][indices].astype(bool)
    result=dict(completed=True,trace=str(path.relative_to(ROOT)),trace_sha256=row['trace_sha256'],
        scope='The model-controlled suffix of the genuine106s failure only.',
        model_controlled_frames=len(indices),all_suffix_choices_stop=bool((data['choices'][indices]==0).all()),
        suffix_nonzero_executed_frames=int((abs(data['commands'][indices]).max(-1)>1e-8).sum()),
        all_suffix_legal_choices_exactly_stop_and_action9=bool(np.array_equal(masks,np.tile(np.arange(12)==0,(len(indices),1))|np.tile(np.arange(12)==9,(len(indices),1)))),
        legal_motion_action9_available_frames=int(masks[:,9].sum()),
        other_legal_motion_action5_available_frames=int(masks[:,5].sum()),
        legal_choice_set_histogram={','.join(map(str,key)):count for key,count in
            Counter(tuple(np.flatnonzero(mask)) for mask in masks).items()},
        remaining_goal_distance_m_at_release=float(labels[0,7]),
        position_error_at_release_m=float(xy[0]),position_error_p95_m=float(np.percentile(xy,95)),
        yaw_error_at_release_degrees=float(abs(yaw[0])*180/np.pi),
        scan_and_prediction_valid_all_frames=bool(((obs[:,438]>.5)&(obs[:,435]>.5)).all()),
        any_nonzero_forecast_contact_risk=bool((obs[:,419:431]>0).any()),
        first_frame_guard_group_by_action=groups[0].tolist(),
        rejected_motion_actions_are_laser_geometry_group_all_frames=bool((groups[:,[1,2,3,4,5,6,7,8,10,11]]==4).all()),
        minimum_observed_lidar_m=float((obs[:,144:216].min(-1)*12.).min()),
        action9_command_m_s_rad_s=[.01,-.5],first_sensor_time_s=float(data['labels'][first,0]),
        new_physical_steps=0,new_ppo_steps=0,
        action9_useful_escape_proven=False,goal_achieved=False,deployment_approved=False)
(OUT/'STOP_CAUSES.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
