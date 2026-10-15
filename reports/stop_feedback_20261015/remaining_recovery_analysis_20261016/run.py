"""Audit remaining waiting/collision cases using completed physical traces only."""
from pathlib import Path
import hashlib,json,os
import numpy as np
import torch
from experiments.audit_decision_waits import measure,segments,intervals
from experiments.stop_recovery_policy import StopRecoveryPPO,masks_tensor
from experiments.train_prefix_recovery import frozen_base_digest
from neural_navigation.decision_feedback import FEEDBACK_REASONS,GROUPS,REASON_SLICE,STAT_SLICE
from neural_navigation.experiment_budget import ExperimentBudget

ROOT=Path(__file__).resolve().parents[3]
OUT=Path(__file__).resolve().parent
os.chdir(ROOT);torch.set_num_threads(1)
plan=json.loads((OUT/'PLAN.json').read_text())
budget=ExperimentBudget(.05,memory_mib=2500,directory=OUT)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
for path,expected in plan['inputs'].items():assert sha(path)==expected,path
models={name:StopRecoveryPPO.load(path,device='cpu') for name,path in plan['models'].items()}
budget.tick()
assert len({frozen_base_digest(x.policy) for x in models.values()})==1
loaded=[];rows=[]
for specification in plan['traces']:
 folder=Path(specification['folder']);report=json.loads((folder/'RESULTS.json').read_text())
 assert report['completed']
 physical=next(r for r in report['rows'] if r['case_id']==specification['case_id'])
 paths=[folder/(physical['case_id']+'_'+physical['variant']+'.npz'),folder/(physical['case_id']+'.npz')]
 path=next(p for p in paths if p.exists() and sha(p)==physical['trace_sha256'])
 with np.load(path,allow_pickle=False) as data:trace={k:data[k].copy() for k in data.files}
 original=trace['observations'];assert original.shape[1]==439
 x=np.concatenate([original[:,:300],original[:,303:]],axis=-1).astype(np.float32)
 masks=trace['masks'].astype(bool);choices=trace['choices'];times=trace['labels'][:,0]
 assert np.isfinite(x).all() and np.all(np.diff(times)>0)
 np.testing.assert_array_equal(masks_tensor(torch.from_numpy(x)).numpy(),masks)
 assert masks[np.arange(len(x)),choices].all()
 predictions={};support={};probabilities={}
 with torch.no_grad():
  for name,model in models.items():
   pieces=[];active=[]
   for start in range(0,len(x),125):
    tensor=torch.from_numpy(x[start:start+125])
    p=model.policy.get_distribution(tensor,masks[start:start+125]).distribution.probs.numpy()
    assert np.isfinite(p).all() and (p[~masks[start:start+125]]==0).all()
    pieces.append(p);active.append(model.policy.recovery_support(tensor).numpy());budget.tick()
   probabilities[name]=np.concatenate(pieces);predictions[name]=probabilities[name].argmax(-1);support[name]=np.concatenate(active)
 for name in support:np.testing.assert_array_equal(support[name],support['w11_bc800'])
 # Fixed initialization commands in prefix traces are not claims of a learned choice.
 controlled=trace.get('model_controlled',np.ones(len(x),bool)).astype(bool)
 actual_model=specification['model'];mismatch=int(np.count_nonzero((predictions[actual_model]!=choices)&controlled))
 assert mismatch==0,(str(path),mismatch)
 stats=x[:,300+STAT_SLICE.start:300+STAT_SLICE.stop]
 available=masks[:,1:].any(-1);idle=np.max(np.abs(trace['commands']),axis=-1)<1e-8
 pending=trace['labels'][:,7]>=.2;active=support['w11_bc800']
 legal_wait=idle&available&pending&controlled
 dt=intervals(times,physical['seconds'])
 reasons=np.array(FEEDBACK_REASONS)[x[:,300+REASON_SLICE.start:300+REASON_SLICE.stop].argmax(-1)]
 groups=x[:,300:396].reshape(-1,12,8).argmax(-1)
 def group_counts(selection):
  return {str(reason):float(dt[selection&(reasons==reason)].sum()) for reason in np.unique(reasons[selection])}
 def segment_row(start,end):
  return dict(start_s=float(times[start]),duration_s=float(dt[start:end].sum()),end_s=float(times[start]+dt[start:end].sum()),legal_actions_at_start=np.flatnonzero(masks[start]).tolist(),valid_prediction_at_start=bool(stats[start,4]>.5),scan_valid_at_start=bool(stats[start,7]>.5),learnable_frames=int(active[start:end].sum()),first_observation_sha256=hashlib.sha256(x[start].tobytes()).hexdigest(),model_choices_at_start={name:int(predictions[name][start]) for name in models},previous_reason=str(reasons[start]),learned_motion_proposals_from_old_ppo=int(np.count_nonzero(predictions['ppo8000'][start:end])),physical_displacement_m=float(np.linalg.norm(trace['labels'][min(end,len(x)-1),1:3]-trace['labels'][start,1:3])))
 waits=[segment_row(a,b) for a,b in segments(legal_wait) if dt[a:b].sum()>=1.-1e-8]
 forbidden=~available&controlled
 rejected={GROUPS[g]:int(np.count_nonzero((groups==g)&~masks&forbidden[:,None])) for g in range(8)}
 for name,pred in predictions.items():assert masks[np.arange(len(x)),pred].all()
 row=dict(case_id=physical['case_id'],actual_model=actual_model,physical=physical,source_trace=str(path),trace_sha256=sha(path),model_controlled_frames=int(controlled.sum()),actual_model_argmax_mismatches=mismatch,active_recovery_frames=int((active&controlled).sum()),active_recovery_legal_wait_s=float(dt[legal_wait&active].sum()),inactive_recovery_legal_wait_s=float(dt[legal_wait&~active].sum()),wait_previous_reasons_s=group_counts(legal_wait),only_stop_rejections_action_frames=rejected,wait_segments=waits,same_observation_cross_policy={name:dict(legal_wait_frames=int(legal_wait.sum()),motion_proposals_during_actual_legal_wait=int(np.count_nonzero((pred!=0)&legal_wait)),matches_actual_supported_actions=int(np.count_nonzero((pred==choices)&active&controlled)),supported_frames=int((active&controlled).sum())) for name,pred in predictions.items()},metrics=measure(trace,physical['seconds'],.2))
 row['controlled_metrics']=measure({k:v[controlled] for k,v in trace.items()
     if k in ['labels','masks','choices','commands']},physical['seconds'],.2)
 row['fixed_initialization_excluded_from_controlled_metrics']=bool(not controlled.all())
 row['controlled_start_sensor_time_s']=float(times[controlled][0])
 rows.append(row);loaded.append((specification,physical,trace,x,predictions,support))
 budget.tick()
divergences=[]
for case in ['recovery_holdout_00','recovery_holdout_01']:
 a=next(z for z in loaded if z[0]['case_id']==case and z[0]['model']=='w11_bc800')
 b=next(z for z in loaded if z[0]['case_id']==case and z[0]['model']=='ppo8000')
 n=min(len(a[2]['choices']),len(b[2]['choices']))
 different=np.flatnonzero(a[2]['choices'][:n]!=b[2]['choices'][:n]);assert len(different)
 i=int(different[0]);parity={}
 for key in ['observations','masks','scans','labels','odometry','measured_velocity','human_positions','commands']:
  stop=i if key=='commands' else i+1
  parity[key]=bool(np.array_equal(a[2][key][:stop],b[2][key][:stop],equal_nan=True))
 assert all(parity.values()),(case,parity)
 p={}
 for name,model in models.items():
  with torch.no_grad():v=model.policy.get_distribution(torch.from_numpy(a[3][i:i+1]),a[2]['masks'][i:i+1]).distribution.probs.numpy()[0]
  p[name]=dict(choice=int(v.argmax()),probabilities=v.tolist())
 divergences.append(dict(case_id=case,first_different_frame=i,sensor_time_s=float(a[2]['labels'][i,0]),common_causal_prefix=parity,legal_actions=np.flatnonzero(a[2]['masks'][i]).tolist(),recovery_supported=bool(a[5]['w11_bc800'][i]),predictions=p))
result=dict(completed=True,previous_goal_turn_classification='progress',previous_goal_turn_evidence='Published now1459d4d and authenticated exact remote commit/tree plus10keyfiles;118checks passed.',current_goal_turn_classification='progress',models=plan['models'],rows=rows,first_hard_case_divergences=divergences,physical_runs_added=0,new_model_trained=False,resource=budget.statistics(),limitations=['Legal instantaneous motion does not establish a safe future recovery trajectory.','Cross-policy actions on a failed history have not been executed and are not demonstrations.','Successful PPO00/01 trajectories prove those initial tasks feasible with current actions, not that the late stalled state can be escaped.','Using any evaluated trajectory for subsequent training changes its status to reused development data.','No guard,mask,body,dynamics,reward,base actor,localizer or router is modified.'],goal_achieved=False,deployment_approved=False)
(OUT/'RESULTS.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(dict(completed=True,divergences=divergences,waiting=[dict(case_id=r['case_id'],model=r['actual_model'],legal_wait_s=r['controlled_metrics']['unfinished_task_executed_idle_with_motion_s'],learnable_wait_s=r['active_recovery_legal_wait_s'],unlearnable_wait_s=r['inactive_recovery_legal_wait_s'],only_stop_s=r['controlled_metrics']['only_stop_s'],same_state_ppo_motion_proposals=r['same_observation_cross_policy']['ppo8000']['motion_proposals_during_actual_legal_wait']) for r in rows],resource=result['resource']),indent=2))
