"""Read-only learned-retention and issued-wait audit of completed traces."""
from pathlib import Path
import hashlib,json,os
import numpy as np,torch
from experiments.stop_recovery_policy import StopRecoveryPPO
from experiments.train_prefix_recovery import frozen_base_digest
from experiments.audit_decision_waits import measure
from neural_navigation.experiment_budget import ExperimentBudget
from neural_navigation.masked_actions import masks_from_observation
from neural_navigation.decision_feedback import FEEDBACK_REASONS,REASON_SLICE

ROOT=Path(__file__).resolve().parents[3]
OUT=Path(__file__).resolve().parent
os.chdir(ROOT);torch.set_num_threads(1)
budget=ExperimentBudget(.05,memory_mib=2500,directory=OUT)
plan=json.loads((OUT/'PLAN.json').read_text())
for name,sha in plan['inputs'].items():assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==sha
with np.load('models/stop_feedback_20261015/w11_recovery_bc800_s231015/demonstrations.npz') as data:
 obs=data['observations'].copy();masks=data['masks'].astype(bool);choices=data['choices'].copy()
assert len(choices)==2089 and masks[np.arange(len(choices)),choices].all()
np.testing.assert_array_equal(masks_from_observation(obs[:,:282]),masks)
with np.load('models/stop_feedback_20261015/proactive_augmented_bc_800_s231015/demonstrations.npz') as old:
 for name,actual in [('observations',obs),('masks',masks),('choices',choices)]:np.testing.assert_array_equal(old[name],actual[:2030])
paths={
 'parent_bc800':'models/stop_feedback_20261015/proactive_augmented_bc_800_s231015/policy_bc.zip',
 'ppo8000':'models/stop_feedback_20261015/proactive_augmented_ppo_8000_s231015/policy.zip',
 'w11_bc800':'models/stop_feedback_20261015/w11_recovery_bc800_s231015/policy_bc.zip'}
models={};retention={}
for name,path in paths.items():
 model=StopRecoveryPPO.load(path,device='cpu');models[name]=model;budget.tick()
 values=[]
 with torch.no_grad():
  for start in range(0,len(obs),125):
   p=model.policy.get_distribution(torch.from_numpy(obs[start:start+125]),masks[start:start+125]).distribution.probs.numpy()
   assert np.isfinite(p).all() and (p[~masks[start:start+125]]==0).all()
   values.append(p);budget.tick()
 probabilities=np.concatenate(values);predictions=probabilities.argmax(-1);target=probabilities[np.arange(len(obs)),choices]
 retained={}
 for group,selection in [('old2030',np.arange(len(obs))<2030),('new59',np.arange(len(obs))>=2030),('reasonable_stop',choices==0),('movement',choices!=0)]:
  wrong=selection & (predictions!=choices)
  retained[group]=dict(frames=int(selection.sum()),target_mismatches=int(wrong.sum()),wait_to_motion=int((wrong & (choices==0)).sum()),motion_to_wait=int((wrong & (choices!=0) & (predictions==0)).sum()),different_motion=int((wrong & (choices!=0) & (predictions!=0)).sum()),minimum_target_probability=float(target[selection].min()),mean_target_probability=float(target[selection].mean()))
 retention[name]=retained
assert len({frozen_base_digest(m.policy) for m in models.values()})==1

def trace(folder,case):
 report_path=Path('reports/stop_feedback_20261015')/folder/'RESULTS.json'
 captured_folder=OUT/'sources'/folder;captured_folder.mkdir(parents=True,exist_ok=True)
 captured_report=captured_folder/'RESULTS.json'
 if not captured_report.exists():captured_report.write_bytes(report_path.read_bytes())
 report_bytes=captured_report.read_bytes();report=json.loads(report_bytes)
 rows=[r for r in report['rows'] if r['case_id']==case];assert len(rows)==1
 row=rows[0];path=report_path.parent/(case+'_'+row['variant']+'.npz')
 captured_trace=captured_folder/path.name
 if not captured_trace.exists():captured_trace.write_bytes(path.read_bytes())
 assert hashlib.sha256(captured_trace.read_bytes()).hexdigest()==row['trace_sha256']
 with np.load(captured_trace) as d:arrays={k:d[k].copy() for k in d.files}
 return row,arrays,dict(report=str(report_path),captured_report=str(captured_report.relative_to(ROOT)),report_sha256=hashlib.sha256(report_bytes).hexdigest(),trace=str(path),captured_trace=str(captured_trace.relative_to(ROOT)),trace_sha256=row['trace_sha256'],report_completed=report['completed'])
parent_row,parent,parent_source=trace('proactive_augmented_bc800_gate_20261016','motion_profile_015')
ppo_row,ppo,ppo_source=trace('proactive_augmented_ppo8000_navigation_20261016','motion_profile_015')
length=min(len(parent['choices']),len(ppo['choices']))
first=int(np.flatnonzero(parent['choices'][:length]!=ppo['choices'][:length])[0])
common={}
for key in ['observations','masks','commands','scans','humans','odometry','measured_velocities','labels']:
 if key in parent and key in ppo:
  # The divergent frame has already chosen a different command, but its
  # observations and initial physical state remain directly comparable.
  stop=first if key=='commands' else first+1
  common[key]=bool(np.array_equal(parent[key][:stop],ppo[key][:stop]))
assert common['observations'] and common['masks'] and common['scans']
x=np.concatenate([parent['observations'][first,:300],parent['observations'][first,303:]]).astype(np.float32)[None]
mask=parent['masks'][first:first+1].astype(bool);distributions={}
with torch.no_grad():
 for name,model in models.items():
  p=model.policy.get_distribution(torch.from_numpy(x),mask).distribution.probs.numpy()[0]
  distributions[name]=dict(choice=int(p.argmax()),probabilities=p.tolist(),supported=bool(model.policy.recovery_support(torch.from_numpy(x))[0]));budget.tick()
reason=FEEDBACK_REASONS[int(x[0,300+REASON_SLICE.start:300+REASON_SLICE.stop].argmax())]
divergence=dict(case_id='motion_profile_015',first_different_frame=first,sensor_time_s=float(parent['labels'][first,0]),common_prefix_including_current_observation=common,parent_choice=int(parent['choices'][first]),ppo_choice=int(ppo['choices'][first]),previous_feedback_reason=reason,legal_actions=np.flatnonzero(mask[0]).tolist(),predictions_on_same_observation=distributions,parent_physical=parent_row,ppo_physical=ppo_row,source=[parent_source,ppo_source])
waits=[]
for folder,name in [('w11_recovery_bc800_navigation_20261016','w11_bc800'),('proactive_augmented_ppo8000_navigation_20261016','ppo8000'),('proactive_augmented_bc800_gate_20261016','parent_bc800')]:
 row,data,source=trace(folder,'recovery_holdout_00')
 waits.append(dict(variant=name,physical=row,metrics=measure(data,row['seconds'],.2),source=source));budget.tick()
result=dict(completed=True,retained_old_arrays_identical=True,frozen_base_identical=True,demonstration_retention=retention,first_regressed_physical_decision=divergence,hard00_actual_waiting=waits,additional_physical_runs=0,changes_to_runtime_or_model=False,resource=budget.statistics(),limitations=['Demonstration mismatch can be a useful different action; it is not itself unsafe.','The regressed task has an actual safe parent and colliding PPO under the same initial condition.','Frozen current-task observations in an audit are not new on-policy navigation.','Partial full-gate reports are explicitly pinned as bytes; these individual outcomes are complete but the18-task reports are not.'],goal_achieved=False,deployment_approved=False)
(OUT/'RESULTS.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(dict(completed=True,retention=retention,first_regression_frame=first,first_regression_sensor_time_s=divergence['sensor_time_s'],parent_choice=divergence['parent_choice'],ppo_choice=divergence['ppo_choice'],hard00=[dict(variant=r['variant'],safe=r['physical']['safe_success'],actual_idle_when_motion_legal_s=r['metrics']['unfinished_task_executed_idle_with_motion_s'],only_stop_s=r['metrics']['only_stop_s']) for r in waits],resource=result['resource']),indent=2))
