from pathlib import Path
import json
import numpy as np
import torch
from experiments.guard_reason_release_actor import GuardReasonReleaseActor
from experiments.long_risk_actor import LongRiskActor,digest
from neural_navigation.masked_actions import ACTIONS
from neural_navigation.experiment_budget import ExperimentBudget
from experiments.refit_phase_tail_pair import write
ROOT=Path.cwd(); OUT=ROOT/"reports/stop_feedback_20261015/stop_preference_diagnosis_20261016"
torch.set_num_threads(1); budget=ExperimentBudget(.1,memory_mib=1400,directory=OUT)
model=ROOT/"models/stop_feedback_20261015/full_mission_retention_pair_s231016/retained500"
actor=GuardReasonReleaseActor(torch.jit.load(str(model/"parent_actor.pt"),map_location="cpu")).eval()
actor.load_state_dict(torch.load(model/"checkpoint.pt",map_location="cpu",weights_only=False)["actor"])
path=ROOT/"reports/stop_feedback_20261015/full_mission_retention_protocol_20261016/demonstrations.npz"
with np.load(path) as d:data={k:d[k].copy() for k in d.files}
x=torch.from_numpy(data["observations"])
with torch.no_grad():
 f,base=actor.features(x); masks=actor.masks(x); active=actor.support(x); old=LongRiskActor.support(actor,x[:,:481]); prior=(base[:,None]-actor.actions[None]).abs().amax(2).argmin(1)
 logit=actor.logits(x).masked_fill(~masks,-1e8)
 current=actor(x); budget.tick()
 near=x[:,144:216].amin(1)*12<.8; risk=x[:,419:431].amax(1)>0
 clear=masks[:,1:].all(1)&~risk&~old
 selected=(active&clear&near).numpy(); stop=data["choices"]==0; disagree=(prior.numpy()!=data["choices"])
 report=dict(source_count=len(x),original_source_count=4067,fit_correct=int((current==actor.actions[data["choices"]]).all(1).sum()),all_clear_near_only_context_count=int(selected.sum()),all_clear_near_only_context_label_stops=int((selected&stop).sum()),all_clear_near_only_context_source_disagreements=int((selected&disagree).sum()),all_clear_near_only_stop_parent_motion=[dict(index=int(i),source_id=int(data["source_ids"][i]),parent_action=int(prior[i]),scan_m=float(x[i,144:216].min()*12),old_idle=float(x[i,434]*5)) for i in np.flatnonzero(selected&stop&(prior.numpy()!=0))])
 def inspect(name,observation):
  y=torch.from_numpy(observation[None]);feat,parent=actor.features(y);legal=actor.masks(y);logits=actor.logits(y).masked_fill(~legal,-1e8);prob=logits.softmax(-1)[0]; baseline=int((parent[:,None]-actor.actions[None]).abs().amax(2).argmin(1)[0]);p=actor.head[0]; projection=p.previous(feat[:,:312]);reasons=p.current_reason(feat[:,312:]);ablated=(actor.head[1:](projection)+2*feat[:,76:88]).masked_fill(~legal,-1e8)
  row=dict(name=name,parent_action=baseline,head_action=int(logits.argmax(1)[0]),active=bool(actor.support(y)[0]),old_history_support=bool(LongRiskActor.support(actor,y[:,:481])[0]),nearest_scan_m=float(y[0,144:216].min()*12),current_risk=bool((y[0,419:431]>0).any()),legal_actions=np.flatnonzero(legal[0].numpy()).tolist(),probabilities=prob.tolist(),without_current_reason_action=int(ablated.argmax(1)[0]),current_reason_projection_norm=float(reasons.norm()),previous_reason=int(y[0,399:419].argmax()),idle_with_choices_s=float(y[0,434]*5))
  ds=(f-feat).norm(dim=1).numpy(); indices=np.argsort(ds)[:8];row["nearest_sources"]=[dict(index=int(i),distance=float(ds[i]),source_id=int(data["source_ids"][i]),target=int(data["choices"][i])) for i in indices]
  return row
 with np.load(ROOT/"reports/stop_feedback_20261015/full_mission_retention_validation_20261016/physical/normal_other16_independent_dynamic_000.npz") as d:
  report["dynamic_first_fork"]=inspect("dynamic_fork_8_6s",d["observations"][43]);report["dynamic_wait_middle"]=inspect("dynamic_wait_35s",d["observations"][175])
 with np.load(ROOT/"reports/stop_feedback_20261015/context_broad_diagnostic_20261016/physical/hard3_w11_training_stall_t106.npz") as d:
  report["late_first"]=inspect("actual106_first",d["observations"][530])
 report.update(completed=True,no_new_physics=True,new_ppo_steps=0,goal_achieved=False,deployment_approved=False,inputs={str(p.relative_to(ROOT)):digest(p) for p in [model/"checkpoint.pt",model/"parent_actor.pt",path,OUT/"audit.py"]},resource=budget.statistics())
 budget.tick();report["resource"]=budget.statistics();write(OUT/"RESULTS.json",report)
 print(json.dumps(report))
