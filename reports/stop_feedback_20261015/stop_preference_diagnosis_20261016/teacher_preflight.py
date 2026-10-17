from pathlib import Path
import json
import numpy as np
import torch
from neural_navigation.masked_actions import ACTIONS
from neural_navigation.experiment_budget import ExperimentBudget
from experiments.long_risk_actor import digest
from experiments.refit_phase_tail_pair import write
ROOT=Path.cwd();OUT=ROOT/"reports/stop_feedback_20261015/stop_preference_diagnosis_20261016"
torch.set_num_threads(1);budget=ExperimentBudget(.1,memory_mib=1400,directory=OUT)
tasks=json.loads((ROOT/"reports/stop_feedback_20261015/context_feedback_longfit_protocol_20261016/tasks.json").read_text())
models={"frozen_parent439":ROOT/"models/stop_feedback_20261015/full_mission_retention_pair_s231016/retained500/parent_actor.pt","reference439":ROOT/"models/stop_feedback_20261015/recovery_followup_bc400_s231016/bc_portable/actor.pt"}
actors={k:torch.jit.load(str(p),map_location="cpu").eval() for k,p in models.items()};report=dict(rows=[],model_sha256={k:digest(p) for k,p in models.items()},new_physics_steps=0,new_ppo_steps=0)
with torch.no_grad():
 for row in tasks["reference"]["normal18"]:
  with np.load(ROOT/row["trace_path"]) as d:
   x=torch.from_numpy(d["observations"]);target=ACTIONS[d["choices"]]
   report["rows"].append(dict(case_id=row["case_id"],frames=len(x),mismatches={k:int((actor(x).numpy()!=target).any(1).sum()) for k,actor in actors.items()}));budget.tick()
budget.tick();report.update(completed=True,resource=budget.statistics());write(OUT/"TEACHER_PREFLIGHT.json",report);print(json.dumps(report))
