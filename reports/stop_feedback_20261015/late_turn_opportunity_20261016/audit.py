from pathlib import Path
import json
import numpy as np
import torch
from experiments.guard_reason_release_actor import GuardReasonReleaseActor
from experiments.long_risk_actor import digest
from neural_navigation.experiment_budget import ExperimentBudget
from neural_navigation.masked_actions import ACTIONS
root=Path.cwd();out=Path(__file__).resolve().parent
torch.set_num_threads(1);budget=ExperimentBudget(.02,memory_mib=1024,directory=out)
model=root/'models/stop_feedback_20261015/normal_retention_refit2_s231016/tail500'
trace=root/'reports/stop_feedback_20261015/context_broad_diagnostic_20261016/physical/hard3_w11_training_stall_t106.npz'
meta=json.loads((model/'RESULTS.json').read_text());assert digest(model/'checkpoint.pt')==meta['checkpoint_sha256']
assert digest(trace)=='9d307c7e9a83a3c1f75c4a3158eb67c97cc9f98222e174e101ebe128ab127767'
actor=GuardReasonReleaseActor(torch.jit.load(str(model/'parent_actor.pt'),map_location='cpu')).eval()
actor.load_state_dict(torch.load(model/'checkpoint.pt',map_location='cpu',weights_only=False)['actor'])
with np.load(trace,allow_pickle=False)as z:
 selected=z['model_controlled'];obs=z['observations'][selected].copy();masks=z['masks'][selected].copy();times=z['labels'][selected,0].copy()
with torch.no_grad():actions=actor(torch.from_numpy(obs)).numpy()
distance=np.abs(actions[:,None]-ACTIONS[None]).max(2);choices=distance.argmin(1);assert(distance.min(1)<1e-6).all();assert masks[np.arange(len(choices)),choices].all()
rows=[dict(time_s=float(times[i]),choice=int(choices[i]),legal=np.flatnonzero(masks[i]).tolist())for i in np.flatnonzero(choices)]
counts={str(int(k)):int(v)for k,v in zip(*np.unique(choices,return_counts=True))}
budget.tick();r=dict(completed=True,inputs={str(p.relative_to(root)):digest(p)for p in[Path(__file__),model/'checkpoint.pt',trace]},frames=len(obs),choices=counts,motion_choices=rows,no_counterfactual_executed=True,no_training_labels_added=True,new_physical_episodes=0,new_optimizer_updates=0,resource=budget.statistics())
(out/'RESULTS.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r),flush=True)
