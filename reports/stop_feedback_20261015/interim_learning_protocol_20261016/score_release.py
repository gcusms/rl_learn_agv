"""Inspect a real fixed recovery input; no actions are executed."""
from pathlib import Path
import hashlib,json,os
import numpy as np
import torch
from experiments.stop_recovery_policy import StopRecoveryPPO
from neural_navigation.masked_actions import COMMANDS,masks_from_observation
from neural_navigation.decision_feedback import FEEDBACK_REASONS,REASON_SLICE,STAT_SLICE
from neural_navigation.experiment_budget import ExperimentBudget
root=Path(__file__).resolve().parents[3]
os.chdir(root)
output=Path('reports/stop_feedback_20261015/proactive_w11_release_scores_20261016')
output.mkdir(exist_ok=False)
torch.set_num_threads(1)
budget=ExperimentBudget(.05,memory_mib=2048,directory=output)
run=Path('models/stop_feedback_20261015/proactive_augmented_ppo_8000_s231015')
source=run/'warmups/episode_00002.npz'
with np.load(source,allow_pickle=False) as data:
 observation=data['observations'][-1].copy()
 metadata=json.loads(str(data['metadata_json']))
if metadata['prefix_id']!='w11_training_stall_t90' or abs(metadata['source_release_s']-90.)>1e-8:
 raise ValueError('The actual shared 90-second physical prefix is required')
mask=masks_from_observation(observation[None,:282])
np.save(output/'observation.npy',observation)
rows=[]
for name,file in [('own_bc','policy_bc.zip'),('pre_update_1000','policy_step_1000.zip')]:
 path=run/file
 model=StopRecoveryPPO.load(path,device='cpu')
 with torch.no_grad():
  tensor=torch.from_numpy(observation[None])
  probabilities=model.policy.get_distribution(tensor,mask).distribution.probs.numpy()[0]
  supported=bool(model.policy.recovery_support(tensor)[0])
 rows.append(dict(name=name,checkpoint=str(path),checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
  collected_timesteps=model.num_timesteps,optimizer_epochs_completed=model._n_updates,
  recovery_support=supported,greedy_choice=int(probabilities.argmax()),
  probabilities=probabilities.tolist(),pure_left_turn_probability=float(probabilities[4]),
  pure_right_turn_probability=float(probabilities[5]),wait_probability=float(probabilities[0])))
 budget.tick()
result=dict(completed=True,scope='Offline probability at an identical actual sensor/history input, not a physical maneuver or recovery result.',
 source_warmup=str(source),source_warmup_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
 observation_sha256=hashlib.sha256((output/'observation.npy').read_bytes()).hexdigest(),
 prefix_id=metadata['prefix_id'],release_s=90.,masks=mask[0].tolist(),commands=COMMANDS.tolist(),
 previous_executed_reason=FEEDBACK_REASONS[int(observation[300+REASON_SLICE.start:300+REASON_SLICE.stop].argmax())],
 previous_choice_wait_s=float(observation[300+STAT_SLICE.start+3])*5.,
 rows=rows,teacher_left_maneuver_feasibility_not_yet_confirmed=True,
 actions_executed=False,training_modified=False,resource=budget.statistics(),goal_achieved=False,deployment_approved=False)
(output/'RESULTS.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2),flush=True)
