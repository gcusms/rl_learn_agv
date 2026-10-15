"""One recorded PPO update, paired in memory; no new simulation or model release."""
from pathlib import Path
import copy,hashlib,importlib.util,json,os,sys
import numpy as np,torch
from stable_baselines3.common.logger import configure as configure_logger
from experiments.stop_recovery_policy import StopRecoveryPPO
from experiments.train_prefix_recovery import frozen_base_digest
from experiments.demonstration_retention import configure
from neural_navigation.experiment_budget import ExperimentBudget

ROOT=Path(__file__).resolve().parents[3];OUT=Path(__file__).resolve().parent
os.chdir(ROOT);torch.set_num_threads(1)
budget=ExperimentBudget(.05,memory_mib=2500,directory=OUT)
run=Path('models/stop_feedback_20261015/proactive_augmented_ppo_8000_s231015')
paths=[run/'policy_step_8000.zip',run/'rollouts/rollout_008000.npz',run/'demonstrations.npz']
source_hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
with np.load(paths[1]) as d:rollout={k:d[k].copy() for k in d.files}
with np.load(paths[2]) as d:examples={k:d[k].copy() for k in d.files}
metadata=json.loads(str(rollout['metadata_json']))
base=StopRecoveryPPO.load(paths[0],device='cpu');budget.tick()
assert base.num_timesteps==8000 and base._n_updates==metadata['optimizer_epochs_before_update']
base_digest=frozen_base_digest(base.policy)
x=torch.from_numpy(rollout['observations'][:,0]);masks=rollout['action_masks'][:,0].astype(bool)
with torch.no_grad():
 log=base.policy.get_distribution(x,masks).log_prob(torch.from_numpy(rollout['sampled_actions'][:,0,0]).long()).numpy()
likelihood_error=float(abs(log-rollout['old_log_prob'][:,0]).max());assert likelihood_error<1e-5

def score(model):
 probabilities=[]
 with torch.no_grad():
  for start in range(0,len(examples['choices']),125):
   p=model.policy.get_distribution(torch.from_numpy(examples['observations'][start:start+125]),examples['masks'][start:start+125]).distribution.probs.numpy()
   probabilities.append(p);budget.tick()
 p=np.concatenate(probabilities);choices=examples['choices'];target=p[np.arange(len(choices)),choices];stop=choices==0
 return dict(examples=len(choices),target_mismatches=int((p.argmax(-1)!=choices).sum()),
  reasonable_wait_mismatches=int(((p.argmax(-1)!=choices)&stop).sum()),
  balanced_target_nll=float(-.5*(np.log(np.maximum(target[stop],1e-30)).mean()+np.log(np.maximum(target[~stop],1e-30)).mean())),
  mean_wait_target_probability=float(target[stop].mean()),mean_motion_target_probability=float(target[~stop].mean()))
legacy_path=run/'source_snapshot/training/experiments/stop_recovery_policy.py'
legacy_spec=importlib.util.spec_from_file_location('retention_legacy_recovery',legacy_path)
legacy=importlib.util.module_from_spec(legacy_spec);sys.modules[legacy_spec.name]=legacy
legacy_spec.loader.exec_module(legacy)
def identical(left,right):
 if torch.is_tensor(left):
  assert left.dtype==right.dtype and left.shape==right.shape and torch.equal(left,right)
 elif isinstance(left,dict):
  assert left.keys()==right.keys()
  for key in left:identical(left[key],right[key])
 elif isinstance(left,(list,tuple)):
  assert len(left)==len(right)
  for a,b in zip(left,right):identical(a,b)
 else:assert left==right
initial_score=score(base);rows=[];unregularized_state=None
for label,coefficient in [('original_ppo_update',0.),('retained_ppo_update',.05),('legacy_ppo_update',0.)]:
 model=copy.deepcopy(base)
 if coefficient:configure(model,examples['observations'],examples['masks'],examples['choices'],coefficient=coefficient,batch_size=128,seed=231016)
 model.set_logger(configure_logger(folder=None,format_strings=[]))
 buffer=model.rollout_buffer;buffer.reset()
 mapping=dict(observations='observations',actions='sampled_actions',action_masks='action_masks',log_probs='old_log_prob',values='old_values',rewards='ppo_rewards',returns='returns',advantages='advantages',episode_starts='episode_starts')
 for name,key in mapping.items():setattr(buffer,name,rollout[key].copy())
 buffer.full=True;buffer.pos=buffer.buffer_size;buffer.generator_ready=False
 np.random.seed(5416);torch.manual_seed(5416)
 model._current_progress_remaining=0.
 if label=='legacy_ppo_update':legacy.StopRecoveryPPO.train(model)
 else:model.train()
 budget.tick()
 if label=='original_ppo_update':
  unregularized_state=copy.deepcopy((model.policy.state_dict(),model.policy.optimizer.state_dict()))
 elif label=='legacy_ppo_update':
  identical(unregularized_state,(model.policy.state_dict(),model.policy.optimizer.state_dict()))
 assert frozen_base_digest(model.policy)==base_digest and model.num_timesteps==8000
 rows.append(dict(variant=label,retention_coefficient=coefficient,optimizer_epochs=model._n_updates-base._n_updates,score=score(model),training_metrics={k:float(v) for k,v in model.logger.name_to_value.items()}))
 del model
assert source_hashes=={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
result=dict(completed=True,source_files_unchanged=True,source_hashes=source_hashes,actual_recorded_samples=250,new_physical_steps=0,source_checkpoint_optimized_through=7750,source_checkpoint_collected_steps=8000,old_likelihood_max_error=likelihood_error,initial_score=initial_score,paired_updates=rows,same_initial_network_optimizer_rollout_and_minibatch_seed=True,disabled_retention_matches_legacy_network_and_optimizer_exactly=True,legacy_train_source=str(legacy_path),legacy_train_source_sha256=hashlib.sha256(legacy_path.read_bytes()).hexdigest(),exported_or_deployed=False,resource=budget.statistics(),limitations=['This is a paired offline update on one recorded rollout.','The arbitrary shared minibatch permutation is not claimed to reproduce the actual historical final8k update exactly.','Lower demonstration loss does not prove a physical safe recovery or unseen-scene generalization.'],goal_achieved=False,deployment_approved=False)
(OUT/'OPTIMIZER_CONTRAST.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(dict(completed=True,initial=initial_score,paired_updates=[dict(variant=r['variant'],score=r['score'],optimizer_epochs=r['optimizer_epochs']) for r in rows],resource=result['resource']),indent=2))
