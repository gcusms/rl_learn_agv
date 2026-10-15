"""Check actual rollout likelihoods and frozen-policy exploration at late waits."""
from pathlib import Path
import hashlib,json,os
import numpy as np
import torch
from experiments.stop_recovery_policy import StopRecoveryPPO
from neural_navigation.decision_feedback import STAT_SLICE
from neural_navigation.experiment_budget import ExperimentBudget

ROOT=Path(__file__).resolve().parents[3];OUT=Path(__file__).resolve().parent
os.chdir(ROOT);torch.set_num_threads(1)
plan=json.loads((OUT/'EXPLORATION_PLAN.json').read_text())
budget=ExperimentBudget(.05,memory_mib=2500,directory=OUT)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
for path,expected in plan['inputs'].items():assert sha(path)==expected,path
arrays={k:[] for k in ['observations','sampled_actions','action_masks','old_log_prob','advantages']}
for path in plan['rollouts']:
 with np.load(path,allow_pickle=False) as d:
  for k in arrays:arrays[k].append(d[k].reshape((-1,436) if k=='observations' else (-1,12) if k=='action_masks' else (-1,)).copy())
 budget.tick()
a={k:np.concatenate(v) for k,v in arrays.items()};x=a['observations'];masks=a['action_masks'].astype(bool);choices=a['sampled_actions'].astype(int)
assert len(x)==8000 and masks[np.arange(len(x)),choices].all()
assert np.array_equal(choices,a['sampled_actions'])
policy=StopRecoveryPPO.load(plan['models']['ppo8000'],device='cpu').policy
budget.tick()
with torch.no_grad():active=policy.recovery_support(torch.from_numpy(x)).numpy()
long_wait=x[:,300+STAT_SLICE.start+3]>=.4
pair=masks[:,0]&masks[:,9]&(masks.sum(-1)==2)
probability=np.exp(a['old_log_prob'].astype(float))
groups={}
for name,selection in [('all_supported',active),('supported_long_wait',active&long_wait),('only_stop_or9',active&pair),('long_wait_only_stop_or9',active&pair&long_wait)]:
 stop=selection&(choices==0);p=probability[stop]
 groups[name]=dict(samples=int(selection.sum()),movement_samples=int(np.count_nonzero(selection&(choices!=0))),stop_samples=int(stop.sum()),sampled_action_counts={str(k):int(np.count_nonzero(selection&(choices==k))) for k in np.unique(choices[selection])},stop_samples_with_sampled_probability_at_least_0999999=int(np.count_nonzero(p>=.999999)),stop_sampled_probability_quantiles=None if not len(p) else np.quantile(p,[0,.5,.95,1]).tolist(),positive_advantage_stop_samples=int(np.count_nonzero(stop&(a['advantages']>0))),negative_advantage_stop_samples=int(np.count_nonzero(stop&(a['advantages']<0))))
 budget.tick()
models={'ppo8000':policy,'parent_bc800':StopRecoveryPPO.load(plan['models']['parent_bc800'],device='cpu').policy,'w11_bc800':StopRecoveryPPO.load(plan['models']['w11_bc800'],device='cpu').policy}
budget.tick();states=[]
for spec in plan['states']:
 with np.load(spec['trace'],allow_pickle=False) as d:
  i=int(np.argmin(abs(d['labels'][:,0]-spec['time_s'])));assert abs(d['labels'][i,0]-spec['time_s'])<1e-7
  obs=d['observations'][i];mask=d['masks'][i:i+1].astype(bool);actual=int(d['choices'][i])
  obs=np.r_[obs[:300],obs[303:]].astype(np.float32)[None]
 output={}
 for name,model in models.items():
  with torch.no_grad():
   features=model.mlp_extractor.policy_net(torch.from_numpy(obs));logits=model.action_net(features)
   current=model.get_distribution(torch.from_numpy(obs),mask).distribution.probs.numpy()[0]
   options={}
   for temperature in [1.,2.,3.,4.]:
    p=torch.softmax((logits/temperature).masked_fill(~torch.from_numpy(mask),-torch.inf),-1).numpy()[0]
    assert np.isfinite(p).all() and (p[~mask[0]]==0).all()
    assert int(p.argmax())==int(current.argmax())
    options[str(temperature)]=dict(stop_probability=float(p[0]),action9_probability=float(p[9]),movement_probability=float(p[1:].sum()),entropy=float(-np.sum(p[p>0]*np.log(p[p>0]))))
   np.testing.assert_allclose(current,list(options) and torch.softmax(logits.masked_fill(~torch.from_numpy(mask),-torch.inf),-1).numpy()[0],rtol=1e-5,atol=1e-7)
  output[name]=dict(current_argmax=int(current.argmax()),supported=bool(model.recovery_support(torch.from_numpy(obs))[0]),current_probabilities=current.tolist(),hypothetical_temperature=options)
  budget.tick()
 states.append(dict(id=spec['id'],time_s=spec['time_s'],actual_choice=actual,legal_actions=np.flatnonzero(mask[0]).tolist(),models=output))
result=dict(completed=True,actual_ppo_samples=8000,actual_recorded_likelihood_groups=groups,frozen_state_probabilities=states,hypothetical_temperature_not_enabled=True,physical_steps_added=0,models_changed=False,resource=budget.statistics(),limitations=['Masked legal actions need not be useful or safe over a complete maneuver.','A recorded stop likelihood near1only bounds competing mass subject to float precision; it is not an exact rare-action probability.','Hypothetical temperature preserves argmax on these states but does not prove navigation improvement.','No sampled action,old probability,reward,mask or deployment policy was edited.'],goal_achieved=False,deployment_approved=False)
(OUT/'EXPLORATION.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(dict(completed=True,groups=groups,states=[dict(id=s['id'],values={n:dict(stop=v['current_probabilities'][0],action9=v['current_probabilities'][9],temperature4=v['hypothetical_temperature']['4.0']) for n,v in s['models'].items()}) for s in states],resource=result['resource']),indent=2))
