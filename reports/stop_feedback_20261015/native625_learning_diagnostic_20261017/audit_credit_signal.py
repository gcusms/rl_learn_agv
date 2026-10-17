"""Reconstruct actual minibatch normalization and delayed terminal credit."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from neural_navigation.experiment_budget import ExperimentBudget


ROOT=Path(__file__).resolve().parents[3]
BASE=ROOT/'reports/stop_feedback_20261015'
HERE=Path(__file__).resolve().parent
MODELS=ROOT/'models/stop_feedback_20261015/native625_suite_ppo_s231017'


def eligibility(x,masks):
    history=x[:,439:481].reshape(-1,3,14)
    context=(history[:,:,12]>.5).any(1)|(x[:,433:435]>0.).any(1)
    near=x[:,144:216].min(1)*12.<.8
    restricted=~masks[:,1:].all(1)
    forecast=x[:,419:431].max(1)>0.
    valid=(x[:,435]>.5)&(x[:,438]>.5)&masks[:,1:].any(1)&(np.linalg.norm(x[:,218:220]*12.,axis=1)>.3)
    return valid&(context|near|restricted|forecast)


def main():
    output=HERE/'CREDIT_SIGNAL_AUDIT.json'
    assert not output.exists()
    torch.set_num_threads(1)
    budget=ExperimentBudget(.1,memory_mib=1100,directory=HERE)
    snapshot=HERE/'STAGE_ONE_ROWS.json'
    rows=json.loads(snapshot.read_text())
    source=BASE/'normal_suite_wait_protocol_v2_20261016/demonstrations.npz'
    with np.load(source,allow_pickle=False) as data:
        labels=data['choices'].copy()
    sizes=[int((labels==0).sum()),int((labels!=0).sum())]
    files=[Path(__file__),snapshot,source]
    results=[]
    all_old,all_mc,spans=[],[],[]
    generator=torch.Generator().manual_seed(231018)
    for row in rows:
        trace=ROOT/row['rollout_path']
        assert hashlib.sha256(trace.read_bytes()).hexdigest()==row['rollout_sha256']
        files.append(trace)
        with np.load(trace,allow_pickle=False) as data:
            x,masks,advantages,values,rewards=[data[k].copy() for k in ['observations','masks','advantages','old_values','rewards']]
        keep=eligibility(x,masks)
        assert int(keep.sum())==row['eligible_actor_decisions']
        returns=np.zeros(len(rewards),np.float64)
        carry=0.
        for i in range(len(rewards)-1,-1,-1):
            carry=float(rewards[i])+.99*carry
            returns[i]=carry
        mc=returns-values
        first_sample=[]
        update_index=0
        for epoch in range(4):
            order=torch.randperm(len(rewards),generator=generator).numpy()
            for start in range(0,len(order),64):
                selected=order[start:start+64]
                # Consume the exact anchor RNG calls used by the real trainer.
                torch.randint(sizes[0],(64,),generator=generator)
                torch.randint(sizes[1],(64,),generator=generator)
                indices=selected[keep[selected]]
                if 0 in indices:
                    at=int(np.flatnonzero(indices==0)[0])
                    actual=torch.from_numpy(advantages[indices].copy())
                    longer=torch.from_numpy(mc[indices].astype(np.float32))
                    if len(indices)>1:
                        actual=(actual-actual.mean())/(actual.std(unbiased=False)+1e-8)
                        longer=(longer-longer.mean())/(longer.std(unbiased=False)+1e-8)
                    first_sample.append(dict(epoch=epoch,update_index=update_index,
                        eligible_minibatch_samples=len(indices),
                        original_normalized_advantage=float(actual[at]),
                        lambda1_same_minibatch_normalized_advantage=float(longer[at]),
                        actor_update_was_applied=bool(row['updates'][update_index]['actor_updated'])))
                update_index+=1
        checkpoint=MODELS/f"episode_{row['episode']:02d}.pt"
        saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
        assert torch.equal(generator.get_state(),saved['minibatch_rng'])
        assert update_index==len(row['updates'])
        files.append(checkpoint)
        start=len(all_mc)
        all_old.extend(advantages[keep].tolist());all_mc.extend(mc[keep].tolist())
        spans.append((start,len(all_mc)))
        results.append(dict(episode=row['episode'],safe_success=row['safe_success'],
            decisions=len(rewards),only_stop_frames=row['only_stop_frames'],
            initial_raw_advantage=float(advantages[0]),initial_mc_advantage=float(mc[0]),
            direct_terminal_contribution_to_initial_gae=float(row['reward_terms']['terminal']*(.99*.95)**(len(rewards)-1)),
            exact_minibatch_rng_reproduced=True,initial_action_minibatches=first_sample))
        budget.tick()
    for (start,end),result in zip(spans,results):
        result['descriptive_cross_episode_mc_mean']=float(np.mean(all_mc[start:end]))
    budget.tick()
    result=dict(completed=True,episodes=results,
        original_gamma=.99,original_lambda=.95,
        comparison_lambda=1.,comparison_is_read_only_not_retrained_or_physically_validated=True,
        no_mixing_logged_policies_as_fresh_ppo_data=True,
        statistics_do_not_alone_prove_root_cause=True,
        inputs={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in set(files)},
        resource=budget.statistics(),no_optimizer_updates=True,no_new_physics=True,
        new_ppo_steps=0,goal_achieved=False,deployment_approved=False)
    output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['inputs']},ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
