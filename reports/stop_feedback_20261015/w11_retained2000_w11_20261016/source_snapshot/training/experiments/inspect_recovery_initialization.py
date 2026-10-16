"""Compare fitted labels and portable actions on the actual BC input arrays.

Offline fit and export agreement are implementation evidence, not navigation
success. Report per-source movement errors and valid-stop errors separately.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from experiments.recovery_demonstrations import load_demonstrations
from experiments.stop_recovery_policy import StopRecoveryPPO
from experiments.train_prefix_recovery import frozen_base_digest
from neural_navigation.experiment_budget import ExperimentBudget
from neural_navigation.masked_actions import ACTIONS


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['model','demonstrations','output']:
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--retention-frames',type=int,default=1058)
    parser.add_argument('--cpu-cores',type=float,default=.35)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1)
    budget=ExperimentBudget(args.cpu_cores,memory_mib=3072,directory=args.output)
    model=StopRecoveryPPO.load(args.model/'policy_bc.zip',device='cpu')
    config=json.loads((args.model/'config.json').read_text())
    if model.num_timesteps or frozen_base_digest(model.policy)!=config['frozen_base_sha256']:
        raise ValueError('Unchanged base and zero PPO initialization required')
    obs,masks,labels,sources=load_demonstrations(json.loads(args.demonstrations.read_text()),model.policy)
    with np.load(args.model/'demonstrations.npz',allow_pickle=False) as saved:
        for name,value in [('observations',obs),('masks',masks),('choices',labels)]:
            np.testing.assert_array_equal(saved[name],value)
    exported=torch.jit.load(str(args.model/'bc_portable/actor.pt'),map_location='cpu').eval()
    model.policy.set_training_mode(False)
    chunks=[]
    for start in range(0,len(obs),256):
        x=obs[start:start+256]
        with torch.no_grad():
            probabilities=model.policy.get_distribution(torch.from_numpy(x),masks[start:start+256]).distribution.probs.numpy()
            runtime=np.concatenate([x[:,:300],np.zeros((len(x),3),np.float32),x[:,300:]],axis=-1)
            np.testing.assert_array_equal(exported(torch.from_numpy(runtime)).numpy(),ACTIONS[probabilities.argmax(-1)])
        chunks.append(probabilities);budget.tick()
    probabilities=np.concatenate(chunks)
    predicted=probabilities.argmax(-1)
    targets=probabilities[np.arange(len(labels)),labels]
    rows=[];misses=[];offset=0
    for source in sources:
        count=source['recovery_frames'];sl=slice(offset,offset+count)
        wrong=np.flatnonzero(predicted[sl]!=labels[sl])
        rows.append(dict(id=source['id'],frames=count,errors=len(wrong),
            movement_errors=int(np.count_nonzero(labels[sl][wrong])),
            wait_errors=int(np.count_nonzero(labels[sl][wrong]==0)),
            minimum_label_probability=float(targets[sl].min())))
        for local in wrong:
            i=offset+int(local)
            misses.append(dict(source=source['id'],supported_index=int(local),
                target=int(labels[i]),choice=int(predicted[i]),target_probability=float(targets[i])))
        offset+=count
    retention=min(args.retention_frames,len(labels))
    result=dict(completed=True,model=str(args.model),ppo_steps=0,
        model_sha256=hashlib.sha256((args.model/'policy_bc.zip').read_bytes()).hexdigest(),
        total=len(labels),correct=int((predicted==labels).sum()),
        retention_frames=retention,retention_errors=int(np.count_nonzero(predicted[:retention]!=labels[:retention])),
        movement_errors=int(np.count_nonzero((predicted!=labels)&(labels!=0))),
        valid_wait_errors=int(np.count_nonzero((predicted!=labels)&(labels==0))),
        minimum_movement_label_probability=float(targets[labels!=0].min()),
        minimum_wait_label_probability=float(targets[labels==0].min()),
        confidence_target_met=bool(targets.min()>=.9),
        target_probability_quantiles=np.quantile(targets,[0,.01,.05,.5,1]).tolist(),
        sources=rows,misses=misses,actual_training_arrays_verified=True,
        portable_actions_identical_on_all_training_states=True,base_unchanged=True,
        scope='Offline fit and export consistency only; full physical navigation and generalization remain required.',
        resource=budget.statistics(),deployment_approved=False)
    (args.output/'RESULTS.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['sources','misses']},indent=2),flush=True)


if __name__=='__main__':
    main()
