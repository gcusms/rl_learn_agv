"""Measure frozen-prior margins versus learned residual on failure inputs."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from sb3_contrib import MaskablePPO
from neural_navigation.masked_actions import masks_from_observation,COMMANDS
from experiments.anticipatory_recovery_ppo import anticipatory_support


def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--trace',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();torch.set_num_threads(1);z=np.load(a.trace)
    model=MaskablePPO.load(a.checkpoint,device='cpu');policy=model.policy;head=policy.action_net
    x=torch.from_numpy(z['observations'][:,:300]);mask=masks_from_observation(x.numpy()[:,:282])
    with torch.no_grad():
        features=policy.mlp_extractor.policy_net(policy.extract_features(x))
        support=features[:,-1:];scale=1+(head.prior_scale-1)*support
        prior=(head.base(features[:,:head.base_width])*scale).numpy()
        residual=(support*head.residual(features[:,head.base_width:-1])).numpy()
        eligible=anticipatory_support(x).numpy()
    rows=[]
    for i,t in enumerate(z['labels'][:,0]):
        legal=np.flatnonzero(mask[i]);rank=legal[np.argsort(prior[i,legal])]
        fast=legal[COMMANDS[legal,0]>.19]
        rows.append(dict(time_s=float(t),actor_eligible=bool(eligible[i]),legal_moving=int(mask[i,1:].sum()),
            prior_winner=int(rank[-1]),adapted_winner=int(np.argmax(np.where(mask[i],prior[i]+residual[i],-np.inf))),
            prior_margin=float(prior[i,rank[-1]]-prior[i,rank[-2]]) if len(rank)>1 else None,
            prior_gap_to_best_legal_fast=float(prior[i,rank[-1]]-np.max(prior[i,fast])) if len(fast) else None,
            max_absolute_residual=float(abs(residual[i,legal]).max())))
    report=dict(scope='Fixed failure sensor history, logit scale diagnosis only; not safety certification',
        checkpoint=str(a.checkpoint.resolve()),rows=rows)
    a.output.write_text(json.dumps(report,indent=2)+'\n')
    for row in rows:
        if 7.3<row['time_s']<8.8:print(json.dumps(row))


if __name__=='__main__':main()
