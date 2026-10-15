"""Measure saved actor sensitivity on recorded inputs, without commands.

Ablations are synthetic diagnostic inputs, not valid alternate sensor states
or proof of semantic understanding. They never enter training.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from sb3_contrib import MaskablePPO
from neural_navigation.masked_actions import masks_from_observation


def audit(policy_path,trace_path):
    trace=json.loads(trace_path.read_text())
    digest=hashlib.sha256(policy_path.read_bytes()).hexdigest()
    if trace['result']['policy_sha256']!=digest:raise ValueError('Trace and actor do not match')
    model=MaskablePPO.load(policy_path,device='cpu');model.policy.set_training_mode(False)
    obs=np.asarray([r['policy_observation'] for r in trace['samples']],np.float32)
    if obs.ndim!=2 or obs.shape[1]!=300 or not np.isfinite(obs).all():raise ValueError('Recorded quality inputs required')
    masks=masks_from_observation(obs[:,:282]);tensor=torch.from_numpy(obs)
    with torch.no_grad():original=model.policy.get_distribution(tensor,action_masks=masks).distribution.probs.clone()
    choices=original.argmax(dim=1).numpy()
    np.testing.assert_array_equal(choices,[r['choice'] for r in trace['samples']])
    variants={}
    for name,start,end in [('added_risk_zero',282,294),('quality_flags_zero',294,300),('all_added_zero',282,300),('previous_reason_to_none',250,266)]:
        changed=obs.copy();changed[:,start:end]=0
        if name=='previous_reason_to_none':changed[:,250]=1
        with torch.no_grad():
            altered=model.policy.get_distribution(torch.from_numpy(changed),action_masks=masks).distribution.probs.clone()
            kl=torch.distributions.kl_divergence(torch.distributions.Categorical(probs=original),
                torch.distributions.Categorical(probs=altered)).numpy()
        new_choices=altered.argmax(dim=1).numpy()
        groups=dict(all=np.ones(len(obs),bool),moving_observed=obs[:,295]>.5,
            moving_held=obs[:,296]>.5,velocity_unknown=obs[:,297]>.5,
            pose_unsupported=obs[:,294]<.5,previous_rejection=obs[:,250]<.5,
            wait_with_current_alternative=(choices==0)&masks[:,1:].any(axis=1))
        result={}
        for group,keep in groups.items():
            n=int(keep.sum());result[group]=dict(frames=n,
                changed_choices=int(((choices!=new_choices)&keep).sum()),
                mean_kl=float(kl[keep].mean()) if n else None,
                max_probability_change=float(torch.abs(original-altered)[torch.from_numpy(keep)].max()) if n else None)
        variants[name]=result
    if hashlib.sha256(policy_path.read_bytes()).hexdigest()!=digest:raise ValueError('Actor changed')
    return dict(policy_sha256=digest,trace_sha256=hashlib.sha256(trace_path.read_bytes()).hexdigest(),
        frames=len(obs),recorded_action_parity=True,ablation=variants,
        limitation='Probability sensitivity only; ablations may be inconsistent synthetic inputs. '
            'Does not establish causal correctness, semantic understanding, or safe recovery. No commands/learning.')


def main():
    p=argparse.ArgumentParser();p.add_argument('--policy',type=Path,required=True)
    p.add_argument('--trace',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():p.error('Fresh output required')
    torch.set_num_threads(1);result=audit(a.policy,a.trace)
    a.output.write_text(json.dumps(result,indent=2));print(json.dumps(result))


if __name__=='__main__':main()
