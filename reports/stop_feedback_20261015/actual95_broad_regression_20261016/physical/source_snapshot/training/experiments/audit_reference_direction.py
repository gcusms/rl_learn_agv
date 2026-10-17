"""Read-only reference gradient diagnostic at the first matched-input divergence."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from sb3_contrib import MaskablePPO
from experiments.train_quality_navigation import digest


def main():
    parser=argparse.ArgumentParser()
    for name in ('reference','learned','regressed','learned-trace','regressed-trace','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():parser.error('Fresh output required')
    a=json.loads(args.learned_trace.read_text());b=json.loads(args.regressed_trace.read_text())
    if a['result']['policy_sha256']!=digest(args.learned) or b['result']['policy_sha256']!=digest(args.regressed):
        raise ValueError('Actor/trace hash mismatch')
    if a['case']!=b['case']:raise ValueError('Same recorded case required')
    index=next(i for i,(x,y) in enumerate(zip(a['samples'],b['samples'])) if x['choice']!=y['choice'])
    x=a['samples'][index];y=b['samples'][index]
    np.testing.assert_array_equal(x['policy_observation'],y['policy_observation'])
    np.testing.assert_array_equal(x['mask'],y['mask'])
    torch.set_num_threads(1);obs=torch.tensor([x['policy_observation']],dtype=torch.float32)
    masks=np.array([x['mask']],bool);models={};probabilities={}
    for label,path in [('reference',args.reference),('learned',args.learned),('regressed',args.regressed)]:
        model=MaskablePPO.load(path,device='cpu');model.policy.set_training_mode(False);models[label]=model
        probabilities[label]=model.policy.get_distribution(obs,action_masks=masks).distribution.probs
    if int(probabilities['learned'].argmax())!=x['choice'] or int(probabilities['regressed'].argmax())!=y['choice']:
        raise ValueError('Actual recorded choices not reproduced')
    teacher=probabilities['reference'].detach();current=probabilities['learned']
    kl=torch.distributions.kl_divergence(torch.distributions.Categorical(probs=teacher),torch.distributions.Categorical(probs=current)).sum()
    actor=[(n,p) for n,p in models['learned'].policy.named_parameters() if n.startswith(('mlp_extractor.policy_net.','action_net.'))]
    gradients=torch.autograd.grad(kl,[p for _,p in actor]);other=dict(models['regressed'].policy.named_parameters())
    dot=sum(float(torch.sum(g*(other[n].detach()-p.detach()))) for (n,p),g in zip(actor,gradients))
    result=dict(first_divergence_frame=index,timestamp=x['timestamp'],same_actual_observation_and_mask=True,
        probabilities={k:v.detach().flatten().tolist() for k,v in probabilities.items()},
        choices={k:int(v.argmax()) for k,v in probabilities.items()},reference_kl_at_learned=float(kl.detach()),
        reference_gradient_dot_actual_parameter_change=dot,
        actual_change_aligned_with_reference_descent=dot<0,
        policy_sha256={k:digest(p) for k,p in [('reference',args.reference),('learned',args.learned),('regressed',args.regressed)]},
        limitation='Single-state local gradient alignment; does not prove training-level causation or safety. No commands or learning.')
    args.output.write_text(json.dumps(result,indent=2));print(json.dumps(result))


if __name__=='__main__':main()
