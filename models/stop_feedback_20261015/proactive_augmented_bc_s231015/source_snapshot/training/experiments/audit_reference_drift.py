"""Compare actors on identical archived sensor inputs, without issuing commands.

This diagnoses policy drift and waiting; it does not certify new actions safe.
Archived observations never enter either actor's training buffer.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from sb3_contrib import MaskablePPO
from experiments.anchored_masked_ppo import AnchoredMaskedPPO
from neural_navigation.masked_actions import ACTIONS,masks_from_observation


def fingerprint(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def compare(reference_path,candidate_path,traces):
    torch.set_num_threads(1)
    hashes=[fingerprint(reference_path),fingerprint(candidate_path)]
    reference=AnchoredMaskedPPO.load(reference_path,device='cpu')
    reference.snapshot_reference()
    candidate=MaskablePPO.load(candidate_path,device='cpu')
    reference.policy.set_training_mode(False);candidate.policy.set_training_mode(False)
    results=[]
    for path in traces:
        trace=json.loads(path.read_text());rows=trace['samples']
        obs=np.asarray([r['policy_observation'] for r in rows],np.float32)
        masks=masks_from_observation(obs)
        with torch.no_grad():
            inputs=torch.from_numpy(obs);mask_tensor=torch.from_numpy(masks)
            old=reference.policy.get_distribution(inputs,action_masks=mask_tensor).distribution
            # Clone before another distribution call: SB3 holds mutable distributions.
            old_probs=old.probs.clone()
            new=candidate.policy.get_distribution(inputs,action_masks=mask_tensor).distribution
            new_probs=new.probs.clone()
            kl=torch.distributions.kl_divergence(
                torch.distributions.Categorical(probs=old_probs),
                torch.distributions.Categorical(probs=new_probs)).numpy()
            eligible=reference.eligible(inputs,mask_tensor).numpy()
        old_choice=old_probs.argmax(dim=1).numpy();new_choice=new_probs.argmax(dim=1).numpy()
        proposed=np.asarray([r['proposed_action'] for r in rows],np.float32)
        np.testing.assert_allclose(ACTIONS[old_choice],proposed,atol=2e-6,rtol=0)
        if not masks[np.arange(len(obs)),new_choice].all():raise ValueError('Invalid masked choice')
        alternative=masks[:,1:].any(axis=1)
        groups=dict(all=np.ones(len(obs),bool),protected_clear=eligible,
            moving_supported=obs[:,281]>.5,
            previous_rejection=obs[:,250]<.5,
            reference_wait_with_alternative=(old_choice==0)&alternative)
        summary={}
        for name,keep in groups.items():
            n=int(keep.sum())
            summary[name]=dict(frames=n,changed_choices=int(((old_choice!=new_choice)&keep).sum()),
                mean_kl=float(kl[keep].mean()) if n else None,
                p95_kl=float(np.quantile(kl[keep],.95)) if n else None,
                reference_wait_frames=int(((old_choice==0)&alternative&keep).sum()),
                candidate_wait_frames=int(((new_choice==0)&alternative&keep).sum()),
                candidate_nonzero_after_reference_wait=int(((old_choice==0)&(new_choice!=0)&keep).sum()))
        results.append(dict(trace=str(path.resolve()),trace_sha256=fingerprint(path),
            recorded_reference_outcome=trace['result'],groups=summary))
    if [fingerprint(reference_path),fingerprint(candidate_path)]!=hashes:
        raise ValueError('Checkpoint changed during comparison')
    return dict(reference_policy_sha256=hashes[0],candidate_policy_sha256=hashes[1],results=results,
        reference_recorded_action_parity=True,commands_issued=False,training_inputs_added=False,
        limitation='Identical recorded observations/masks only. Changed nonzero choices are not proven safe or successful; '
            'group averages are not a global KL guarantee. Closed-loop evaluation is required.')


def main():
    p=argparse.ArgumentParser();p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--candidate',type=Path,required=True)
    p.add_argument('--trace',type=Path,action='append',required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():p.error('Fresh output required')
    result=compare(a.reference,a.candidate,a.trace)
    a.output.write_text(json.dumps(result,indent=2,allow_nan=False))
    print(json.dumps([dict(trace=r['trace'],groups=r['groups']) for r in result['results']]))


if __name__=='__main__':main()
