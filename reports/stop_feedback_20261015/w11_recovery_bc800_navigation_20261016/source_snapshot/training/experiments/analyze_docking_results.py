"""Evidence from saved physical traces: parity and heading input sensitivity."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from sb3_contrib import MaskablePPO
from neural_navigation.masked_actions import ACTIONS,masks_from_observation


def main():
    p=argparse.ArgumentParser();p.add_argument('--evaluation',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--baseline-checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();torch.set_num_threads(1)
    report=json.loads((a.evaluation/'RESULTS.json').read_text());assert report['completed']
    model=MaskablePPO.load(a.checkpoint,device='cpu');baseline=MaskablePPO.load(a.baseline_checkpoint,device='cpu')
    frames=0;regression=0;observations=[]
    for row in report['rows']:
        if row['variant'] not in ('trained','trained_v4_localization'):continue
        with np.load(a.evaluation/f"{row['case_id']}_{row['variant']}.npz") as data:
            predicted=model.predict(data['observations'],action_masks=data['masks'],deterministic=True)[0]
            np.testing.assert_array_equal(predicted,data['choices']);frames+=len(predicted)
            if row['category']=='position_regression':
                old=baseline.predict(data['observations'],action_masks=data['masks'],deterministic=True)[0]
                np.testing.assert_array_equal(old,predicted);regression+=len(predicted)
            else:
                near=data['observations'][data['labels'][:,7]<.10]
                if len(near)>16:near=near[np.linspace(0,len(near)-1,16).astype(int)]
                observations.extend(near)
    inputs=np.array(observations,np.float32);masks=masks_from_observation(inputs[:,:282]);rows={}
    left=inputs.copy();right=inputs.copy();aligned=inputs.copy()
    left[:,300:]=[1,0,1];right[:,300:]=[-1,0,1];aligned[:,300:]=[0,1,1]
    for name,policy in [('previous_3k',baseline),('candidate',model)]:
        with torch.no_grad():
            pl=policy.policy.get_distribution(torch.from_numpy(left),action_masks=masks).distribution.probs.numpy()
            pr=policy.policy.get_distribution(torch.from_numpy(right),action_masks=masks).distribution.probs.numpy()
            pa=policy.policy.get_distribution(torch.from_numpy(aligned),action_masks=masks).distribution.probs.numpy()
        cl=pl.argmax(axis=1);cr=pr.argmax(axis=1)
        rows[name]=dict(inputs=len(inputs),mean_total_variation=float((.5*abs(pl-pr).sum(axis=1)).mean()),
            changed_choices=int((cl!=cr).sum()),opposite_rotation_sign_choices=int((ACTIONS[cl,1]*ACTIONS[cr,1]<0).sum()),
            aligned_mean_stop_probability=float(pa[:,0].mean()),aligned_stop_choices=int((pa.argmax(axis=1)==0).sum()))
    result=dict(native_portable_parity=dict(all_matching=True,frames=frames),
        position_only_navigation_branch_parity=dict(all_matching=True,frames=regression),
        heading_counterfactuals=rows,scope='trace parity and task-input sensitivity; not new physical successes')
    a.output.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))


if __name__=='__main__':main()
