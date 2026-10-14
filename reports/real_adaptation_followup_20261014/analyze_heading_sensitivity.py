"""Offline counterfactual task-input sensitivity, not a navigation test."""
from pathlib import Path
import json
import numpy as np
import torch
from sb3_contrib import MaskablePPO
from neural_navigation.masked_actions import masks_from_observation,ACTIONS

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
RUN=ROOT/'training/runs/goal_pose_balanced_3k_20261014'
EVAL=ROOT/'training/runs/goal_pose_paired_evaluation_20261014'


def main():
    torch.set_num_threads(1);report=json.loads((EVAL/'RESULTS.json').read_text());assert report['completed']
    observations=[]
    for row in report['rows']:
        if row['variant']!='initial' or row['category']=='position_regression':continue
        with np.load(EVAL/f"{row['case_id']}_initial.npz") as a:
            obs=a['observations'][a['labels'][:,7]<.2]
            if len(obs)>16:obs=obs[np.linspace(0,len(obs)-1,16).astype(int)]
            observations.extend(obs)
    inputs=np.array(observations,np.float32)
    if not len(inputs):raise ValueError('No sampled near-goal inputs for diagnostic')
    left=inputs.copy();right=inputs.copy();aligned=inputs.copy()
    left[:,300:]=[1.,0.,1.];right[:,300:]=[-1.,0.,1.];aligned[:,300:]=[0.,1.,1.]
    masks=masks_from_observation(inputs[:,:282]);rows={}
    for label,name in [('initial','policy_initial.zip'),('trained','policy.zip')]:
        model=MaskablePPO.load(RUN/name,device='cpu');model.policy.set_training_mode(False)
        with torch.no_grad():
            pl=model.policy.get_distribution(torch.from_numpy(left),action_masks=masks).distribution.probs.numpy()
            pr=model.policy.get_distribution(torch.from_numpy(right),action_masks=masks).distribution.probs.numpy()
            pa=model.policy.get_distribution(torch.from_numpy(aligned),action_masks=masks).distribution.probs.numpy()
        variation=.5*abs(pl-pr).sum(axis=1);cl=pl.argmax(axis=1);cr=pr.argmax(axis=1)
        rows[label]=dict(inputs=len(inputs),mean_total_variation=float(variation.mean()),
            maximum_total_variation=float(variation.max()),
            deterministic_action_changed=int((cl!=cr).sum()),
            opposite_sign_rotation_choices=int(((ACTIONS[cl,1]*ACTIONS[cr,1])<0).sum()),
            zero_heading_error_mean_stop_probability=float(pa[:,0].mean()),
            zero_heading_error_stop_choices=int((pa.argmax(axis=1)==0).sum()),
            first_layer_heading_columns_l2=float(model.policy.mlp_extractor.policy_net[0].weight[:,300:].detach().norm()))
        del model
    result=dict(scope='Counterfactual +/-90-degree heading error on identical near-goal observations and masks from INITIAL evaluation trajectories; not independent success or physical accuracy',
        selected_with='physical near-goal labels for offline sampling only; no truth appended to policy observations',rows=rows)
    (OUT/'heading_sensitivity.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
    settling=[]
    for row in report['rows']:
        if row['category']=='position_regression':continue
        with np.load(EVAL/f"{row['case_id']}_{row['variant']}.npz") as a:
            labels=a['labels'];within=(labels[:,7]<.12)&(labels[:,8]<.15)
            stopped=(abs(labels[:,9])<.03)&(abs(labels[:,10])<.08)
            run=0;longest=0
            for good in within&stopped:
                run=run+1 if good else 0;longest=max(longest,run)
            settling.append(dict(case_id=row['case_id'],variant=row['variant'],
                pose_within_tolerance_frames=int(within.sum()),stopped_and_aligned_frames=int((within&stopped).sum()),
                longest_stopped_and_aligned_seconds=max(0,longest-1)*.2,
                angular_speed_median_when_aligned_rad_s=float(np.median(abs(labels[within,10]))) if within.any() else None))
    (OUT/'settling_diagnosis.json').write_text(json.dumps(settling,indent=2))
    parity={}
    for variant,checkpoint in [('legacy',ROOT/'models/quality_recovery_20261014/policy_2000.zip'),
        ('initial',RUN/'policy_initial.zip'),('trained',RUN/'policy.zip')]:
        model=MaskablePPO.load(checkpoint,device='cpu');model.policy.set_training_mode(False);frames=0
        for row in report['rows']:
            if row['variant']!=variant:continue
            with np.load(EVAL/f"{row['case_id']}_{variant}.npz") as a:
                native=model.predict(a['observations'],action_masks=a['masks'],deterministic=True)[0]
                np.testing.assert_array_equal(native,a['choices']);frames+=len(native)
        parity[variant]=dict(all_matching=True,frames=frames)
        del model
    (OUT/'native_portable_trace_parity.json').write_text(json.dumps(parity,indent=2));print(parity)


if __name__=='__main__':main()
