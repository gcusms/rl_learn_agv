"""Frozen five-case recovery diagnostics; no action substitution or training."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from sb3_contrib import MaskablePPO
import experiments.evaluate_quality_regression as evaluator
from experiments.time_consistent_tracking import TimeConsistentFreshRuntime
from experiments.train_quality_navigation import digest


def behavior(trace,model):
    observations=np.asarray([r['policy_observation'] for r in trace['samples']],np.float32)
    masks=np.asarray([r['mask'] for r in trace['samples']],bool)
    choices=np.asarray([r['choice'] for r in trace['samples']])
    eligible=(masks[:,1:].any(axis=1))&(choices==0)
    previous_stop=observations[:,250]<.5
    with torch.no_grad():
        original=model.policy.get_distribution(torch.from_numpy(observations),action_masks=masks)
        probabilities=original.distribution.probs.clone()
        hypothetical=observations.copy()
        hypothetical[:,250:266]=0;hypothetical[:,250]=1;hypothetical[:,266]=0
        altered=model.policy.get_distribution(torch.from_numpy(hypothetical),action_masks=masks)
        changed=(altered.mode().numpy()!=choices)
        probability_delta=(altered.distribution.probs-probabilities).abs().sum(-1).numpy()/2
    return dict(wait_with_alternative_frames=int(eligible.sum()),
        previous_stop_with_alternative_frames=int((previous_stop&masks[:,1:].any(axis=1)).sum()),
        mean_sampled_nonzero_probability_while_greedy_waiting=float((1-probabilities[eligible,0]).mean()) if eligible.any() else None,
        stop_feedback_ablation_changed_greedy_frames=int((changed&previous_stop).sum()),
        stop_feedback_ablation_mean_probability_distance=float(probability_delta[previous_stop].mean()) if previous_stop.any() else None,
        limitation='Probability and feedback sensitivity diagnostics only. Ablation does not execute commands and cannot establish understanding or future safety.')


def main():
    p=argparse.ArgumentParser()
    for name in ('bundle','policy','reference-run','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():p.error('Fresh output required')
    torch.set_num_threads(1);a.output.mkdir()
    manifest=json.loads((a.bundle/'manifest.json').read_text())
    assert manifest['policy_sha256']==digest(a.policy)
    assert json.loads((a.bundle/'hardware.json').read_text())['near_field_memory']['clearance_increasing_rotation']
    model=MaskablePPO.load(a.policy,device='cpu')
    project=Path(__file__).resolve().parents[2]
    sources={str(path.relative_to(project)):digest(path) for path in (project/'training').rglob('*.py')
        if 'runs' not in path.relative_to(project).parts}
    report=dict(completed=False,cases=[],started_unix=time.time(),policy_sha256=digest(a.policy),
        bundle_manifest_sha256=digest(a.bundle/'manifest.json'),source_sha256=sources,
        captured_before_evaluation=True,deployment_approved=False,
        limitation='Five reused development cases, not independent acceptance or real hardware validation.')
    def save():(a.output/'RESULTS.json').write_text(json.dumps(report,indent=2))
    save();evaluator.QualitySensorNavigationRuntime=TimeConsistentFreshRuntime
    for cid in ('independent_dynamic_006','independent_dynamic_021','independent_dynamic_023',
                'motion_profile_005','train_failure_w11_e7'):
        case=json.loads((a.reference_run/(cid+'.json')).read_text())['case']
        trace=evaluator.evaluate(case,a.bundle)
        obs=np.asarray([r['policy_observation'] for r in trace['samples']],np.float32)
        masks=np.asarray([r['mask'] for r in trace['samples']],bool)
        np.testing.assert_array_equal(model.predict(obs,action_masks=masks,deterministic=True)[0],
            [r['choice'] for r in trace['samples']])
        path=a.output/(cid+'.json');path.write_text(json.dumps(trace,allow_nan=False))
        report['cases'].append(dict(trace['result'],behavior=behavior(trace,model),trace_sha256=digest(path)))
        report['completed']=len(report['cases'])==5;save();print(json.dumps(trace['result']),flush=True)
    report.update(finished_unix=time.time(),source_hashes_unchanged=all(digest(project/path)==expected for path,expected in sources.items()))
    save()


if __name__=='__main__':main()
