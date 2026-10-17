"""Measure the Gaussian proposals on exact saved actor inputs, never commands."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import numpy as np


def cdf(value):return .5*math.erfc(-value/math.sqrt(2.))


def clipped_interval_mass(mean,std,lower,upper):
    """Gaussian mass whose clipping to [-1,1] lands in this interval."""
    if std<=0 or not np.isfinite([mean,std,lower,upper]).all() or lower>upper:
        raise ValueError('Finite ordered interval and positive standard deviation required')
    if upper < -1 or lower > 1:return 0.
    low=0. if lower<=-1 else cdf((lower-mean)/std)
    high=1. if upper>=1 else cdf((upper-mean)/std)
    return max(0.,high-low)


def normalized_v(v,hardware):
    return v/(hardware['forward_limit_m_s'] if v>=0 else hardware['reverse_limit_m_s'])


def analyze_inputs(trace,model,hardware):
    rows=[x for x in trace['samples'] if x.get('policy_observation') is not None]
    if not rows:return dict(case_id=trace['case']['id'],available=False,reason='Exact actor inputs were not captured')
    import torch
    observations=np.asarray([x['policy_observation'] for x in rows],dtype=np.float32)
    if observations.shape[1:]!=model.observation_space.shape or not np.isfinite(observations).all():
        raise ValueError('Captured actor observation dimension/value mismatch')
    if model.use_sde or model.policy.squash_output:raise ValueError('Independent nonsquashed Gaussian required')
    with torch.no_grad():
        distribution=model.policy.get_distribution(torch.from_numpy(observations)).distribution
        means=distribution.mean.cpu().numpy();stds=distribution.stddev.cpu().numpy()
    proposed=np.asarray([x['proposed_action'] for x in rows])
    np.testing.assert_allclose(np.clip(means,-1,1),proposed,atol=2e-6,rtol=2e-6)
    probes=hardware['policy_observation']['probes_m_s_rad_s'];selected=[]
    for row,mean,std in zip(rows,means,stds):
        flags=row['safety_context']['action_feasibility']
        reason=row['stop_reason']
        if reason in (None,'estimated_goal_reached') or not any(flags):continue
        masses=[]
        for flag,(v,w) in zip(flags,probes):
            if not flag:continue
            vx=clipped_interval_mass(float(mean[0]),float(std[0]),normalized_v(v-.005,hardware),normalized_v(v+.005,hardware))
            wz=clipped_interval_mass(float(mean[1]),float(std[1]),(w-.05)/hardware['yaw_limit_rad_s'],(w+.05)/hardware['yaw_limit_rad_s'])
            masses.append(vx*wz)
        # These nine small neighborhoods do not overlap in the configured
        # contracts, except at boundaries of zero probability measure.
        selected.append(dict(timestamp=row['timestamp'],guard_reason=reason,
            raw_mean=mean.tolist(),std=std.tolist(),probe_neighborhood_mass=float(sum(masses))))
    return dict(case_id=trace['case']['id'],available=True,captured_frames=len(rows),
        saturated_mean_fraction=np.mean(np.abs(means)>1,axis=0).tolist(),
        min_raw_mean=means.min(axis=0).tolist(),max_raw_mean=means.max(axis=0).tolist(),
        projected_frames_with_accepted_probe=len(selected),
        median_neighborhood_mass=float(np.median([x['probe_neighborhood_mass'] for x in selected])) if selected else None,
        selected_frames=selected,
        limitation='Proposal mass within +/-0.005m/s and +/-0.05rad/s of present-frame probes, '
        'not probability of safe execution. Nearby velocities may be unsafe, and other valid actions may exist.')


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True)
    p.add_argument('--traces',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():p.error('Fresh output required')
    from stable_baselines3 import PPO
    import torch
    torch.set_num_threads(1)
    manifest=json.loads((a.run/'portable/manifest.json').read_text());policy=a.run/'candidate/policy.zip'
    if hashlib.sha256(policy.read_bytes()).hexdigest()!=manifest['policy_sha256']:p.error('Checkpoint/export mismatch')
    model=PPO.load(policy,device='cpu');hardware=json.loads((a.run/'portable/hardware.json').read_text())
    cases=[analyze_inputs(json.loads(path.read_text()),model,hardware) for path in sorted(a.traces.glob('*.json'))]
    a.output.write_text(json.dumps(dict(policy_sha256=manifest['policy_sha256'],cases=cases),indent=2))
    for case in cases:print(case['case_id'],case['available'],case.get('projected_frames_with_accepted_probe'))


if __name__=='__main__':main()
