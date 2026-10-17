"""Verify complete paired quality-policy diagnostics, without simulation/training."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from sb3_contrib import MaskablePPO
from experiments.audit_stop_feedback import audit
from neural_navigation.masked_actions import masks_from_observation,ACTIONS,COMMANDS


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(directory):
    config=json.loads((directory/'candidate/config.json').read_text())
    summary=json.loads((directory/'MATCHED_RESULTS.json').read_text())
    if not summary['completed'] or len(summary['results'])!=1+len(config['evaluation_seeds']):
        raise ValueError('Incomplete matched diagnostic')
    project=Path(__file__).resolve().parents[2]
    for relative,expected in config['source_sha256'].items():
        if digest(project/relative)!=expected:raise ValueError(f'Launched source changed: {relative}')
    results=[]
    for label,filename,portable in [('initial','policy_initial.zip','initial_portable'),('final','policy.zip','portable')]:
        policy=directory/'candidate'/filename;model=MaskablePPO.load(policy,device='cpu')
        if getattr(model,'training_reward_scale',1.)!=config['training_reward_scale']:
            raise ValueError('Saved value and reward units disagree')
        manifest=json.loads((directory/portable/'manifest.json').read_text())
        if manifest['policy_sha256']!=digest(policy):raise ValueError('Export model hash mismatch')
        for name,expected in manifest['files'].items():
            if digest(directory/portable/name)!=expected:raise ValueError('Export file hash mismatch')
        actor=torch.jit.load(str(directory/portable/'actor.pt'))
        for index,pair in enumerate(summary['results']):
            path=directory/f'{label}_{index}.json';data=json.loads(path.read_text());rows=data['samples']
            if data['result']!=pair[label] or data['result']['policy_sha256']!=digest(policy):
                raise ValueError('Recorded summary/model mismatch')
            other=json.loads((directory/f'{"final" if label=="initial" else "initial"}_{index}.json').read_text())
            if data['case']!=other['case']:raise ValueError('Paired initial conditions differ')
            obs=np.asarray([r['policy_observation'] for r in rows],np.float32)
            if obs.shape!=(len(rows),300) or not np.isfinite(obs).all():raise ValueError('Finite actual quality inputs required')
            masks=masks_from_observation(obs[:,:282]);choices=np.asarray([r['choice'] for r in rows])
            np.testing.assert_array_equal(masks,[r['mask'] for r in rows])
            np.testing.assert_array_equal(model.predict(obs,action_masks=masks,deterministic=True)[0],choices)
            if not masks[np.arange(len(rows)),choices].all():raise ValueError('Rejected choice executed')
            with torch.no_grad():
                np.testing.assert_allclose(actor(torch.from_numpy(obs)).numpy(),ACTIONS[choices],atol=0,rtol=0)
                values=model.policy.predict_values(torch.from_numpy(obs)).flatten().numpy()
            np.testing.assert_allclose([r['proposed_action'] for r in rows],ACTIONS[choices],atol=0,rtol=0)
            nonzero=choices!=0
            np.testing.assert_allclose(np.asarray([r['command'] for r in rows])[nonzero],COMMANDS[choices[nonzero]],atol=2e-8,rtol=0)
            feedback=audit(dict(case=data['case'],samples=[dict(r,policy_observation=r['policy_observation'][:282]) for r in rows]))
            if feedback['counts']['missing'] or feedback['counts']['feedback_mismatch']:raise ValueError('Parking feedback disagrees')
            voluntary=int(((choices==0)&masks[:,1:].any(axis=1)).sum())
            forced=int((~masks[:,1:].any(axis=1)).sum())
            if voluntary!=pair[label]['voluntary_wait_frames'] or forced!=pair[label]['only_stop_frames'] or len(rows)!=pair[label]['frames']:
                raise ValueError('Recorded waiting counts disagree')
            results.append(dict(label=label,index=index,trace_sha256=digest(path),**pair[label],
                previous_rejection_frames=int((obs[:,250]<.5).sum()),value_std=float(values.std()),feedback_verified=True))
    return dict(executions=len(results),pairs=len(summary['results']),results=results,source_hashes_unchanged=True,
        recorded_actor_export_and_feedback_verified=True,
        limitation='Nine paired development tasks, including one reused training failure. Not the full 48-task regression or final acceptance; replay parity does not prove semantic understanding.')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    if args.output.exists():parser.error('Fresh output required')
    torch.set_num_threads(1);result=verify(args.run)
    args.output.write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='results'}))


if __name__=='__main__':main()
