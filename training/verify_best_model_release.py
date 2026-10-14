"""Verify release hashes and actor parity on recorded historical inputs."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from sb3_contrib import MaskablePPO
from neural_navigation.masked_actions import ACTIONS, COMMANDS, masks_from_observation


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(bundle, project):
    manifest=json.loads((bundle/'RELEASE_MANIFEST.json').read_text())
    for name, expected in manifest['files'].items():
        if digest(bundle/name)!=expected: raise ValueError(f'Changed package file: {name}')
    for name, expected in manifest['runtime_source_sha256'].items():
        if digest(project/name)!=expected: raise ValueError(f'Wrong historical runtime: {name}')
    for name, expected in manifest['scene_config_sha256'].items():
        if digest(project/name)!=expected: raise ValueError(f'Changed scene/sensor configuration: {name}')
    if digest(bundle/'policy_2000.zip') != '7a3dad60f76060daf6b311b4bc6db921d3925ef0cdcf11683fe99e282173de11':
        raise ValueError('Wrong selected 2000-step checkpoint')
    torch.set_num_threads(1)
    model=MaskablePPO.load(bundle/'policy_2000.zip',device='cpu')
    actor=torch.jit.load(str(bundle/'portable/actor.pt'),map_location='cpu').eval()
    if model.observation_space.shape!=(300,) or model.action_space.n!=12:
        raise ValueError('Expected 300-field, 12-action native model')
    recorded=np.load(bundle/'reports/parity_inputs.npz',allow_pickle=False)
    obs=recorded['observations']; choices=recorded['choices']; masks=recorded['masks']
    if obs.shape!=(9924,300) or not np.isfinite(obs).all():
        raise ValueError('Expected 9924 actual finite historical inputs')
    np.testing.assert_array_equal(masks_from_observation(obs[:,:282]),masks)
    for start in range(0,len(obs),256):
        stop=start+256; batch=obs[start:stop]
        indices=model.predict(batch,action_masks=masks[start:stop],deterministic=True)[0]
        np.testing.assert_array_equal(indices,choices[start:stop])
        with torch.no_grad(): actual=actor(torch.from_numpy(batch)).numpy()
        np.testing.assert_array_equal(actual,ACTIONS[indices])
    if not masks[np.arange(len(obs)),choices].all(): raise ValueError('Recorded masked action')
    moving=(choices!=0)&~recorded['goal_stops']
    np.testing.assert_allclose(recorded['commands'][moving],COMMANDS[choices[moving]],rtol=0,atol=2e-8)
    results=json.loads((bundle/'reports/historical_cases_results.json').read_text())
    summary=dict(tasks=len(results),safe_success=sum(r['result']['safe_success'] for r in results),
                 collisions=sum(r['result']['collisions'] for r in results),
                 timeouts=sum(r['result']['timeout'] for r in results))
    if summary != dict(tasks=48,safe_success=46,collisions=1,timeouts=1):
        raise ValueError('Historical evaluation totals changed')
    return dict(verified=True,recorded_frames=len(obs),checkpoint_and_export_parity=True,
                selected_executed_nonzero_parity=True,package_hashes_verified=True,
                historical_runtime_files_verified=len(manifest['runtime_source_sha256']),
                historical_summary=summary,full48_rerun=False,deployment_approved=False)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--bundle',type=Path,default=Path('models/quality_recovery_20261014'))
    args=p.parse_args()
    print(json.dumps(verify(args.bundle,Path(__file__).resolve().parents[1]),indent=2))


if __name__=='__main__': main()
