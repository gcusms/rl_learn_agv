"""Export discrete masked PPO as sensor-runtime normalized (vx,wz) choices."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np
import torch
from sb3_contrib import MaskablePPO
from experiments.masked_navigation import PortableMaskedActor
from neural_navigation.masked_actions import ACTIONS,action_contract,masks_from_observation
from neural_navigation.safety_context import PREDICTIVE_DIM,PREDICTIVE_SCHEMAS,validate_contract


def export_actor(policy,components,output):
    if output.exists(): raise ValueError('Fresh masked export directory required')
    model=MaskablePPO.load(policy,device='cpu')
    if model.observation_space.shape!=(PREDICTIVE_DIM,) or model.action_space.n!=len(ACTIONS):
        raise ValueError('Masked V4 velocity actor required')
    hardware=json.loads((components/'hardware.json').read_text())
    validate_contract(hardware['policy_observation'])
    if hardware['policy_observation']['schema'] not in PREDICTIVE_SCHEMAS:
        raise ValueError('V4 probe semantics required')
    if hardware.get('policy_action')!=action_contract():
        raise ValueError('Explicit categorical velocity contract required')
    observations=np.random.default_rng(92).uniform(-1,1,(256,PREDICTIVE_DIM)).astype(np.float32)
    observations[:,235:244]=np.random.default_rng(93).integers(0,2,(256,9))
    observations[:,267:269]=np.random.default_rng(94).integers(0,2,(256,2))
    observations[0,235:244]=0;observations[0,267:269]=0
    masks=masks_from_observation(observations)
    indices=model.predict(observations,action_masks=masks,deterministic=True)[0]
    actor=PortableMaskedActor(model.policy).eval()
    portable=torch.jit.freeze(torch.jit.trace(actor,torch.zeros(1,PREDICTIVE_DIM)))
    with torch.no_grad():actual=portable(torch.from_numpy(observations)).numpy()
    np.testing.assert_array_equal(actual,ACTIONS[indices])
    if not masks[np.arange(len(indices)),indices].all():raise ValueError('Masked selection failed')
    output.mkdir(parents=True);portable.save(str(output/'actor.pt'))
    for name in ('localizer.pt','router.pt'):shutil.copy2(components/name,output/name)
    (output/'hardware.json').write_text(json.dumps(hardware,indent=2))
    digest=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
    manifest=dict(policy_source=str(policy.resolve()),policy_sha256=digest(policy),
        observation_dimensions=PREDICTIVE_DIM,action_dimensions=2,categorical_choices=len(ACTIONS),
        precision='float32',device='cpu',comparison_inputs=len(observations),max_action_error=0.,
        deployment_approved=False,board_tested=False,action_contract=action_contract(),
        files={name:digest(output/name) for name in ('actor.pt','localizer.pt','router.pt','hardware.json')})
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2))
    return manifest


def main():
    parser=argparse.ArgumentParser()
    for name in ('policy','components','output'):parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();torch.set_num_threads(1)
    print(json.dumps(export_actor(args.policy,args.components,args.output),indent=2))


if __name__=='__main__':main()
