"""Package the evaluated adapter candidate; never replace published artifacts."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path
import numpy as np
import torch
from sb3_contrib import MaskablePPO
from prototypes.narrow_observation import contract,masks
from prototypes.narrow_training_env import NarrowTrainingEnv
from prototypes.narrow_fine_runtime import FineSensorRuntime
from prototypes.train_narrow_long import CASES


def main():
    p=argparse.ArgumentParser();p.add_argument('--evaluation',type=Path,required=True);args=p.parse_args()
    status=json.loads((args.evaluation/'status.json').read_text())
    if status['status']!='complete' or not status['source_hashes_unchanged'] or not status['published_actor_unchanged']:
        p.error('Completed evaluation with unchanged source and published actor required')
    original=Path('models/recovery_20261012/portable');portable=args.evaluation/'portable'
    if portable.exists():p.error('Fresh package required')
    torch.set_num_threads(1);shutil.copytree(original,portable)
    shutil.copy2(args.evaluation/'actor_adapter_untrained.pt',portable/'actor.pt')
    (portable/'fine_observation_action_contract.json').write_text(json.dumps(contract(),indent=2))
    manifest=json.loads((portable/'manifest.json').read_text())
    manifest.update(fine_interface=contract(),deployment_approved=False,additional_training_steps=0,
        requires_narrow_sensor_runtime=True,required_runtime='prototypes.narrow_fine_runtime.FineSensorRuntime',
        candidate_kind='adapter_repair_without_new_training',runtime_source_sha256=status['source_sha256'])
    for name in ['actor.pt','fine_observation_action_contract.json']:
        manifest['files'][name]=hashlib.sha256((portable/name).read_bytes()).hexdigest()
    (portable/'manifest.json').write_text(json.dumps(manifest,indent=2))
    case=CASES[0];scene=Path('training/runs/narrow_long_50000_v2_20261013/scenes/gap_150')
    env=NarrowTrainingEnv(scene,original,seed=case['seed'],max_steps=250)
    try:
        env.reset(options=dict(start=case['start'],goal=case['goal'],case_seed=case['seed']))
        runtime=FineSensorRuntime(portable,scene/'decision_map.yaml',case['start'],case['goal'],'shortcut')
        result=runtime.step(env.env.scan,env.env.observed_odometry,env.env.measured_velocity,
            env.env.elapsed,capture_observation=True)
        obs=np.asarray(result['policy_observation'],np.float32)
        model=MaskablePPO.load(args.evaluation/'policy_adapter_untrained.zip',device='cpu')
        choice,_=model.predict(obs,action_masks=masks(obs),deterministic=True)
        if int(choice)!=result['fine_choice']:raise RuntimeError('Packaged runtime and checkpoint disagree')
        checks=dict(packaged_runtime_checkpoint_parity=True,choice=int(choice),
            original_published_actor_unchanged=hashlib.sha256((original/'actor.pt').read_bytes()).hexdigest()==status['published_actor_sha256'],
            additional_training_steps=0,deployment_approved=False)
        (args.evaluation/'package_checks.json').write_text(json.dumps(checks,indent=2));print(json.dumps(checks))
    finally:env.close()


if __name__=='__main__':main()
