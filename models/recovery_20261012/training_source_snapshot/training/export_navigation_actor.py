"""Export a deterministic PPO actor for the simulator-independent runtime."""
import argparse
import copy
import hashlib
import json
import shutil
from pathlib import Path
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import FlattenExtractor
from navrl.geometry import LidarProfile,ROOT
from neural_navigation.safety_context import OBSERVATION_DIM,PREDICTIVE_DIM,validate_contract


class Actor(torch.nn.Module):
    def __init__(self,policy):
        super().__init__()
        if not isinstance(policy.features_extractor,FlattenExtractor) or policy.squash_output:
            raise ValueError('Export currently supports nonsquashed flat MLP actors only')
        self.net=copy.deepcopy(policy.mlp_extractor.policy_net)
        self.action=copy.deepcopy(policy.action_net)
    def forward(self,observation):
        return torch.clamp(self.action(self.net(observation.flatten(1))),-1.,1.)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--policy',type=Path,required=True)
    parser.add_argument('--components',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():parser.error('Use a fresh export directory')
    torch.set_num_threads(1);model=PPO.load(args.policy,device='cpu')
    dimensions=model.observation_space.shape[0]
    if dimensions not in (235,OBSERVATION_DIM,PREDICTIVE_DIM) or model.action_space.shape!=(2,):raise ValueError('Incompatible actor interface')
    component_hardware=json.loads((args.components/'hardware.json').read_text()) if (args.components/'hardware.json').exists() else {}
    if dimensions in (OBSERVATION_DIM,PREDICTIVE_DIM):
        validate_contract(component_hardware.get('policy_observation'))
        if component_hardware['policy_observation']['dimensions']!=dimensions:
            raise ValueError('Actor and observation contract dimensions differ')
    elif 'policy_observation' in component_hardware:
        raise ValueError('Legacy actor cannot use safety-context hardware contract')
    actor=Actor(model.policy).eval();dummy=torch.zeros(1,dimensions)
    traced=torch.jit.freeze(torch.jit.trace(actor,dummy))
    observations=np.random.default_rng(1).uniform(-1,1,(128,dimensions)).astype(np.float32)
    observations[0]=0.;observations[1]=1.;observations[2]=-1.
    reference=model.predict(observations,deterministic=True)[0]
    with torch.no_grad():exported=traced(torch.from_numpy(observations)).numpy()
    np.testing.assert_allclose(exported,reference,atol=2e-6,rtol=2e-6)
    args.output.mkdir(parents=True)
    traced.save(str(args.output/'actor.pt'))
    for name in ['localizer.pt','router.pt']:shutil.copy2(args.components/name,args.output/name)
    scene=json.loads((ROOT/'assets/room_navigation_scripted/physical_scene.json').read_text())
    hardware=dict(lidar=LidarProfile().signature,robot_footprint_m=scene['robot_collision_box_m'][:2],
                  control_period_s=.2,forward_limit_m_s=.2,reverse_limit_m_s=.08,yaw_limit_rad_s=.5,
                  real_robot_calibrated=False)
    if (args.components/'hardware.json').exists():
        previous=json.loads((args.components/'hardware.json').read_text())
        if previous['lidar']!=hardware['lidar'] or not np.allclose(previous['robot_footprint_m'],hardware['robot_footprint_m'],atol=1e-9,rtol=0):
            raise ValueError('Component sensor/footprint contract differs from the trained profile')
        if 'near_field_memory' in previous:hardware['near_field_memory']=previous['near_field_memory']
        if dimensions in (OBSERVATION_DIM,PREDICTIVE_DIM):hardware['policy_observation']=previous['policy_observation']
    (args.output/'hardware.json').write_text(json.dumps(hardware,indent=2))
    manifest=dict(policy_source=str(args.policy.resolve()),policy_sha256=hashlib.sha256(args.policy.read_bytes()).hexdigest(),
                  observation_dimensions=dimensions,action_dimensions=2,precision='float32',device='cpu',
                  comparison_inputs=len(observations),max_action_error=float(np.max(np.abs(reference-exported))),
                  deployment_approved=False,board_tested=False,
                  files={name:hashlib.sha256((args.output/name).read_bytes()).hexdigest() for name in ['actor.pt','localizer.pt','router.pt','hardware.json']})
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps(manifest,indent=2))


if __name__=='__main__':main()
