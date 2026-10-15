"""Preserve directional navigation weights; opt into ray-verified tracking.

Outputs are an experimental bundle, not an approved deployment. Measured
motion is a separate, explicitly enabled input adapter, not an actor feature.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
from neural_navigation.verified_static_localizer import contract
from neural_navigation.measured_odometry import contract as motion_contract


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build(source,output):
    source=Path(source);output=Path(output)
    original=json.loads((source/'manifest.json').read_text())
    for name,expected in original['files'].items():
        if Path(name).name!=name or digest(source/name)!=expected:raise ValueError('Invalid parent component: '+name)
    output.mkdir(parents=True,exist_ok=False)
    files={}
    for name in original['files']:
        if name in ('localization_kernel.so','localization_contract.json'):continue
        shutil.copy2(source/name,output/name);files[name]=digest(output/name)
    config=output/'localization_contract.json';config.write_text(json.dumps(contract(),indent=2)+'\n')
    files[config.name]=digest(config)
    manifest=dict(original,files=files,deployment_approved=False,
        localization_method=contract()['schema'],localization_is_neural=False,
        parent_manifest_sha256=digest(source/'manifest.json'))
    manifest.pop('native_kernel',None)
    manifest['motion_input_adapter']=dict(optional=True,automatic=False,
        entrypoint='neural_navigation.measured_odometry.MeasuredNavigationAdapter',contract=motion_contract())
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    for name in ('actor.pt','router.pt','hardware.json','goal_pose_observation.json','docking_contract.json'):
        if (source/name).exists() and digest(source/name)!=digest(output/name):raise ValueError('Preserved component mismatch')
    runtime=Path(__file__).parents[1]/'neural_navigation'
    sources=['fast_static_localizer.py','verified_static_localizer.py','measured_odometry.py','core.py']
    card=dict(experimental=True,deployment_approved=False,actor_trained=False,
        parent=str(source.resolve()),actor_router_actions_heading_contract_preserved=True,
        localization_uses_native_kernel=False,motion_input_changed_automatically=False,
        known_map_and_initial_pose_required=True,global_relocalization=False,online_slam=False,
        runtime_sources_sha256={name:digest(runtime/name) for name in sources},
        limitations=['Fixed-input replay does not establish closed-loop navigation preservation.',
            'No independent real pose truth or target-board performance measurement.',
            'Measured wheel/gyro adapter requires explicit integration; existing ROS EKF may instead select encoder source.'],
        manifest_sha256=digest(output/'manifest.json'))
    (output/'MODEL_CARD.json').write_text(json.dumps(card,indent=2)+'\n')
    return card


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();print(json.dumps(build(args.source,args.output),indent=2))
