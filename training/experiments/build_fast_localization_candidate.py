"""Copy a verified bundle and opt into fast tracking; never overwrite a release."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
from neural_navigation.fast_static_localizer import contract,native_contract,sparse_contract,temporal_contract


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build(source, output, native=False, sparse=False, temporal=False, uncertainty_budget=False):
    source=Path(source);output=Path(output)
    if output.exists():raise FileExistsError(output)
    if sparse and temporal:raise ValueError('Choose one experimental contract')
    if uncertainty_budget and not temporal:raise ValueError('Use the temporal contract for this uncertainty experiment')
    original=json.loads((source/'manifest.json').read_text())
    for name,expected in original['files'].items():
        if Path(name).name!=name or digest(source/name)!=expected:raise ValueError('Invalid source bundle: '+name)
    output.mkdir(parents=True)
    for name in original['files']:shutil.copy2(source/name,output/name)
    configuration=temporal_contract(native) if temporal else (sparse_contract(native) if sparse else (native_contract() if native else contract()))
    (output/'localization_contract.json').write_text(json.dumps(configuration,indent=2)+'\n')
    manifest=dict(original,deployment_approved=False,localization_method=configuration['schema'],
        localization_is_neural=False,requires_supplied_initial_map_pose=True,
        parent_manifest_sha256=digest(source/'manifest.json'))
    manifest['files']=dict(original['files'],**{'localization_contract.json':digest(output/'localization_contract.json')})
    if native:
        import subprocess,platform
        kernel=Path(__file__).parents[1]/'neural_navigation/static_tracking_kernel.cpp'
        subprocess.run(['g++','-O3','-DNDEBUG','-std=c++17','-shared','-fPIC',str(kernel),'-I/usr/include/eigen3','-o',str(output/'localization_kernel.so')],check=True)
        manifest['files']['localization_kernel.so']=digest(output/'localization_kernel.so')
        manifest['native_kernel']=dict(architecture=platform.machine(),source_sha256=digest(kernel),
            build='g++ -O3 -DNDEBUG -std=c++17 -shared -fPIC -I/usr/include/eigen3',runtime_compilation=False)
    if uncertainty_budget:
        from neural_navigation.weak_geometry_uncertainty import contract as uncertainty_contract
        (output/'uncertainty_contract.json').write_text(json.dumps(uncertainty_contract(),indent=2)+'\n')
        manifest['files']['uncertainty_contract.json']=digest(output/'uncertainty_contract.json')
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    for name in ('actor.pt','router.pt','hardware.json','goal_pose_observation.json','docking_contract.json'):
        if (source/name).exists() and digest(source/name)!=digest(output/name):raise ValueError('Preserved component mismatch')
    card=dict(experimental=True,deployment_approved=False,parent=str(source.resolve()),
        method='geometric robust static-map tracking, no particle or neural proposal search',
        actor_trained=False,actor_router_hardware_byte_identical=True,
        routing_acceptance_changed=False,uncertainty_budget_enabled=bool(uncertainty_budget),
        limitations=['Known map and initial pose required; no global relocalization or SLAM backend replacement.',
            'Covariance is provisional, not real-world calibration.',
            'Changed localization inputs can change policy decisions; closed-loop regression is required.'],
        manifest_sha256=digest(output/'manifest.json'))
    (output/'MODEL_CARD.json').write_text(json.dumps(card,indent=2)+'\n');return card


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--native',action='store_true')
    p.add_argument('--sparse',action='store_true',help='V2 sparse-support tracking with count-scaled information')
    p.add_argument('--temporal',action='store_true',help='Sparse support over fixed physical-time offsets')
    p.add_argument('--uncertainty-budget',action='store_true',help='Uncalibrated weak-geometry budget; does not change the pose')
    a=p.parse_args();print(json.dumps(build(a.source,a.output,a.native,a.sparse,a.temporal,a.uncertainty_budget),indent=2))
