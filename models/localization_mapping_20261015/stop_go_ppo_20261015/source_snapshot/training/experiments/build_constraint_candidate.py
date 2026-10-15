"""Create a separately hashed predictive-mask experiment without editing a release."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
from neural_navigation.goal_pose_runtime import check_files
from neural_navigation.predictive_constraint import contract,cumulative_contract,registered_contract
from neural_navigation.masked_actions import action_contract


def build(source,output,mode='predictive',cumulative=True,registered=False,registration_residual=False):
    if mode not in ('predictive','uncertainty','combined'):raise ValueError('Unknown constraint experiment')
    source=Path(source);output=Path(output)
    if output.exists():raise FileExistsError(output)
    manifest=json.loads((source/'manifest.json').read_text());check_files(source,manifest)
    output.mkdir(parents=True)
    for name in manifest['files']:shutil.copy2(source/name,output/name)
    hardware=json.loads((output/'hardware.json').read_text())
    if mode in ('predictive','combined'):
        hardware.setdefault('near_field_memory',{})['predictive_action_constraint']=(registered_contract(registration_residual) if registered else cumulative_contract() if cumulative else contract())
        hardware['policy_action']=action_contract('registered_residual' if registration_residual else 'registered' if registered else 'cumulative' if cumulative else True)
    (output/'hardware.json').write_text(json.dumps(hardware,indent=2)+'\n')
    manifest.update(deployment_approved=False,action_contract=hardware['policy_action'],
        parent_manifest_sha256=hashlib.sha256((source/'manifest.json').read_bytes()).hexdigest())
    manifest['files']['hardware.json']=hashlib.sha256((output/'hardware.json').read_bytes()).hexdigest()
    if mode in ('uncertainty','combined'):
        from neural_navigation.weak_geometry_uncertainty import contract as uncertainty_contract
        if not json.loads((source/'localization_contract.json').read_text())['schema'].startswith('fast_static_tracking_'):
            raise ValueError('Weak geometry budget requires the geometric localizer')
        (output/'uncertainty_contract.json').write_text(json.dumps(uncertainty_contract(),indent=2)+'\n')
        manifest['files']['uncertainty_contract.json']=hashlib.sha256((output/'uncertainty_contract.json').read_bytes()).hexdigest()
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');check_files(output,manifest)
    for name in ('actor.pt','router.pt','localizer.pt','localization_contract.json','docking_contract.json'):
        if (source/name).exists():assert (source/name).read_bytes()==(output/name).read_bytes()
    (output/'MODEL_CARD.json').write_text(json.dumps(dict(deployment_approved=False,
        method=mode,actor_trained=False,
        actor_router_localizer_byte_identical=True,parent=str(source.resolve()),
        future_safety_certified=False,uncertainty_calibrated=False),indent=2)+'\n')
    return output


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--mode',choices=['predictive','uncertainty','combined'],default='predictive')
    p.add_argument('--legacy-contour',action='store_true')
    p.add_argument('--registered-motion',action='store_true')
    p.add_argument('--registration-residual',action='store_true')
    a=p.parse_args()
    if a.registration_residual and not a.registered_motion:p.error('Registration residual requires registered motion')
    print(build(a.source,a.output,a.mode,not a.legacy_contour,a.registered_motion,a.registration_residual))
