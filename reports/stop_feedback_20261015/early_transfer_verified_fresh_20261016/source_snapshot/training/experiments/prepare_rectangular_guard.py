"""Create an isolated frozen-actor geometry experiment; no default promotion."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
from neural_navigation.safety_context import validate_contract,contract,BOUNDED_PREDICTIVE_SCHEMA


def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--manifest',type=Path,action='append',required=True)
    parser.add_argument('--case',action='append',required=True)
    parser.add_argument('--sigma-scale',type=float,default=0.)
    parser.add_argument('--force-response',action='store_true')
    parser.add_argument('--bounded-pose-risk',action='store_true')
    args=parser.parse_args()
    if args.output.exists() or not 0<=args.sigma_scale<=3:
        parser.error('Fresh output and uncertainty scale 0..3 required')
    source=args.source.resolve(); output=args.output.resolve()
    original=json.loads((source/'manifest.json').read_text())
    for name,checksum in original['files'].items():
        if digest(source/name)!=checksum: raise ValueError('Source bundle hash mismatch')
    hardware=json.loads((source/'hardware.json').read_text())
    validate_contract(hardware['policy_observation'])
    cases={}; manifests={}
    for path in args.manifest:
        data=json.loads(path.read_text())
        if data.get('kind') not in ('development','training_failure_replay'):
            raise ValueError('Development cases only; keep acceptance sealed')
        manifests[str(path.resolve())]=digest(path)
        for case in data['cases']:
            if case['id'] in cases and case!=cases[case['id']]:
                raise ValueError('Conflicting frozen case definitions')
            cases[case['id']]=case
    if len(set(args.case))!=len(args.case) or any(k not in cases for k in args.case):
        raise ValueError('Unique existing case IDs required')
    output.mkdir(parents=True); bundle=output/'portable';shutil.copytree(source,bundle)
    memory=dict(hardware.get('near_field_memory',{}))
    memory.update(static_footprint_schema='oriented_rectangle_v1',
        static_footprint_margin_m=.04, static_footprint_pose_sigma_scale=args.sigma_scale)
    if args.force_response:
        memory.update(motion_response_schema='force_response_v1',response_gain_s_inv=20.,
            response_linear_acceleration_m_s2=.5,response_angular_acceleration_rad_s2=1.)
    hardware['near_field_memory']=memory
    if args.bounded_pose_risk:
        hardware['policy_observation']=contract(BOUNDED_PREDICTIVE_SCHEMA)
    (bundle/'hardware.json').write_text(json.dumps(hardware,indent=2))
    exported=dict(original); exported['files']=dict(original['files'])
    exported['files']['hardware.json']=digest(bundle/'hardware.json')
    exported['deployment_approved']=False
    exported['experimental_guard']='oriented_rectangle_v1'
    if args.force_response:exported['experimental_motion_response']='force_response_v1'
    if args.bounded_pose_risk:
        exported['policy_observation']=hardware['policy_observation']
        exported['experimental_risk']='bounded_historical_pose_support'
    (bundle/'manifest.json').write_text(json.dumps(exported,indent=2))
    selected=dict(kind='development',cases=[cases[k] for k in args.case],
        source_manifests=manifests,purpose='Frozen-actor static geometry diagnostic, not final acceptance')
    (output/'cases.json').write_text(json.dumps(selected,indent=2))
    project=Path(__file__).resolve().parents[2]; snapshots={}
    for path in sorted((project/'training').rglob('*.py')):
        relative=path.relative_to(project)
        if 'runs' in relative.parts: continue
        snapshots[str(relative)]=digest(path)
        target=output/'code_snapshot'/relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target)
    config=dict(source_bundle=str(source),actor_unchanged=True,
        actor_sha256=digest(bundle/'actor.pt'),policy_sha256=original['policy_sha256'],
        guard=memory,localization_uncertainty_calibrated=False,
        force_response_parameters_calibrated=False,
        nominal_geometry_only=args.sigma_scale==0.,deployment_approved=False,
        speed_limits_unchanged=True,laser_memory_and_rear_guard_unchanged=True,
        launch_source_sha256=snapshots)
    (output/'config.json').write_text(json.dumps(config,indent=2))
    print(json.dumps(dict(bundle=str(bundle),cases=str(output/'cases.json'),tasks=len(args.case))))


if __name__=='__main__': main()
