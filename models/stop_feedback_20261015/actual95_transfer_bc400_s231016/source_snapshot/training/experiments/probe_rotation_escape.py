"""Frozen actor with opt-in clearance-increasing pure rotation, five failures."""
import argparse
import json
from pathlib import Path
import shutil
import time
import torch
import experiments.evaluate_quality_regression as evaluator
from experiments.time_consistent_tracking import TimeConsistentFreshRuntime
from experiments.train_quality_navigation import digest


def main():
    parser=argparse.ArgumentParser()
    for name in ('bundle','reference-run','output'):parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():parser.error('Fresh output required')
    torch.set_num_threads(1);args.output.mkdir()
    bundle=args.output/'portable';shutil.copytree(args.bundle,bundle)
    hardware=json.loads((bundle/'hardware.json').read_text())
    old=hardware['near_field_memory'].get('clearance_increasing_rotation',False)
    if old:raise ValueError('Expected a baseline with rotation escape disabled')
    hardware['near_field_memory']['clearance_increasing_rotation']=True
    (bundle/'hardware.json').write_text(json.dumps(hardware,indent=2))
    manifest=json.loads((bundle/'manifest.json').read_text())
    manifest['files']={name:digest(bundle/name) for name in manifest['files']}
    manifest['diagnostic_change']='Enable existing sensor-checked clearance-increasing pure rotation; no actor or physical-limit changes'
    manifest['deployment_approved']=False
    (bundle/'manifest.json').write_text(json.dumps(manifest,indent=2))
    if digest(bundle/'actor.pt')!=digest(args.bundle/'actor.pt'):raise ValueError('Frozen actor changed')
    project=Path(__file__).resolve().parents[2];sources={}
    for source in (project/'training').rglob('*.py'):
        relative=source.relative_to(project)
        if 'runs' in relative.parts:continue
        target=args.output/'source_snapshot'/relative;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(source,target);sources[str(relative)]=digest(source)
    (args.output/'SOURCE_PROVENANCE.json').write_text(json.dumps(dict(source_sha256=sources,
        captured_before_evaluation=True,source_bundle_manifest_sha256=digest(args.bundle/'manifest.json')),indent=2))
    evaluator.QualitySensorNavigationRuntime=TimeConsistentFreshRuntime
    ids=('independent_dynamic_006','independent_dynamic_021','independent_dynamic_023',
         'motion_profile_005','train_failure_w11_e7')
    report=dict(completed=False,cases=[],started_unix=time.time(),deployment_approved=False,
        actor_sha256=digest(bundle/'actor.pt'),bundle_manifest_sha256=digest(bundle/'manifest.json'),
        limitation='Five repeatedly used development failures, frozen actor, changed safety-context feasibility. No manual action replacement; not broad acceptance or real hardware validation.')
    def save():(args.output/'RESULTS.json').write_text(json.dumps(report,indent=2))
    save()
    try:
        for cid in ids:
            case=json.loads((args.reference_run/(cid+'.json')).read_text())['case']
            trace=evaluator.evaluate(case,bundle);path=args.output/(cid+'.json')
            path.write_text(json.dumps(trace,allow_nan=False))
            report['cases'].append(dict(trace['result'],trace_sha256=digest(path)))
            report['completed']=len(report['cases'])==len(ids);save()
            print(json.dumps(trace['result']),flush=True)
        report['finished_unix']=time.time();save()
    except Exception as error:
        report.update(failed=True,error=str(error));save();raise


if __name__=='__main__':main()
