"""Frozen actor, changed experimental localizer; targeted sensor-only diagnostics."""
import argparse
import json
from pathlib import Path
import shutil
import time
import torch
from experiments.train_quality_navigation import digest
from experiments.evaluate_quality_regression import evaluate


def main():
    parser=argparse.ArgumentParser()
    for name in ('parent-bundle','localizer','training-trace','regression-run','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():parser.error('Fresh output required')
    torch.set_num_threads(1);args.output.mkdir();bundle=args.output/'portable';bundle.mkdir()
    original=json.loads((args.parent_bundle/'manifest.json').read_text())
    for name,expected in original['files'].items():
        if digest(args.parent_bundle/name)!=expected:raise ValueError('Original bundle changed')
        shutil.copy2(args.parent_bundle/name,bundle/name)
    shutil.copy2(args.localizer,bundle/'localizer.pt')
    manifest=dict(original,localizer_changed=True,parent_bundle_manifest_sha256=digest(args.parent_bundle/'manifest.json'),
        parent_localizer_sha256=original['files']['localizer.pt'],localizer_source_sha256=digest(args.localizer),
        deployment_approved=False,files={p.name:digest(p) for p in bundle.iterdir() if p.is_file()})
    (bundle/'manifest.json').write_text(json.dumps(manifest,indent=2))
    training=json.loads(args.training_trace.read_text())['case']
    history=dict(id='reused_training_recovery_10458',category='reused_training_recovery',scene=training['scene'],
        seed=training['seed'],**training['options'],**training['interventions'])
    cases=[history]+[json.loads((args.regression_run/(cid+'.json')).read_text())['case'] for cid in ('independent_dynamic_013','motion_profile_015')]
    results=[];report=dict(completed=False,cases=results,bundle_manifest=manifest,
        training_trace_sha256=digest(args.training_trace),started_unix=time.time(),deployment_approved=False,
        limitation='Three targeted development tasks including a reused training task. Frozen actor with changed localization observations; not full regression, covariance calibration or real hardware proof.')
    def save():(args.output/'RESULTS.json').write_text(json.dumps(report,indent=2))
    save()
    try:
        for case in cases:
            trace=evaluate(case,bundle);path=args.output/(case['id']+'.json');path.write_text(json.dumps(trace,allow_nan=False))
            results.append(dict(trace['result'],trace_sha256=digest(path)));report['completed']=len(results)==3;save();print(json.dumps(trace['result']),flush=True)
        report['finished_unix']=time.time();save()
    except Exception as error:report.update(error=str(error),failed=True);save();raise


if __name__=='__main__':main()
