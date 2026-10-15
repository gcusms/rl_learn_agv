"""Four paired-development cases isolating risk tracking from routing updates."""
import argparse
import json
from pathlib import Path
import time
import torch
import experiments.evaluate_quality_regression as evaluator
from experiments.isolated_frame_prediction import IsolatedFramePredictionRuntime
from experiments.train_quality_navigation import digest


def main():
    parser=argparse.ArgumentParser()
    for name in ('bundle','training-trace','regression-run','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():parser.error('Fresh output required')
    torch.set_num_threads(1);args.output.mkdir()
    evaluator.QualitySensorNavigationRuntime=IsolatedFramePredictionRuntime
    training=json.loads(args.training_trace.read_text())['case']
    history=dict(id='reused_training_recovery_10458',category='reused_training_recovery',scene=training['scene'],
                 seed=training['seed'],**training['options'],**training['interventions'])
    cases=[history]+[json.loads((args.regression_run/(cid+'.json')).read_text())['case']
                    for cid in ('motion_profile_015','independent_dynamic_013','train_failure_w11_e7')]
    report=dict(completed=False,cases=[],started_unix=time.time(),
                bundle_manifest_sha256=digest(args.bundle/'manifest.json'),deployment_approved=False,
                runtime_variant=getattr(IsolatedFramePredictionRuntime,'experiment_variant','isolated_sensor_frame_risk_prediction'),
                runtime_class=IsolatedFramePredictionRuntime.__module__+'.'+IsolatedFramePredictionRuntime.__name__,
                limitation='Four repeatedly used development cases, frozen actor, explicitly selected experimental sensor runtime. Not final acceptance or proof of semantic understanding.')
    def save():(args.output/'RESULTS.json').write_text(json.dumps(report,indent=2))
    save()
    try:
        for case in cases:
            trace=evaluator.evaluate(case,args.bundle)
            path=args.output/(case['id']+'.json');path.write_text(json.dumps(trace,allow_nan=False))
            report['cases'].append(dict(trace['result'],trace_sha256=digest(path)))
            report['completed']=len(report['cases'])==len(cases);save()
            print(json.dumps(trace['result']),flush=True)
        report['finished_unix']=time.time();save()
    except Exception as error:
        report.update(failed=True,error=str(error));save();raise


if __name__=='__main__':main()
