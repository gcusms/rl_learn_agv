"""Three targeted frozen-actor tests of per-frame perception, no promotion."""
import argparse
import json
from pathlib import Path
import time
import torch
import experiments.evaluate_quality_regression as evaluator
from experiments.fresh_perception_runtime import FreshPerceptionRuntime
from experiments.train_quality_navigation import digest


def main():
    parser=argparse.ArgumentParser()
    for name in ('bundle','training-trace','regression-run','output'):parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():parser.error('Fresh output required')
    torch.set_num_threads(1);args.output.mkdir()
    # Select the experimental runtime explicitly inside this isolated process;
    # retain the same scenario, score, actor/mask and stop-feedback protocol.
    evaluator.QualitySensorNavigationRuntime=FreshPerceptionRuntime
    training=json.loads(args.training_trace.read_text())['case']
    history=dict(id='reused_training_recovery_10458',category='reused_training_recovery',scene=training['scene'],
        seed=training['seed'],**training['options'],**training['interventions'])
    cases=[history]+[json.loads((args.regression_run/(cid+'.json')).read_text())['case'] for cid in ('motion_profile_015','independent_dynamic_013')]
    results=[];report=dict(completed=False,cases=results,started_unix=time.time(),
        bundle_manifest_sha256=digest(args.bundle/'manifest.json'),runtime_variant='per_sensor_frame_fusion',
        deployment_approved=False,limitation='Three targeted development cases, one reused training case. Changed perception also changes actor inputs and dynamic map; no action substitution, full regression or real hardware proof.')
    def save():(args.output/'RESULTS.json').write_text(json.dumps(report,indent=2))
    save()
    try:
        for case in cases:
            trace=evaluator.evaluate(case,args.bundle);path=args.output/(case['id']+'.json');path.write_text(json.dumps(trace,allow_nan=False))
            results.append(dict(trace['result'],trace_sha256=digest(path)));report['completed']=len(results)==3;save();print(json.dumps(trace['result']),flush=True)
        report['finished_unix']=time.time();save()
    except Exception as error:report.update(error=str(error),failed=True);save();raise


if __name__=='__main__':main()
