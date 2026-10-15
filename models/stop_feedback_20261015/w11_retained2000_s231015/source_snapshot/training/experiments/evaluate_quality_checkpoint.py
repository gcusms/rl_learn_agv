"""Frozen intermediate quality actor on a recorded training case, no learning."""
import argparse
import json
from pathlib import Path
import time
import torch
from experiments.train_quality_navigation import export,evaluate,digest
from experiments.audit_quality_learning import audit


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--failure-pool',type=Path,required=True)
    parser.add_argument('--training-log',type=Path,required=True);parser.add_argument('--replay-step',type=int,required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    if args.output.exists():parser.error('Fresh output required')
    pool=json.loads(args.failure_pool.read_text())
    if pool['kind']!='training':parser.error('Training-only case templates required')
    records=[json.loads(x) for x in args.training_log.read_text().splitlines()]
    matches=[r for r in records if r.get('step')==args.replay_step and 'worker' in r and 'training_case_json' in r]
    if len(matches)!=1:parser.error('Unique recorded training task required')
    case=json.loads(matches[0]['training_case_json']);before=digest(args.checkpoint)
    if case['stage']!='robust':parser.error('Robust training task required')
    torch.set_num_threads(1);args.output.mkdir()
    status={'status':'running','started_unix':time.time(),'policy_sha256':before}
    save=lambda:(args.output/'status.json').write_text(json.dumps(status,indent=2))
    save()
    try:
        export(args.checkpoint,args.checkpoint.parent,args.output/'portable')
        result=evaluate(args.checkpoint,args.checkpoint.parent,pool['templates'],case,args.output/'history.json')
        sensitivity=audit(args.checkpoint,args.output/'history.json')
        if digest(args.checkpoint)!=before:raise ValueError('Frozen checkpoint changed')
        (args.output/'SENSITIVITY.json').write_text(json.dumps(sensitivity,indent=2))
        status.update(status='complete',result=result,finished_unix=time.time(),training_case_reused=True,deployment_approved=False)
        save();print(json.dumps(result))
    except Exception as error:
        status.update(status='failed',error=str(error));save();raise


if __name__=='__main__':main()
