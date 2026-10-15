"""Collect complete checked training examples with a declared offline turn.

These are supervised training probes, never neural-policy validation scores.
Only complete physically safe episodes are eligible for a later explicit
data preparation step; all collisions and timeouts remain in the report.
"""
import argparse
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from experiments.evaluate_docking_navigation import evaluate
from experiments.experiment_provenance import record_provenance
from experiments.probe_feedback_alternatives import ProbeActor
from neural_navigation.decision_feedback_runtime import StopFeedbackRuntime
from neural_navigation.experiment_budget import ExperimentBudget


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['bundle','cases','output']:
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--cpu-cores',type=float,default=.35)
    args = parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1)
    budget = ExperimentBudget(args.cpu_cores,memory_mib=3072,directory=args.output)
    cases = [c for c in json.loads(args.cases.read_text())
        if c['id'].startswith('train_failure_proactive_variation_')]
    if len(cases)!=8 or len({c['seed'] for c in cases})!=8:
        raise ValueError('The eight pre-frozen training conditions are required')
    window = dict(start_s=6.,end_s=7.4,choice=5)
    (args.output/'cases.json').write_text(json.dumps(cases,indent=2)+'\n')
    record_provenance(args.output,args.bundle)
    report = dict(completed=False,rows=[],training_examples=True,
        not_a_trained_policy=True,deployment_approved=False,
        case_source_sha256=hashlib.sha256(args.cases.read_bytes()).hexdigest(),
        declaration=window,
        scope='Offline training probes using only currently legal intervention commands. Not policy scores or unseen-map evaluation.',
        reference_after_window='Original stop-only BC recovery model; original body, guard, prediction and sensor conditions.',
        failed_episodes_retained=True,automatic_training_data_promotion=False)
    for case in cases:
        created = []

        class Runtime(StopFeedbackRuntime):
            def __init__(self,*a,**kw):
                super().__init__(*a,**kw)
                actor = ProbeActor(self.policy.policy,self.core,window)
                self.policy.policy = actor
                created.append(actor)

        with patch('neural_navigation.goal_pose_runtime.resolve_runtime',return_value=Runtime):
            row,trace = evaluate(case,args.bundle,budget)
        if len(created)!=1:
            raise ValueError('Expected one explicit offline intervention runtime')
        path = args.output/(case['id']+'.npz')
        np.savez_compressed(path,**trace)
        row.update(specification=dict(id=case['id'],**window),
            intervention=created[0].entries,
            trace_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            not_a_trained_policy=True)
        report['rows'].append(row)
        report['resource']=budget.statistics()
        (args.output/'RESULTS.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps({k:v for k,v in row.items() if k!='intervention'}),flush=True)
    report.update(completed=True,safely_completed_examples=sum(r['safe_success'] for r in report['rows']))
    (args.output/'RESULTS.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__ == '__main__':
    main()
