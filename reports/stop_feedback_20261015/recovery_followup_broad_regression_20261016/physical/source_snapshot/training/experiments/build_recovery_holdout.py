"""Freeze distinct recovery initializations; the learner never sees this data.

The declared brief turn initializes a case. Only commands accepted by the
unchanged guard may be used. The reference then controls the entire remainder.
Every safely reached release state is retained, including no-choice states;
failed initializations are reported explicitly, never counted as policy wins.
"""
import argparse
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from experiments.evaluate_docking_navigation import evaluate
from experiments.probe_feedback_alternatives import ProbeActor
from neural_navigation.decision_feedback_runtime import StopFeedbackRuntime
from neural_navigation.experiment_budget import ExperimentBudget


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['bundle', 'cases', 'output']:
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--cpu-cores', type=float, default=.15)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    budget = ExperimentBudget(args.cpu_cores, memory_mib=3072, directory=args.output)
    cases = json.loads(args.cases.read_text())
    (args.output/'cases.json').write_bytes(args.cases.read_bytes())
    from experiments.experiment_provenance import record_provenance
    record_provenance(args.output, args.bundle)
    initialization = dict(id='early_rotation', start_s=6., end_s=7.4, choice=5)
    release = 20.
    report = dict(completed=False, rows=[], release_s=release,
        case_specifications_sha256=hashlib.sha256(args.cases.read_bytes()).hexdigest(),
        initialization=initialization, training_examples=False, deployment_approved=False,
        scope='New seeds, poses and sensor conditions in the same known map; recovery continuation, not unseen-map generalization.')
    specifications = []
    for case in cases:
        created = []

        class Runtime(StopFeedbackRuntime):
            def __init__(self, *a, **kw):
                super().__init__(*a, **kw)
                actor = ProbeActor(self.policy.policy, self.core, initialization)
                self.policy.policy = actor
                created.append(actor)

        with patch('neural_navigation.goal_pose_runtime.resolve_runtime', return_value=Runtime):
            row, trace = evaluate(case, args.bundle, budget)
        if len(created) != 1:
            raise ValueError('Expected exactly one declared initialization runtime')
        path = args.output/(case['id']+'.npz')
        np.savez_compressed(path, **trace)
        releases = np.flatnonzero(abs(trace['labels'][:, 0]-release) < 1e-8)
        reached = len(releases) == 1
        row.update(variant='reference_continuation', trace_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            initialization_reached_release=reached, initialization=initialization,
            intervention=created[0].entries, not_a_full_mission_score=True)
        if reached:
            index = int(releases[0])
            row.update(release_motion_choices=int(trace['masks'][index, 1:].sum()),
                reference_release_choice=int(trace['choices'][index]))
            specifications.append(dict(id=case['id'], case=case, release_s=release,
                trace=str(path)))
        report['rows'].append(row)
        report['resource'] = budget.statistics()
        (args.output/'RESULTS.json').write_text(json.dumps(report, indent=2)+'\n')
        (args.output/'prefixes.json').write_text(json.dumps(specifications, indent=2)+'\n')
        print(json.dumps({k: v for k, v in row.items() if k != 'intervention'}), flush=True)
    report.update(completed=True, initialized_cases=len(specifications),
        failed_initializations=len(cases)-len(specifications),
        initialized_long_wait_cases=sum(r['initialization_reached_release']
            and r['reference_release_choice'] == 0 and r['release_motion_choices'] > 0 for r in report['rows']))
    (args.output/'RESULTS.json').write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
