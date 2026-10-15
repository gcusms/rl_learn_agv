"""Check feasible alternatives after a verified physical recovery prefix.

These declared interventions initialize successful demonstrations only if the
entire continuation is physically safe. They are never runtime rules, model
scores or permission to bypass an inadmissible action.
"""
import argparse
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

import experiments.evaluate_feedback_recovery as recovery
from experiments.probe_feedback_alternatives import ProbeActor, intervention_windows
from neural_navigation.decision_feedback_runtime import StopFeedbackRuntime
from neural_navigation.experiment_budget import ExperimentBudget


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['bundle', 'prefix', 'specifications', 'output']:
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--cpu-cores', type=float, default=.15)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    budget = ExperimentBudget(args.cpu_cores, directory=args.output)
    prefix = json.loads(args.prefix.read_text())
    specifications = json.loads(args.specifications.read_text())
    names = set()
    for spec in specifications:
        name = spec['id']
        if not name or Path(name).name != name or name in names:
            raise ValueError('Unique simple probe ids required')
        names.add(name)
        for window in intervention_windows(spec):
            if window['start_s'] >= 0 and window['start_s'] < prefix['release_s']-1e-8:
                raise ValueError('A probe must not change the initialization prefix')
    from experiments.experiment_provenance import record_provenance
    record_provenance(args.output, args.bundle)
    (args.output/'prefix.json').write_bytes(args.prefix.read_bytes())
    (args.output/'specifications.json').write_bytes(args.specifications.read_bytes())
    report = dict(completed=False, not_a_trained_policy=True, deployment_approved=False,
        prefix_specification_sha256=hashlib.sha256(args.prefix.read_bytes()).hexdigest(),
        bundle_manifest_sha256=hashlib.sha256((args.bundle/'manifest.json').read_bytes()).hexdigest(),
        rows=[], limitations=['Timed interventions are diagnostic, not learned actions or deployment rules.',
            'A legal choice is not necessarily useful or safe throughout its continuation.',
            'Only complete safe trajectories may later supply demonstration examples.'])
    for spec in specifications:
        created = []

        class Runtime(StopFeedbackRuntime):
            def __init__(self, *a, **kw):
                super().__init__(*a, **kw)
                actor = ProbeActor(self.policy.policy, self.core, spec)
                self.policy.policy = actor
                created.append(actor)

        # The existing evaluator puts its checked SensorPrefixActor outside
        # this probe, retaining all its sensor, command and physical checks.
        with patch.object(recovery, 'StopFeedbackRuntime', Runtime):
            row, trace = recovery.evaluate_recovery(prefix, args.bundle, budget)
        if len(created) != 1:
            raise ValueError('Expected one explicitly instrumented runtime')
        path = args.output/(spec['id']+'.npz')
        np.savez_compressed(path, **trace)
        row.update(specification=spec, intervention=created[0].entries,
            trace_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), not_a_trained_policy=True)
        report['rows'].append(row)
        report['resource'] = budget.statistics()
        (args.output/'RESULTS.json').write_text(json.dumps(report, indent=2)+'\n')
        print(json.dumps({k: v for k, v in row.items() if k != 'intervention'}), flush=True)
    report['completed'] = True
    (args.output/'RESULTS.json').write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
