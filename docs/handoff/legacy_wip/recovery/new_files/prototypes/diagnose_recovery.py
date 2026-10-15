"""Paired development evaluation with sensor-only routing/guard diagnostics.

No action substitution, mask relaxation or default model replacement. Recorded
truth remains exclusively in the evaluator's scoring, not these diagnostics.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import time

import numpy as np
import torch
from sb3_contrib import MaskablePPO

import experiments.evaluate_quality_regression as evaluation
from experiments.time_consistent_tracking import TimeConsistentFreshRuntime
from neural_navigation.masked_actions import COMMANDS
from prototypes.continued_recovery_policy import continued_support


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--policy', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--case', action='append')
    parser.add_argument('--efficient-routes', action='store_true')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Fresh output required')
    args.output.mkdir(parents=True)
    torch.set_num_threads(1)
    policy = MaskablePPO.load(args.policy, device='cpu').policy
    report = dict(status='running', started_unix=time.time(), efficient_routes=args.efficient_routes,
                  cases=[], deployment_approved=False)
    def save():
        (args.output/'RESULTS.json').write_text(json.dumps(report, indent=2))
    save()
    diagnostics = []

    class ObservedRuntime(TimeConsistentFreshRuntime):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            self.core.efficient_route_selection = args.efficient_routes
            self.frame = 0

        def step(self, *a, **kw):
            result = super().step(*a, **kw)
            if self.frame in (35, 60, 100, 200, 300, 500, 650):
                core = self.core
                memory = self.guard.memory
                entries = list(memory.entries.values()) if memory else []
                diagnostics.append(dict(frame=self.frame, timestamp=core.elapsed,
                    route_complete=core.diagnostics()['route_complete'],
                    route_probability=core.current_probability, path_points=len(core.path),
                    route_end=core.path[-1].tolist(), goal=core.goal.tolist(),
                    route_choice=core.replan_reason, route_candidates=getattr(core,'route_candidate_count',1),
                    efficiency_active=getattr(core,'route_efficiency_active',False),
                    efficiency_gate=getattr(core,'route_efficiency_gate_reason',None),
                    start_in_hard=bool(core.laser_fusion.hard[core.cell(core.estimate[:2])]),
                    map_agreement=float(core.localizer.map_agreement),
                    pose_std=core.localizer.std.tolist(), memory_points=len(entries),
                    memory_older_than_2s=sum(core.elapsed-x[1]>2 for x in entries),
                    unmatched_returns=core.unmatched_returns,
                    tracks=len(core.laser_fusion.predictor.tracks),
                    reason=result['stop_reason']))
            self.frame += 1
            return result

    evaluation.QualitySensorNavigationRuntime = ObservedRuntime
    try:
        paths = [args.reference/f'{name}.json' for name in args.case] if args.case else [
            args.reference/f'{name}.json' for name in (
                'independent_dynamic_006','independent_dynamic_021','independent_dynamic_023',
                'motion_profile_005','train_failure_w11_e7')]
        for path in paths:
            case = json.loads(path.read_text())['case']
            diagnostics.clear()
            trace = evaluation.evaluate(case, args.bundle)
            observations = torch.tensor([r['policy_observation'] for r in trace['samples']], dtype=torch.float32)
            masks = np.asarray([r['mask'] for r in trace['samples']])
            wait = np.asarray([r['choice']==0 and any(r['mask'][1:]) for r in trace['samples']])
            with torch.no_grad():
                probabilities = policy.get_distribution(observations, action_masks=masks).distribution.probs.numpy()
                support = continued_support(observations).numpy()
            # Check that the checkpoint being diagnosed matches the actor driving.
            expected = probabilities.argmax(-1)
            actual = np.asarray([r['choice'] for r in trace['samples']])
            np.testing.assert_array_equal(expected, actual)
            trace['diagnostics'] = list(diagnostics)
            trace['wait_audit'] = dict(wait_frames=int(wait.sum()),
                recovery_supported_wait_frames=int(support[wait].sum()),
                prediction_supported_wait_frames=int((observations.numpy()[wait,294]>.5).sum()),
                mean_nonzero_probability_when_waiting=float((1-probabilities[wait,0]).mean()) if wait.any() else None,
                stop_reasons=dict(Counter(r['stop_reason'] for r in trace['samples'] if r['choice']==0)))
            trace['action_probabilities'] = probabilities.tolist()
            (args.output/f"{case['id']}.json").write_text(json.dumps(trace, allow_nan=False))
            report['cases'].append(dict(trace['result'], wait_audit=trace['wait_audit']))
            save()
            print(json.dumps(report['cases'][-1]), flush=True)
        report.update(status='complete', finished_unix=time.time())
    except BaseException as error:
        report.update(status='failed', error=repr(error))
        raise
    finally:
        save()


if __name__ == '__main__':
    main()
