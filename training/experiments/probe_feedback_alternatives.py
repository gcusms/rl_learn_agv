"""Offline feasibility probes, never a learned-policy score or deployment mode.

Temporarily substitute one currently legal action during a declared sensor-time
window, then let the archived policy continue. The ordinary guard, executed
action history, sensors, robot body and obstacle trajectory remain in use.
Fixed intervention times are diagnostic interventions, not policy inputs.
"""
import argparse
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from experiments.evaluate_docking_navigation import evaluate
from neural_navigation.decision_feedback_runtime import StopFeedbackRuntime
from neural_navigation.experiment_budget import ExperimentBudget
from neural_navigation.masked_actions import ACTIONS, masks_from_observation


def intervention_windows(specification):
    """Validate disjoint diagnostic windows; never relax an action mask."""
    windows = specification.get('windows', [specification])
    if not isinstance(windows, list) or not 1 <= len(windows) <= 16:
        raise ValueError('Provide one to sixteen diagnostic time windows')
    result = []
    for window in windows:
        choice = window.get('choice')
        try:
            start, end = float(window['start_s']), float(window['end_s'])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError('Finite ordered time window required') from error
        if (type(choice) is not int or not 0 <= choice < 12
                or not np.isfinite([start, end]).all() or end <= start):
            raise ValueError('Finite ordered time window and existing action required')
        result.append(dict(start_s=start, end_s=end, choice=choice))
    result.sort(key=lambda window: window['start_s'])
    if any(a['end_s'] > b['start_s'] for a, b in zip(result, result[1:])):
        raise ValueError('Diagnostic intervention windows must not overlap')
    return result


class ProbeActor:
    def __init__(self, actor, core, specification):
        self.actor, self.core, self.specification = actor, core, specification
        self.windows = intervention_windows(specification)
        self.entries = []

    def predict(self, observation):
        reference = self.actor.predict(observation)
        now = float(self.core.elapsed)
        for index, window in enumerate(self.windows):
            start, end, choice = (window[k] for k in ('start_s', 'end_s', 'choice'))
            if start - 1e-8 <= now < end - 1e-8:
                mask = masks_from_observation(np.asarray(observation)[:282])
                accepted = bool(mask[choice])
                result = ACTIONS[choice].copy() if accepted else reference
                self.entries.append(dict(sensor_timestamp=now, requested_choice=int(choice),
                    window_index=index, requested_choice_legal=accepted, replaced_reference=accepted,
                    reference_action=np.asarray(reference).tolist(), proposed_action=np.asarray(result).tolist()))
                return result
        return reference


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('bundle', 'cases', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--case-id', default='motion_profile_015')
    parser.add_argument('--specifications', type=Path, help='Optional explicit list of time-window interventions')
    parser.add_argument('--cpu-cores', type=float, default=.15)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    budget = ExperimentBudget(args.cpu_cores, memory_mib=3072, directory=args.output)
    case = next(c for c in json.loads(args.cases.read_text()) if c['id'] == args.case_id)
    specifications = [dict(id='unchanged_policy', start_s=-2., end_s=-1., choice=0)]
    for name, start, end, choice in [
        ('early_wait', 7.4, 9.4, 0), ('slow_straight', 7.6, 9., 3),
        ('continue_straight', 7.8, 8.6, 1), ('rotate_right', 7.8, 8.6, 5),
        ('rotate_left', 7.6, 8.2, 4)]:
        specifications.append(dict(id=name, start_s=start, end_s=end, choice=choice))
    if args.specifications:
        specifications=json.loads(args.specifications.read_text())
    ids=set()
    for specification in specifications:
        name=specification['id']
        if name in ids or Path(name).name!=name or not name:
            raise ValueError('Unique simple intervention names required')
        ids.add(name)
        intervention_windows(specification)
    (args.output / 'case.json').write_text(json.dumps(case, indent=2) + '\n')
    (args.output / 'specifications.json').write_text(json.dumps(specifications,indent=2)+'\n')
    report = dict(completed=False, not_a_trained_policy=True, deployment_approved=False,
        case_id=case['id'], bundle_manifest_sha256=hashlib.sha256((args.bundle/'manifest.json').read_bytes()).hexdigest(),
        purpose='Check whether alternative currently legal actions can avoid an archived failure.',
        limitations=['Fixed sensor-time intervention is not a learned decision or generalization.',
                     'No change to the shared guard, body, input observations or action history.',
                     'Successful probes are not PPO success rates and are not automatically used for training.'], rows=[])
    for specification in specifications:
        created = []

        class ProbeRuntime(StopFeedbackRuntime):
            def __init__(self, *positional, **kwargs):
                super().__init__(*positional, **kwargs)
                actor = ProbeActor(self.policy.policy, self.core, specification)
                self.policy.policy = actor
                created.append(actor)

        with patch('neural_navigation.goal_pose_runtime.resolve_runtime', return_value=ProbeRuntime):
            row, trace = evaluate(case, args.bundle, budget)
        if len(created) != 1:
            raise RuntimeError('Expected one explicitly instrumented runtime')
        path = args.output / (specification['id'] + '.npz')
        np.savez_compressed(path, **trace)
        row.update(specification=specification, intervention=created[0].entries,
                   trace_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                   not_a_trained_policy=True)
        report['rows'].append(row)
        report['resource'] = budget.statistics()
        (args.output / 'RESULTS.json').write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps({k: v for k, v in row.items() if k != 'intervention'}), flush=True)
    report['completed'] = True
    (args.output / 'RESULTS.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__': main()
