"""Replay the training deadlock detector against saved sensor-runtime frames.

Truth and actor fields are deliberately unused. Does not terminate replay,
choose actions, update weights, or add development cases to training.
"""
import argparse
import json
from pathlib import Path
from navrl.geometry import ROOT
from navrl.map import StaticMap
from experiments.static_deadlock_curriculum import StaticDeadlockDetector


def audit(trace):
    detector = StaticDeadlockDetector()
    case = trace['case']
    static_map = StaticMap(ROOT/case.get('scene', 'assets/room_navigation_scripted')/'decision_map.yaml', radius=.3)
    previous_time = None
    first = None
    for row in trace['samples']:
        now = row['timestamp']
        dt = .2 if previous_time is None else now-previous_time
        previous_time = now
        if row.get('policy_observation') is None:
            raise ValueError('Exact actor observations required for detector audit')
        hit = detector.step(row['policy_observation'], row['estimate'],
            bool(static_map.lookup(row['estimate'][:2], static_map.free, outside=False)),
            row['stop_reason'], row['diagnostics'], dt)
        if hit and first is None:
            first = now
    return dict(case_id=case['id'], first_static_deadlock_s=first,
                original_safe_success=trace['result']['safe_success'])


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--traces', nargs='+', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        p.error('Fresh audit output required')
    for directory in args.traces:
        if not directory.is_dir() or not list(directory.glob('*.json')):
            p.error(f'Missing or empty traces: {directory}')
    cases = [audit(json.loads(path.read_text())) for directory in args.traces
             for path in sorted(directory.glob('*.json'))]
    result = dict(executions=len(cases), unique_case_ids=len({r['case_id'] for r in cases}),
                  cases=cases, limitation='Read-only same-frame sensor detector audit; the training wrapper '
                  'uses pre-action geometry and post-transition localization health. Neither this audit nor '
                  'early training termination establishes a successful recovery or improved trained model.')
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps([r for r in cases if r['first_static_deadlock_s'] is not None]))


if __name__ == '__main__':
    main()
