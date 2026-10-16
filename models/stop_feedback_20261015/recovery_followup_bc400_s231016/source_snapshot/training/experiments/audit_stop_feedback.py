"""Audit exact recorded actor feedback; read-only, no inferred observations."""
import argparse
import json
from pathlib import Path
import numpy as np
from neural_navigation.safety_context import BASE_DIM, OBSERVATION_DIM, PREDICTIVE_DIM, REASONS, REASON_START


def audit(trace):
    counts = dict(captured=0, missing=0, feedback_mismatch=0,
                  feasible_idle=0, behind_waypoint_with_forward=0,
                  blocked_without_probe=0)
    mismatches = []
    previous = None
    first_all_rejected = None
    for row in trace['samples']:
        raw = row.get('policy_observation')
        if raw is None:
            counts['missing'] += 1
            previous = row
            continue
        obs = np.asarray(raw, float)
        if obs.shape not in ((OBSERVATION_DIM,), (PREDICTIVE_DIM,)) or not np.isfinite(obs).all():
            raise ValueError('Expected exact finite context actor input')
        counts['captured'] += 1
        context = obs[BASE_DIM:OBSERVATION_DIM]
        expected_reason = 'none' if previous is None else previous.get('stop_reason') or 'none'
        if expected_reason not in REASONS:
            expected_reason = 'other'
        expected_action = np.zeros(2)
        if previous is not None:
            v, w = previous['command']
            expected_action = np.clip([v / (.2 if v >= 0 else .08), w / .5], -1, 1)
        reason_vector = np.zeros(len(REASONS))
        reason_vector[REASONS.index(expected_reason)] = 1
        matches = (np.allclose(context[12:14], expected_action, atol=2e-6, rtol=0)
                   and np.allclose(context[REASON_START:REASON_START+len(REASONS)],
                                   reason_vector, atol=2e-6, rtol=0))
        if not matches:
            counts['feedback_mismatch'] += 1
            mismatches.append(row['timestamp'])
        flags = context[:9] > .5
        forward = bool(flags[[0, 1, 2, 5, 6, 7, 8]].any())
        v, w = row['command']
        relevant = row.get('stop_reason') != 'estimated_goal_reached'
        counts['feasible_idle'] += int(relevant and forward and abs(v) < .01 and abs(w) < .05)
        counts['behind_waypoint_with_forward'] += int(relevant and forward and obs[216] < -.025)
        if relevant and not flags.any() and abs(v) < .01 and abs(w) < .05:
            counts['blocked_without_probe'] += 1
            if first_all_rejected is None:
                first_all_rejected = dict(timestamp=row['timestamp'], reason=row.get('stop_reason'),
                                          estimate=row.get('estimate'))
        previous = row
    return dict(case_id=trace['case'].get('id', f"training_step_{trace.get('training_step', 'unknown')}"), counts=counts,
                mismatch_timestamps=mismatches, first_all_rejected=first_all_rejected)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--traces', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Use a fresh output file')
    for directory in args.traces:
        if not directory.is_dir() or not list(directory.glob('*.json')):
            parser.error(f'Missing or empty trace directory: {directory}')
    results = [audit(json.loads(path.read_text())) for directory in args.traces
               for path in sorted(directory.glob('*.json'))]
    args.output.write_text(json.dumps(dict(cases=results, executions=len(results),
        unique_case_ids=len({r['case_id'] for r in results}),
        limitation='Frame counts are not independent episodes. Behind-target direction is a diagnostic, '
                   'not proof of a bad route. Nine rejected probes do not exhaust continuous actions. '
                   'Missing actor inputs cannot establish feedback correctness.'), indent=2))
    print(json.dumps({key: sum(r['counts'][key] for r in results) for key in results[0]['counts']}))


if __name__ == '__main__':
    main()
