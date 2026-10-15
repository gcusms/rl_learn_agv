"""Separate real guard stops from ineffective proposals in saved runtime traces.

Display/diagnostic only. Never modifies a policy, guard or robot command.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import numpy as np
from neural_navigation.safety_context import OBSERVATION_DIM, PREDICTIVE_DIM

FORWARD_PROBES = [0, 1, 2, 5, 6, 7, 8]
TURN_PROBES = [3, 4]


def longest_span(samples, mask):
    if not samples:
        return 0.
    times = np.array([s['timestamp'] for s in samples])
    dt = float(np.median(np.diff(times))) if len(times) > 1 else .2
    start = None
    longest = 0.
    for index, active in enumerate(mask):
        if active:
            if start is None: start = times[index]
            longest = max(longest, float(times[index]-start+dt))
        else: start = None
    return longest


def analyze(trace):
    rows = trace['samples']
    counts = Counter()
    reasons = Counter()
    masks = {k: [] for k in ['stopped_with_forward_probe', 'idle_with_forward_probe',
                            'idle_with_only_turn_probe', 'no_verified_probe',
                            'rejected_reverse_with_forward_probe']}
    for row in rows:
        context = row.get('safety_context', {})
        reason = row.get('stop_reason') or 'none'
        reasons[reason] += 1
        proposal = np.asarray(row['model_command'])
        executed = np.asarray(row['command'])
        projected = not np.allclose(proposal, executed, atol=1e-8, rtol=0)
        counts['partial_projection_steps'] += int(projected and np.linalg.norm(executed) > 1e-8)
        counts['zero_command_steps'] += int(np.linalg.norm(executed) < 1e-8)
        flags = context.get('action_feasibility')
        masks_now = dict.fromkeys(masks, False)
        if flags is not None:
            flags = np.asarray(flags)
            if flags.shape != (9,): raise ValueError('Expected the nine-probe context prefix')
            fast = np.zeros(2)
            raw = row.get('policy_observation')
            if raw is not None and len(raw) == PREDICTIVE_DIM:
                raw = np.asarray(raw, float)
                if not np.isfinite(raw).all(): raise ValueError('Nonfinite V4 actor observation')
                fast = raw[OBSERVATION_DIM:OBSERVATION_DIM+2]
            counts['context_steps'] += 1
            # Reached-goal zero command is intentional, not a stalled policy.
            relevant = reason != 'estimated_goal_reached'
            forward = bool(np.any(flags[FORWARD_PROBES]) or np.any(fast > .5))
            turn = bool(np.any(flags[TURN_PROBES]))
            idle = abs(executed[0]) < .01 and abs(executed[1]) < .05
            counts['forward_probe_available_steps'] += int(relevant and forward)
            counts['fast_only_forward_probe_steps'] += int(relevant and np.any(fast > .5)
                and not np.any(flags[FORWARD_PROBES]))
            counts['only_turn_probe_available_steps'] += int(relevant and not forward and turn)
            masks_now.update(
                stopped_with_forward_probe=relevant and forward and np.linalg.norm(executed) < 1e-8,
                idle_with_forward_probe=relevant and forward and idle,
                idle_with_only_turn_probe=relevant and not forward and turn and idle,
                rejected_reverse_with_forward_probe=(relevant and forward and
                    proposal[0] < 0 and abs(executed[0]) < .01 and
                    reason in ('reverse_without_rear_visibility', 'reverse_duration_limit')),
                no_verified_probe=relevant and not bool(np.any(flags) or np.any(fast > .5)))
            previous_reason = context['policy_previous_stop_reason']
            counts['known_previous_guard_reason_steps'] += int(previous_reason not in ('none', 'other', 'estimated_goal_reached'))
        for name in masks:
            masks[name].append(masks_now[name])
    context_available = counts['context_steps'] > 0
    return dict(case_id=trace['case']['id'], result=trace['result'], steps=len(rows),
                reason_counts=dict(reasons), partial_projection_steps=counts['partial_projection_steps'],
                context_steps=counts['context_steps'],
                classifications={name: dict(steps=int(sum(mask)), longest_seconds=longest_span(rows, mask))
                                 for name, mask in masks.items()} if context_available else None,
                context_counters=dict(counts) if context_available else None,
                thresholds=dict(idle_translation_m_s=.01, idle_rotation_rad_s=.05),
                limitation='Finite measured probe set, not proof that all other actions are unsafe. '
                           'V4 also counts the two fast-turn acceptance inputs; legacy traces use nine probes. '
                           'Small commands count as ineffective here even if nonzero; no claim of real motor response.')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--traces', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists(): p.error('Fresh diagnostic output required')
    results = [analyze(json.loads(path.read_text())) for path in sorted(a.traces.glob('*.json'))]
    a.output.write_text(json.dumps(dict(cases=results), indent=2))
    for result in results:
        print(result['case_id'], json.dumps(result['classifications']))


if __name__ == '__main__': main()
