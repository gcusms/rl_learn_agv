"""Check causal stop feedback on saved real PPO samples, not an invented state.

The actual executed reason becomes the next input. Timer values are checked
against sensor time and the previous action's choices, across rollouts/reset.
Learning statistics are descriptive and do not replace closed-loop evaluation.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from neural_navigation.decision_feedback import (
    FEEDBACK_REASONS, WAIT_SCHEMA, REASON_SLICE, STAT_SLICE, contract)


def audit(paths):
    rows = []
    last_reason = 'none'
    idle_start = None
    reset = True
    previous_case = None
    previous_time = None
    maximum_error = 0.
    frame_index = 0
    first_schema = None
    for path in sorted(paths):
        with np.load(path, allow_pickle=False) as data:
            meta = json.loads(str(data['metadata_json']))
            schema = meta['feedback_schema']; contract(schema)
            if first_schema is not None and schema != first_schema:
                raise ValueError('Do not combine different feedback semantics in one audit')
            first_schema = schema
            observations = data['observations']
            if observations.shape[1] != 1:
                raise ValueError('This sequential audit requires one environment')
            obs = observations[:, 0]
            choices = data['sampled_actions'][:, 0, 0].astype(int)
            masks = data['action_masks'][:, 0].astype(bool)
            advantages = data['advantages'][:, 0]
            labels = [json.loads(s) for s in data['execution_diagnostics_json'][:, 0]]
            starts = data['episode_starts'][:, 0].astype(bool)
            for i, label in enumerate(labels):
                timestamp = float(label['sensor_timestamp'])
                if bool(starts[i]) != reset:
                    raise ValueError('Episode reset does not follow the actual preceding termination')
                if starts[i]:
                    reset = True; last_reason = 'none'; idle_start = None
                    previous_case = None; previous_time = None
                if not reset and (label['case_json'] != previous_case or timestamp <= previous_time):
                    raise ValueError('Missing reset or non-monotonic episode time')
                reason_values = obs[i, 300+REASON_SLICE.start:300+REASON_SLICE.stop]
                reason_index = FEEDBACK_REASONS.index(last_reason)
                expected = np.zeros(len(FEEDBACK_REASONS)); expected[reason_index] = 1.
                if not np.array_equal(reason_values, expected):
                    raise ValueError('Input reason leaks current execution or crosses a reset')
                expected_timer = 0. if idle_start is None else min((timestamp-idle_start)/5., 1.)
                actual_timer = float(obs[i, 300+STAT_SLICE.start+3])
                error = abs(actual_timer-expected_timer)
                maximum_error = max(maximum_error, error)
                if error > 2e-6:
                    raise ValueError('Waiting timer differs from actual sensor-time history')
                idle = bool(np.max(abs(np.asarray(label['executed_command']))) < 1e-8)
                available = bool(masks[i, 1:].any())
                case = json.loads(label['case_json'])
                row = dict(frame=frame_index, rollout=path.name, sensor_timestamp=timestamp,
                    case_id=case.get('source_training_case_id', 'general_curriculum'),
                    sampled_choice=int(choices[i]), actual_idle=idle, legal_motion_choices=int(masks[i, 1:].sum()),
                    input_previous_reason=last_reason, executed_reason=label['executed_reason'],
                    choice_wait_timer_s=actual_timer*5., raw_advantage=float(advantages[i]),
                    true_geodesic_progress_m=label['true_geodesic_progress_m'],
                    native_reward=label['native_reward'], episode_start=bool(starts[i]), done=bool(label['done']))
                rows.append(row); frame_index += 1
                reason = label['executed_reason']
                if reason not in FEEDBACK_REASONS:
                    raise ValueError('Uncontracted executed reason')
                extending = idle and reason == 'policy_wait'
                if schema == WAIT_SCHEMA:
                    extending = idle and available and reason != 'estimated_goal_reached'
                idle_start = (timestamp if idle_start is None else idle_start) if extending else None
                last_reason = reason
                reset = bool(label['done'])
                previous_time = timestamp; previous_case = label['case_json']
    if not rows:
        raise ValueError('No complete rollout files')
    if not rows[0]['episode_start']:
        raise ValueError('Provide rollouts from an episode reset for causal verification')
    waits = [r for r in rows if r['actual_idle'] and r['legal_motion_choices'] > 0]
    mislabeled = [r for r in waits if r['executed_reason'] not in ('policy_wait', 'estimated_goal_reached')]
    return dict(completed=True, feedback_schema=first_schema, rollout_files=len(paths), sampled_frames=len(rows),
        input_reasons_causally_verified=len(rows), timer_frames_verified=len(rows),
        maximum_timer_normalized_error=maximum_error, only_stop_frames=sum(r['legal_motion_choices']==0 for r in rows),
        idle_with_motion_choices=len(waits),
        idle_with_choices_but_guard_label_frames=len(mislabeled),
        longest_observed_choice_wait_s=max(r['choice_wait_timer_s'] for r in rows),
        positive_wait_timer_with_guard_reason=sum(r['choice_wait_timer_s']>0 and
            r['input_previous_reason'] not in ('none','policy_wait','only_stop_available','estimated_goal_reached') for r in rows),
        long_wait_samples=[r for r in rows if r['choice_wait_timer_s']>=2.-1e-7],
        deployment_approved=False,
        limitations=['Timer saturates at 5 s; a positive timer does not imply movement is useful.',
                     'A raw rollout advantage is not a normalized per-minibatch PPO actor gradient.',
                     'No collision/success claim is inferred from reason visibility or timer coverage.'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    result = audit(list((args.model_run/'rollouts').glob('*.npz')))
    status = json.loads((args.model_run/'status.json').read_text())
    result['training_complete_at_audit'] = status['status']=='complete'
    result['declared_training_steps'] = status.get('additional_ppo_steps',status.get('target_steps'))
    (args.output/'RESULTS.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='long_wait_samples'},indent=2))


if __name__=='__main__': main()
