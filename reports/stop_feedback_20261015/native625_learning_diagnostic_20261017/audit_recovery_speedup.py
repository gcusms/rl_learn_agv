"""Locate the first policy-action difference in the actual96s paired recovery."""
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / 'reports/stop_feedback_20261015'
HERE = Path(__file__).resolve().parent


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    output = HERE / 'RECOVERY_SPEEDUP_CAUSAL_AUDIT.json'
    assert not output.exists()
    parent_report = BASE / 'normal_suite_wait_validation_v2_20261016/RESULTS.json'
    candidate_report = BASE / 'native625_batch_credit_validation_20261017/RESULTS.json'
    p, q = [json.loads(path.read_text()) for path in [parent_report, candidate_report]]
    assert p['completed'] and q['completed']
    before = next(r for r in p['groups']['hard3']['rows'] if r['case_id'] == 'legal_wait_parent_t90')
    after = next(r for r in q['groups']['hard3']['rows'] if r['case_id'] == 'legal_wait_parent_t90')
    assert before['safe_success'] and after['safe_success']
    paths = [ROOT / r['trace_path'] for r in [before, after]]
    assert all(digest(path) == row['trace_sha256'] for path, row in zip(paths, [before, after]))
    models = [ROOT / row['variant'] / 'portable' for row in [before, after]]
    hardware_files = ['hardware.json', 'localizer.pt', 'router.pt', 'decision_feedback.json',
        'release_observation.json', 'docking_contract.json', 'localization_contract.json']
    unchanged = {name: digest(models[0] / name) == digest(models[1] / name) for name in hardware_files}
    assert all(unchanged.values())
    with np.load(paths[0], allow_pickle=False) as old, np.load(paths[1], allow_pickle=False) as new:
        count = min(len(old['choices']), len(new['choices']))
        first = int(np.flatnonzero(old['choices'][:count] != new['choices'][:count])[0])
        prefix = {key: bool(np.array_equal(old[key][:first+1], new[key][:first+1], equal_nan=True))
            for key in ['observations', 'masks', 'scans', 'odometry', 'measured_velocity', 'labels', 'human_positions']}
        prefix['choices_before_difference'] = bool(np.array_equal(old['choices'][:first], new['choices'][:first]))
        prefix['commands_before_difference'] = bool(np.array_equal(old['commands'][:first], new['commands'][:first]))
        assert all(prefix.values())
        assert old['masks'][first, old['choices'][first]] and new['masks'][first, new['choices'][first]]
        n = before['frozen_prefix_frames']
        assert n == after['frozen_prefix_frames'] == 480
        def metrics(data, row):
            masks, choices = data['masks'][n:], data['choices'][n:]
            return dict(post_release_seconds=row['post_release_seconds'],
                total_travel_including_common_prefix_m=row['travel_m'],
                post_release_only_stop_frames=int((~masks[:, 1:].any(1)).sum()),
                post_release_wait_with_choices_frames=int(((choices == 0) & masks[:, 1:].any(1)).sum()),
                collisions=row['collisions'], false_arrival=row['false_arrival'], physical_guards=row['physical_guards'])
        pair = dict(parent=metrics(old, before), candidate=metrics(new, after))
        difference = dict(frame=first, sensor_time_s=float(old['labels'][first, 0]),
            parent_choice=int(old['choices'][first]), candidate_choice=int(new['choices'][first]),
            parent_command_m_s_rad_s=old['commands'][first].tolist(),
            candidate_command_m_s_rad_s=new['commands'][first].tolist(),
            legal_actions=np.flatnonzero(old['masks'][first]).tolist(),
            observation_and_physical_state_identical=True)
    files = [Path(__file__), parent_report, candidate_report, *paths]
    files += [folder / name for folder in models for name in hardware_files + ['actor.pt', 'manifest.json']]
    result = dict(completed=True, same_actual_release_s=before['release_sensor_timestamp_s'],
        prefix_parity=prefix, unchanged_components=unchanged, first_difference=difference,
        paired_recovery_metrics=pair,
        recovery_time_relative_change=after['post_release_seconds']/before['post_release_seconds']-1.,
        policy_weight_change_not_sensor_or_safety_contract_change=True,
        interpretation='The new actor turns in place where the old actor advances while turning; later stop-only frames fall from90 to5.',
        one_reused_development_state_not_independent_generalization=True,
        remaining_two_hard_states_unsolved=True, no_new_physics=True, no_optimizer_updates=True,
        new_ppo_steps=0, goal_achieved=False, deployment_approved=False,
        inputs={str(path.relative_to(ROOT)): digest(path) for path in files})
    with output.open('x') as handle:
        json.dump(result, handle, indent=2)
        handle.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k != 'inputs'}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
