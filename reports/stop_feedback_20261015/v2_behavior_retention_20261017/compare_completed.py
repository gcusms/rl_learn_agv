"""Compare only actually completed matched cases; keep recovery scope explicit.

This is a JSON-only read of saved physical outcomes. It does not execute a
policy, consume prospective seeds, or treat all voluntary waits as mistakes.
"""
from datetime import datetime
import hashlib
import json
from pathlib import Path
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
BASE = HERE.parent


def read(path): return json.loads(path.read_text())
def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def flatten(report):
    rows = {}
    for group, block in report['groups'].items():
        for row in block['rows']:
            key = row['case_id']
            assert key not in rows, key
            rows[key] = dict(row, report_group=group)
    return rows


def metrics(row):
    keys = ['safe_success', 'collisions', 'false_arrival', 'physical_guards',
        'protocol_valid', 'seconds', 'travel_m', 'frames', 'voluntary_wait_frames',
        'only_stop_frames', 'stop_choices']
    if row.get('recovery_test'):
        keys += ['post_release_seconds', 'model_controlled_frames',
            'post_release_wait_with_choices_frames', 'post_release_only_stop_frames']
    return {key:row[key] for key in keys}


def main():
    current_path = BASE/'v2_behavior_retention_validation_20261017/RESULTS.json'
    prior_path = BASE/'guard_aligned_mixed_ppo_resume_validation_20261017/RESULTS.json'
    stable_path = BASE/'normal_suite_wait_validation_v2_20261016/RESULTS.json'
    fit_path = HERE/'FIT_RESULTS.json'
    captured = {path:path.read_bytes() for path in [current_path,prior_path,stable_path,fit_path]}
    current, prior, stable, fit = [json.loads(captured[path]) for path in [current_path, prior_path, stable_path, fit_path]]
    assert current['model'] == fit['model']
    assert prior['completed'] and stable['completed']
    before, older, after = [flatten(report) for report in [prior, stable, current]]
    rows = []
    for key, row in after.items():
        if key not in before:
            continue
        reference = before[key]
        recovery = bool(row.get('recovery_test'))
        assert recovery == bool(reference.get('recovery_test'))
        if recovery:
            assert row['release_sensor_timestamp_s'] == reference['release_sensor_timestamp_s']
            assert row['prefix_trace_sha256'] == reference['prefix_trace_sha256']
        value = dict(case_id=key, group=row['report_group'],
            recovery_is_initialization_then_learned_continuation=recovery,
            prior_ppo=metrics(reference), current_auxiliary=metrics(row),
            time_delta_s=row['seconds']-reference['seconds'],
            travel_delta_m=row['travel_m']-reference['travel_m'],
            voluntary_wait_frame_delta=row['voluntary_wait_frames']-reference['voluntary_wait_frames'],
            only_stop_frame_delta=row['only_stop_frames']-reference['only_stop_frames'])
        if key in older:
            value['older_stable'] = metrics(older[key])
        rows.append(value)
    normal = [row for row in rows if row['group']=='normal18']
    assert len(normal) <= 18
    normal_summary = dict(completed_matched_cases=len(normal), expected_cases=18,
        complete=len(normal)==18, measured_on_current_completed_cases_only=True)
    for name in ['prior_ppo','current_auxiliary','older_stable']:
        available = [row[name] for row in normal if name in row]
        normal_summary[name] = dict(cases=len(available),
            safe_successes=sum(bool(row['safe_success']) for row in available),
            collisions=sum(row['collisions'] for row in available),
            time_s=sum(row['seconds'] for row in available),
            travel_m=sum(row['travel_m'] for row in available),
            voluntary_wait_frames=sum(row['voluntary_wait_frames'] for row in available),
            only_stop_frames=sum(row['only_stop_frames'] for row in available))
    output = dict(at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
        current_validation_complete=current['completed'],
        actual_current_physical_episodes=current['new_physical_episodes'],
        completed_matched_cases=len(rows), normal18=normal_summary, rows=rows,
        auxiliary_updates=fit['accepted_supervised_updates'], new_ppo_steps=0,
        source_additional_ppo_steps=fit['source_prior_additional_ppo_steps'],
        dynamic00_and01_are_training_development_cases=True,
        voluntary_wait_is_not_automatically_a_policy_failure=True,
        only_stop_frames_have_no_motion_action_for_actor_to_select=True,
        recovery_prefix_metrics_not_full_mission_policy_scores=True,
        unexecuted_cases_not_counted_as_passed=True,
        no_new_localizer_router_or_hardware_improvement_claim=True,
        inputs={str(path.relative_to(ROOT)):hashlib.sha256(raw).hexdigest() for path,raw in captured.items()},
        comparison_implementation_sha256=digest(Path(__file__)),
        current_validation_snapshot=str((HERE/'COMPARISON_VALIDATION_SNAPSHOT.json').relative_to(ROOT)),
        source_validation_safety_and_efficiency_gates=current.get('gates',{}),
        goal_achieved=False, deployment_approved=False)
    path = HERE/'COMPLETED_CASE_COMPARISON.json'
    snapshot = HERE/'COMPARISON_VALIDATION_SNAPSHOT.json'
    snapshot_temporary = snapshot.with_suffix('.tmp')
    snapshot_temporary.write_bytes(captured[current_path])
    snapshot_temporary.replace(snapshot)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(output, ensure_ascii=False, indent=2)+'\n')
    temporary.replace(path)
    print(json.dumps(dict(at=output['at'], matched_cases=len(rows),
        current_physical_episodes=output['actual_current_physical_episodes'],normal18=normal_summary),ensure_ascii=False))


if __name__ == '__main__':
    main()
