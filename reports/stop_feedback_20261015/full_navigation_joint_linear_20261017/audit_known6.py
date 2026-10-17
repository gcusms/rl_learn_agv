"""Check all six known full missions against their actual safe reference versions."""
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
from zoneinfo import ZoneInfo

import numpy as np

from neural_navigation.experiment_budget import ExperimentBudget


ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
BASE = HERE.parent


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def same(a, b):
    return np.array_equal(a, b, equal_nan=True) if a.dtype.kind in 'fc' else np.array_equal(a, b)


def main():
    assert not (HERE / 'KNOWN6_PHYSICAL_AUDIT.json').exists()
    budget = ExperimentBudget(.1, 550, HERE)
    source_paths = [BASE / 'full_navigation_margin_resume_validation_20261017/RESULTS.json',
        BASE / 'v2_behavior_retention_validation_20261017/RESULTS.json']
    source_paths = [source_paths[0]] + [source_paths[1]] * 5
    ids = ['pre_ppo_holdout_sensor_01', 'pre_ppo_holdout_dynamic_02',
        'pre_ppo_holdout_dynamic_00', 'pre_ppo_holdout_dynamic_01',
        'pre_ppo_holdout_dynamic_03', 'pre_ppo_holdout_sensor_00']
    source_rows = [next(r for r in read(path)['groups']['observed6']['rows'] if r['case_id'] == key)
        for key, path in zip(ids, source_paths)]
    validation_path = BASE / 'full_navigation_joint_linear_validation_20261017/RESULTS.json'
    raw = validation_path.read_bytes()
    report = json.loads(raw)
    assert report['model'] == 'models/stop_feedback_20261015/full_navigation_joint_linear_s231017'
    fit = read(HERE / 'FIT_RESULTS.json')
    assert fit['fit_passed'] and fit['model'] == report['model']
    assert digest(ROOT / report['model'] / 'checkpoint.pt') == fit['checkpoint_sha256']
    rows = {row['case_id']:row for row in report['groups']['observed6']['rows']}
    assert len(rows) == 6
    assert all(key in rows for key in ids), 'Wait until all six known missions finish'
    snapshot = HERE / 'KNOWN6_VALIDATION_SNAPSHOT.json'
    assert not snapshot.exists()
    snapshot.write_bytes(raw)
    paths = [Path(__file__), snapshot, HERE / 'FIT_RESULTS.json', HERE / 'ADMISSION.json', *source_paths]
    results = []
    for key, source in zip(ids, source_rows):
        row = rows[key]
        assert source['safe_success'] and source['protocol_valid']
        assert not any(source[name] for name in ('collisions', 'false_arrival', 'physical_guards'))
        assert row['safe_success'] and row['protocol_valid']
        assert not any(row[name] for name in ('collisions', 'false_arrival', 'physical_guards', 'timeout'))
        assert all(row[name] <= source[name] * 1.2 + 1e-8 for name in ('seconds', 'travel_m'))
        pair = [ROOT / record['trace_path'] for record in (source, row)]
        for record, path in zip((source, row), pair):
            assert digest(path) == record['trace_sha256']
            paths.append(path)
        with np.load(pair[0], allow_pickle=False) as old, np.load(pair[1], allow_pickle=False) as new:
            fields = ['observations', 'masks', 'choices', 'commands', 'scans', 'odometry', 'measured_velocity',
                'labels', 'human_positions', 'route_diagnostics', 'route_reasons']
            matched = {name:same(old[name], new[name]) for name in fields}
            matched['predictive_constraints_excluding_compute_ms'] = same(np.delete(old['constraint_diagnostics'], 2, axis=1),
                np.delete(new['constraint_diagnostics'], 2, axis=1))
            assert new['observations'].shape == (row['frames'], 625)
            assert 'model_controlled' not in new.files or new['model_controlled'].all()
            assert new['labels'][-1, 7] < .2
            np.testing.assert_array_equal(new['masks'], np.c_[np.ones(row['frames'], bool),
                new['observations'][:, 235:244] > .5, new['observations'][:, 267:269] > .5])
            assert new['masks'][np.arange(row['frames']), new['choices']].all()
            from neural_navigation.masked_actions import COMMANDS
            np.testing.assert_allclose(new['commands'][:-1], COMMANDS[new['choices'][:-1]], rtol=0, atol=2e-8)
        results.append(dict(case_id=key, safe_success=True, seconds=row['seconds'], travel_m=row['travel_m'],
            voluntary_wait_frames=row['voluntary_wait_frames'], frames=row['frames'],
            exact_source_fields=matched, all_non_timing_trace_fields_exact=all(matched.values()),
            complete_learned_mission=True, runtime_teacher_or_forced_action_sequence=False))
        budget.tick()
    model = ROOT / report['model'] / 'portable'
    parent = ROOT / 'models/stop_feedback_20261015/full_navigation_margin_resume_s231017/portable'
    old_manifest, new_manifest = read(parent / 'manifest.json'), read(model / 'manifest.json')
    preserved = {}
    for name, expected in old_manifest['files'].items():
        if name not in ('actor.pt', 'navigation_preservation.json'):
            preserved[name] = digest(model / name) == expected
            assert preserved[name], name
    hashes = {str(path.relative_to(ROOT)):digest(path) for path in paths}
    budget.tick()
    assert not {'torch', 'mujoco', 'rclpy'} & sys.modules.keys()
    result = dict(completed=True, at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
        model=report['model'], known_cases=results, safe_cases=len(results),
        recorded_frames=sum(row['frames'] for row in results),
        current_reference_successes_and20pct_efficiency_all_passed=True,
        all_known_non_timing_reference_traces_exact=all(row['all_non_timing_trace_fields_exact'] for row in results),
        protected_files_unchanged=preserved,
        original_body4cm_masks_memory_dynamics_localizer_router_terminal_preserved=True,
        all_six_cases_are_training_development_not_generalization=True,
        original28_other_known_hard97_6_106_and_conditional_fresh_requirements_not_waived=True,
        inputs=hashes, resource=budget.statistics(),
        model_or_physics_not_loaded=True, new_physical_episodes=0, new_ppo_steps=0,
        goal_achieved=False, deployment_approved=False)
    (HERE / 'KNOWN6_PHYSICAL_AUDIT.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k:result[k] for k in ('completed', 'model', 'safe_cases', 'recorded_frames', 'all_known_non_timing_reference_traces_exact', 'resource')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
