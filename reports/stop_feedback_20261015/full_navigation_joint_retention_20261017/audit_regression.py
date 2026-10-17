"""Locate the actual first decision divergence using recorded physical trials."""
from datetime import datetime
import hashlib
import json
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

from neural_navigation.experiment_budget import ExperimentBudget


ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
BASE = HERE.parent


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def same(a, b):
    return np.array_equal(a, b, equal_nan=True) if a.dtype.kind in 'fc' else np.array_equal(a, b)


def main():
    assert not (HERE / 'REGRESSION_AUDIT.json').exists()
    budget = ExperimentBudget(.1, 550, HERE)
    before_path = BASE / 'v2_behavior_retention_validation_20261017/RESULTS.json'
    after_path = BASE / 'full_navigation_margin_resume_validation_20261017/RESULTS.json'
    before, after = [json.loads(path.read_text()) for path in (before_path, after_path)]
    assert before['completed'] and after['completed'] and after['all_fixed28_passed']
    key = 'pre_ppo_holdout_dynamic_02'
    a, b = [next(row for row in report['groups']['observed6']['rows'] if row['case_id'] == key)
        for report in (before, after)]
    assert a['safe_success'] and b['timeout'] and not b['safe_success']
    paths = [before_path, after_path, Path(__file__)]
    for row in (a, b):
        path = ROOT / row['trace_path']
        assert digest(path) == row['trace_sha256']
        paths.append(path)
    with np.load(ROOT / a['trace_path'], allow_pickle=False) as old, np.load(ROOT / b['trace_path'], allow_pickle=False) as new:
        count = min(len(old['choices']), len(new['choices']))
        differences = np.flatnonzero(old['choices'][:count] != new['choices'][:count])
        assert len(differences)
        first = int(differences[0])
        fields = ['observations', 'masks', 'scans', 'odometry', 'measured_velocity',
            'labels', 'human_positions', 'route_diagnostics', 'route_reasons']
        parity = {name:same(old[name][:first + 1], new[name][:first + 1]) for name in fields}
        parity['prior_choices'] = same(old['choices'][:first], new['choices'][:first])
        parity['prior_commands'] = same(old['commands'][:first], new['commands'][:first])
        assert all(parity.values()), parity
        choice_before, choice_after = int(old['choices'][first]), int(new['choices'][first])
        assert new['masks'][first, choice_before] and new['masks'][first, choice_after]
        x = new['observations']
        allowed = new['masks'][:, 1:].any(1)
        supported = ((x[:, 435] > .5) & (x[:, 438] > .5) & allowed
            & (np.linalg.norm(x[:, 218:220] * 12., axis=1) > .3))
        waits = (new['choices'] == 0) & allowed
        durations = np.diff(np.r_[new['labels'][:, 0], b['seconds']])
        longest = run = 0.
        for waiting, duration in zip(waits, durations):
            run = run + duration if waiting else 0.
            longest = max(longest, run)
        details = dict(first_divergence_frame=first, first_divergence_sensor_time_s=float(new['labels'][first, 0]),
            first_old_choice=choice_before, first_new_choice=choice_after,
            first_old_command=old['commands'][first].tolist(), first_new_command=new['commands'][first].tolist(),
            first_both_actions_originally_legal=True, all_prior_inputs_and_physics_exact=parity,
            first_new_head_trainable=bool(supported[first]),
            voluntary_wait_frames=int(waits.sum()), voluntary_wait_seconds=float(durations[waits].sum()),
            longest_voluntary_wait_seconds=float(longest), trainable_voluntary_wait_frames=int((waits & supported).sum()),
            only_stop_frames=int((~allowed).sum()), final_xy_error_m=float(new['labels'][-1, 7]))
    budget.tick()
    result = dict(completed=True, at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(), case_id=key,
        previous_safe_seconds=a['seconds'], current_timeout_seconds=b['seconds'], **details,
        conclusion='The learned full-navigation head caused an actual ordinary-navigation regression; current mask allowed the prior useful action. Fixed28 retention alone did not cover this complete-mission state distribution.',
        next_experiment='Jointly preserve all4468existing references and all5previously safe complete known missions; leave frozen prospective seeds excluded and all original gates intact.',
        model_and_physics_not_loaded=True, new_training_updates=0, new_physical_episodes=0,
        inputs={str(path.relative_to(ROOT)):digest(path) for path in paths}, resource=budget.statistics(),
        goal_achieved=False, deployment_approved=False)
    (HERE / 'REGRESSION_AUDIT.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('inputs', 'all_prior_inputs_and_physics_exact')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
