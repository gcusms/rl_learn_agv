"""Read-only coverage of recorded v2 PPO states; no model or physics loads.

An identical mask/cause signature is deliberately not called an identical
state: history, ranges, goals and velocity still matter for choosing an action.
"""
from collections import defaultdict
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


def masks(observations):
    return np.concatenate((np.ones((len(observations), 1), bool),
        observations[:, 235:244] > .5, observations[:, 267:269] > .5), axis=1)


def support(observations, legal):
    return ((observations[:, 435] > .5) & (observations[:, 438] > .5)
        & legal[:, 1:].any(axis=1)
        & (np.linalg.norm(observations[:, 218:220] * 12., axis=1) > .3))


def signature(observation, legal):
    return np.packbits(legal).tobytes() + np.packbits(observation[303:399] > .5).tobytes()


def main():
    budget = ExperimentBudget(.1, 550, HERE)
    data_path = HERE / 'training_data.npz'
    admission_path = HERE / 'ADMISSION.json'
    admission = json.loads(admission_path.read_text())
    assert admission['admitted'] and digest(data_path) == admission['archive_sha256']
    with np.load(data_path, allow_pickle=False) as data:
        x = data['observations'].copy()
        y = data['actions'].copy()
        legal = data['masks'].copy()
        groups = (data['case_index'] >= 0).astype(np.int64)
    np.testing.assert_array_equal(masks(x), legal)
    assert len(x) == 5608 and np.isfinite(x).all()
    exact_inputs = {row.tobytes() for row in x}
    pools = defaultdict(list)
    powers = 1 << np.arange(12)
    codes = (legal * powers).sum(1)
    for index, row in enumerate(x):
        pools[signature(row, legal[index])].append(index)
    source_paths = [Path(__file__), data_path, admission_path, HERE / 'actor.py']
    records = []
    for folder in ('guard_aligned_mixed_ppo_20261017',
            'guard_aligned_mixed_ppo_resume_20261017'):
        report_path = BASE / folder / 'RESULTS.json'
        source_paths.append(report_path)
        report = json.loads(report_path.read_text())
        for episode in report['episodes']:
            # The resumed report also contains prior complete episodes.
            if episode.get('release_frame') is None or episode['rollout_path'].split('/')[-2] != folder:
                continue
            path = ROOT / episode['rollout_path']
            assert digest(path) == episode['rollout_sha256']
            source_paths.append(path)
            with np.load(path, allow_pickle=False) as data:
                q = data['observations'].copy()
                allowed = data['masks'].copy()
                actions = data['actions'].copy()
            assert q.shape[1] == 625 and np.isfinite(q).all()
            np.testing.assert_array_equal(masks(q), allowed)
            trainable = support(q, allowed)
            mask_codes = (allowed * powers).sum(1)
            patterns = []
            for code in np.unique(mask_codes[trainable]):
                selected = trainable & (mask_codes == code)
                source = codes == code
                patterns.append(dict(legal_actions=np.flatnonzero(allowed[np.flatnonzero(selected)[0]]).tolist(),
                    recorded_frames=int(selected.sum()),
                    recorded_stop_choices=int((selected & (actions == 0)).sum()),
                    supervised_old_stop=int((source & (groups == 0) & (y == 0)).sum()),
                    supervised_old_motion=int((source & (groups == 0) & (y != 0)).sum()),
                    supervised_new_stop=int((source & (groups == 1) & (y == 0)).sum()),
                    supervised_new_motion=int((source & (groups == 1) & (y != 0)).sum())))
            eligible_with_positive = 0
            eligible_exact = 0
            for index in np.flatnonzero(trainable):
                source = pools.get(signature(q[index], allowed[index]), [])
                if source and (y[source] != 0).any():
                    eligible_with_positive += 1
                eligible_exact += q[index].tobytes() in exact_inputs
            records.append(dict(rollout=episode['rollout_path'],
                release_frame=episode['release_frame'],
                safe_success=episode['safe_success'], recorded_decisions=len(q),
                only_stop_frames=int((~allowed[:, 1:].any(1)).sum()),
                new_head_trainable_frames=int(trainable.sum()),
                trainable_stop_choices=int((trainable & (actions == 0)).sum()),
                trainable_motion_choices=int((trainable & (actions != 0)).sum()),
                eligible_exact625_source_rows=eligible_exact,
                eligible_frames_with_positive_labels_for_same_mask_and_current_cause_signature=eligible_with_positive,
                same_mask_patterns=patterns))
            budget.tick()
    assert len(records) == 8
    previous_path = BASE / 'full_navigation_residual_20261017/HARD_COVERAGE.json'
    source_paths.append(previous_path)
    previous = json.loads(previous_path.read_text())
    previous_index = {(row['rollout'], row['release_frame']): row
        for row in previous['recorded_prior_v2_training_rollouts']}
    for row in records:
        old = previous_index[(row['rollout'], row['release_frame'])]
        field = 'eligible_frames_with_positive_labels_for_same_mask_and_current_cause_signature'
        row['previous4468_labels_positive_signature_frames'] = old[field]
        row['additional_positive_signature_frames_from_new1140_labels'] = row[field] - old[field]
    hashes = {str(p.relative_to(ROOT)): digest(p) for p in source_paths}
    result = dict(completed=True, at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
        training_labels=len(x), supervised_data_unchanged=True,
        recorded_prior_v2_training_rollouts=records,
        source_labels_not_automatically_correct_for_other_states=True,
        same_mask_and_cause_not_same_full_observation=True,
        does_not_evaluate_current_candidate_or_prove_infeasibility=True,
        no_pseudo_positive_labels_or_failed_trace_imitation=True,
        current_head_support_formula_matches_actor_source=True,
        model_and_physics_not_loaded=True, new_training_updates=0,
        new_ppo_steps=0, new_physical_episodes=0,
        resource=budget.statistics(), input_hashes=hashes,
        original_full_known_safety_regression_hard97_6_106_and_conditional_fresh_gates_preserved=True,
        goal_achieved=False, deployment_approved=False)
    path = HERE / 'HARD_COVERAGE.json'
    assert not path.exists()
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(dict(completed=True, rows=[{k:v for k,v in row.items()
        if k != 'same_mask_patterns'} for row in records], resource=result['resource']),
        ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
