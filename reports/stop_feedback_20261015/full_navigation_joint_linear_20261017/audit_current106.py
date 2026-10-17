"""Classify the current106 failure from captured arrays, without a model."""
from collections import Counter, defaultdict
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
from zoneinfo import ZoneInfo

import numpy as np

from neural_navigation.decision_feedback import FEEDBACK_REASONS, GROUPS
from neural_navigation.experiment_budget import ExperimentBudget
from neural_navigation.masked_actions import COMMANDS


ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    output = HERE / 'CURRENT106_FAILURE_AUDIT.json'
    assert not output.exists()
    budget = ExperimentBudget(.1, 550, HERE)
    result_path = HERE.parent / 'full_navigation_joint_linear_validation_20261017/RESULTS.json'
    raw = result_path.read_bytes()
    validation = json.loads(raw)
    assert validation['completed'] and validation['all_fixed28_passed']
    assert validation['model'] == 'models/stop_feedback_20261015/full_navigation_joint_linear_s231017'
    row = next(r for r in validation['groups']['hard3']['rows']
        if r['case_id'] == 'w11_training_stall_t106')
    assert row['timeout'] and row['protocol_valid'] and not row['safe_success']
    assert not any(row[k] for k in ('collisions', 'false_arrival', 'physical_guards'))
    trace = ROOT / row['trace_path']
    assert digest(trace) == row['trace_sha256']
    with np.load(trace, allow_pickle=False) as data:
        x, masks, choices, commands = [data[k].copy() for k in
            ('observations', 'masks', 'choices', 'commands')]
        labels, controlled = data['labels'].copy(), data['model_controlled'].copy()
    np.testing.assert_array_equal(controlled, np.arange(700) >= 530)
    assert abs(labels[530, 0] - 106.) < 2e-8
    np.testing.assert_array_equal(masks, np.c_[np.ones(len(x), bool), x[:, 235:244] > .5, x[:, 267:269] > .5])
    assert masks[np.arange(len(x)), choices].all()
    np.testing.assert_allclose(commands, COMMANDS[choices], rtol=0, atol=2e-8)
    eligible = ((x[:, 435] > .5) & (x[:, 438] > .5) & masks[:, 1:].any(1)
        & (np.linalg.norm(x[:, 218:220] * 12., axis=1) > .3))
    causes = x[:, 303:399].reshape(len(x), 12, 8)
    assert np.isin(causes, (0., 1.)).all()
    np.testing.assert_array_equal(causes.sum(2), np.ones((len(x), 12)))
    np.testing.assert_array_equal(causes[:, 1:, 0] > .5, masks[:, 1:])
    reference_path = HERE / 'training_data.npz'
    with np.load(reference_path, allow_pickle=False) as data:
        reference, targets, reference_masks = [data[k].copy() for k in
            ('observations', 'actions', 'masks')]
    assert len(reference) == 5608
    matches = defaultdict(Counter)
    exact = defaultdict(Counter)
    for state, mask, target in zip(reference, reference_masks, targets):
        signature = (np.packbits(mask).tobytes(), state[303:399].tobytes())
        matches[signature][int(target)] += 1
        exact[state.tobytes()][int(target)] += 1
    counts, signatures = [], Counter()
    for i in np.flatnonzero(controlled):
        signature = (np.packbits(masks[i]).tobytes(), x[i, 303:399].tobytes())
        matching = matches.get(signature, {})
        exact_matching = exact.get(x[i].tobytes(), {})
        counts.append(dict(frame=int(i), elapsed_s=float(labels[i, 0]),
            trainable=bool(eligible[i]), legal_motion_actions=(np.flatnonzero(masks[i, 1:]) + 1).tolist(),
            reference_motion_labels=sum(n for action, n in matching.items() if action != 0),
            reference_stop_labels=int(matching.get(0, 0)),
            exact625_reference_matches=int(sum(exact_matching.values()))))
        signatures[signature] += 1
        budget.tick()
    snapshot = HERE / 'CURRENT106_FAILURE_VALIDATION_SNAPSHOT.json'
    assert not snapshot.exists()
    snapshot.write_bytes(raw)
    paths = [Path(__file__), snapshot, trace, reference_path, HERE / 'FIT_RESULTS.json', HERE / 'actor.py',
        ROOT / 'training/neural_navigation/decision_feedback.py',
        ROOT / 'training/neural_navigation/masked_actions.py']
    hashes = {str(path.relative_to(ROOT)): digest(path) for path in paths}
    budget.tick()
    assert not {'torch', 'mujoco', 'rclpy'} & sys.modules.keys()
    result = dict(completed=True, at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
        model=validation['model'], source_case=row['case_id'],
        post_release_frames=int(controlled.sum()),
        post_release_trainable_frames=int((controlled & eligible).sum()),
        post_release_stop_choices=int((controlled & (choices == 0)).sum()),
        post_release_only_stop_frames=int((controlled & ~masks[:, 1:].any(1)).sum()),
        per_motion_action_legal_frames={str(i): int(masks[controlled, i].sum()) for i in range(1, 12)},
        previous_execution_reasons=dict(Counter(FEEDBACK_REASONS[int(i)]
            for i in x[controlled, 399:419].argmax(1))),
        rejected_action_groups={name: int(((causes[:, 1:, k] > .5)
            & ~masks[:, 1:] & controlled[:, None]).sum()) for k, name in enumerate(GROUPS)},
        supported_reference_motion_match_frames=sum(c['trainable'] and c['reference_motion_labels'] > 0 for c in counts),
        supported_reference_no_motion_match_frames=sum(c['trainable'] and c['reference_motion_labels'] == 0 for c in counts),
        exact625_reference_match_frames=sum(c['exact625_reference_matches'] > 0 for c in counts),
        distinct_post_release_mask_cause_signatures=len(signatures),
        rows=counts, physical_seconds=row['seconds'], final_xy_error_m=row['final_xy_error_m'],
        currently_legal_does_not_prove_safe_complete_path=True,
        matched_mask_cause_does_not_mean_same_full_state=True,
        no_infeasibility_or_neural_understanding_claim=True,
        inputs=hashes, resource=budget.statistics(),
        model_or_physics_not_loaded=True, new_ppo_steps=0,
        new_physical_episodes=0, new_training_updates=0,
        goal_achieved=False, deployment_approved=False)
    temporary = output.with_suffix('.tmp')
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(output)
    print(json.dumps({k: result[k] for k in ('completed', 'post_release_frames',
        'post_release_trainable_frames', 'post_release_stop_choices', 'post_release_only_stop_frames',
        'supported_reference_motion_match_frames', 'exact625_reference_match_frames', 'resource')},
        ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
