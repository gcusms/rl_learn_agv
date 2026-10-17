"""Check causal feedback against completed physical traces, without training."""
import hashlib
import json
from pathlib import Path

import numpy as np

from neural_navigation.experiment_budget import ExperimentBudget


ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / 'reports/stop_feedback_20261015'
OUTPUT = Path(__file__).resolve().parent / 'FEEDBACK_PHYSICAL_AUDIT.json'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    assert not OUTPUT.exists(), 'Keep the first audit snapshot immutable'
    source = BASE / 'normal_suite_wait_validation_v2_20261016/RESULTS.json'
    report = json.loads(source.read_text())
    contract_path = ROOT / report['selected_model'] / 'portable/decision_feedback.json'
    contract = json.loads(contract_path.read_text())
    assert contract['schema'] == 'stop_decision_feedback_v2'
    names = contract['previous_reasons']
    assert len(names) == 20 and len(contract['rejection_groups']) == 8
    budget = ExperimentBudget(.1, memory_mib=700, directory=OUTPUT.parent)
    rows, reason_counts, files = [], {}, [Path(__file__), contract_path]
    # The live RESULTS file is snapshotted below, rather than pinned while it grows.
    snapshot = json.dumps(report, sort_keys=True).encode()
    for group, values in report['groups'].items():
        for row in values['rows']:
            path = ROOT / row['trace_path']
            assert digest(path) == row['trace_sha256']
            files.append(path)
            with np.load(path, allow_pickle=False) as trace:
                x, masks, commands, times = [trace[k].copy() for k in
                    ['observations', 'masks', 'commands', 'labels']]
                choices = trace['choices'].copy()
            times = times[:, 0]
            assert x.shape == (len(times), 625) and np.isfinite(x).all()
            groups = x[:, 303:399].reshape(-1, 12, 8)
            assert ((groups == 0) | (groups == 1)).all()
            np.testing.assert_array_equal(groups.sum(2), np.ones_like(masks))
            np.testing.assert_array_equal(groups[:, 1:, 0] == 1, masks[:, 1:])
            previous = x[:, 399:419]
            assert ((previous == 0) | (previous == 1)).all()
            np.testing.assert_array_equal(previous.sum(1), np.ones(len(times)))
            np.testing.assert_allclose(x[:, 431], masks[:, 1:].sum(1) / 11., atol=1e-7, rtol=0)
            assert np.isin(x[:, 435], [0., 1.]).all() and np.isin(x[:, 438], [0., 1.]).all()
            assert masks[np.arange(len(choices)), choices].all()
            held_since = with_choices_since = None
            for index in range(len(times) - 1):
                reason = names[int(previous[index+1].argmax())]
                reason_counts[reason] = reason_counts.get(reason, 0) + 1
                idle = bool(abs(commands[index]).max() < 1e-8)
                held = idle and reason != 'estimated_goal_reached'
                movable_idle = held and bool(masks[index, 1:].any())
                held_since = (times[index] if held_since is None else held_since) if held else None
                with_choices_since = (times[index] if with_choices_since is None else with_choices_since) if movable_idle else None
                expected = [0. if start is None else min((times[index+1]-start)/5., 1.)
                    for start in [held_since, with_choices_since]]
                np.testing.assert_allclose(x[index+1, 433:435], expected, atol=2e-6, rtol=0)
                if reason == 'policy_wait':
                    assert idle and masks[index, 1:].any()
                if reason == 'only_stop_available':
                    assert idle and not masks[index, 1:].any()
            rows.append(dict(group=group, case_id=row['case_id'], frames=len(times),
                reason_mask_and_timer_checks_passed=True,
                reconstructed_reasons_are_runtime_feedback_not_independent_cause_labels=True))
            budget.tick()
    result = dict(completed=True, rows=rows, audited_frames=sum(r['frames'] for r in rows),
        audited_episodes=len(rows), observed_executed_feedback_counts=reason_counts,
        source_report_snapshot_sha256=hashlib.sha256(snapshot).hexdigest(),
        source_report_was_complete=report['completed'],
        inputs={str(p.relative_to(ROOT)):digest(p) for p in set(files)},
        resource=budget.statistics(), no_new_physics=True, no_optimizer_updates=True,
        independent_accuracy_of_guard_reasons_not_proven_by_consistency=True,
        new_ppo_steps=0, goal_achieved=False, deployment_approved=False)
    temporary = OUTPUT.with_suffix('.tmp')
    temporary.write_text(json.dumps(result, indent=2)+'\n')
    temporary.replace(OUTPUT)
    print(json.dumps({k:v for k,v in result.items() if k not in ['rows', 'inputs']}), flush=True)


if __name__ == '__main__':
    main()
