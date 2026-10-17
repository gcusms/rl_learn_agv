"""Independently audit feedback actually emitted during completed v2 physics.

This reads closed, checksum-verified traces from a point-in-time report. It
does not run a simulator, train a network, change permissions, or count source
initialization frames as policy-controlled navigation. A passing audit proves
feature transport and bookkeeping, never useful recovery or safety by itself.
"""
from collections import Counter
from datetime import datetime
import hashlib
import json
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
SOURCE = ROOT / 'reports/stop_feedback_20261015/guard_forecast_migration_validation_20261017/RESULTS.json'


def digest(data):
    return hashlib.sha256(data).hexdigest()


def assemble_history(current, times):
    # Use sensor timestamps and independent array indexing, not runtime/history
    # imports. Missing history must remain zero and is not certified clearance.
    result = np.zeros((len(times), 42), np.float32)
    for index, now in enumerate(times):
        for part, age in enumerate((.4, .2, 0.)):
            target = now-age
            match = np.flatnonzero(times[:index+1] <= target+1e-8)
            if len(match) and target-times[match[-1]] <= .05+1e-8:
                result[index, part*14:(part+1)*14] = current[match[-1]]
    return result


def audit():
    assert not (HERE/'RESULTS.json').exists(), 'Evidence is append-only; use a new audit directory for another snapshot'
    source_bytes = SOURCE.read_bytes()
    report = json.loads(source_bytes)
    (HERE/'SOURCE_SNAPSHOT.json').write_bytes(source_bytes)
    paths = {str(Path(__file__).relative_to(ROOT)): digest(Path(__file__).read_bytes())}
    rows, totals = [], Counter()
    bundle = ROOT/'models/stop_feedback_20261015/guard_aligned_migration_s231017/portable'
    hardware_path, feedback_path = bundle/'hardware.json', bundle/'decision_feedback.json'
    commands = np.asarray(json.loads(hardware_path.read_text())['policy_action']['choices'])
    contract = json.loads(feedback_path.read_text())
    paths.update({str(p.relative_to(ROOT)): digest(p.read_bytes()) for p in (hardware_path, feedback_path)})
    assert len(contract['action_commands']) == 12
    np.testing.assert_array_equal(commands, contract['action_commands'])
    for group, data in report['groups'].items():
        for row in data['rows']:
            path = ROOT/row['trace_path']
            raw = path.read_bytes()
            assert digest(raw) == row['trace_sha256'], path
            paths[str(path.relative_to(ROOT))] = digest(raw)
            with np.load(path, allow_pickle=False) as saved:
                x, masks, actions = saved['observations'], saved['masks'], saved['choices']
                times, short = saved['labels'][:, 0], saved['constraint_diagnostics']
                n = len(times)
                assert x.shape == (n, 625) and masks.shape == (n, 12) and short.shape == (n, 15)
                assert np.isfinite(x).all() and np.all(np.diff(times) > 0.)
                expected_mask = np.column_stack([np.ones(n, bool), x[:, 235:244] > .5, x[:, 267:269] > .5])
                np.testing.assert_array_equal(masks, expected_mask)
                assert masks[np.arange(n), actions].all()
                reasons = x[:, 303:399].reshape(n, 12, 8)
                np.testing.assert_array_equal(reasons.sum(2), np.ones((n, 12), np.float32))
                assert np.isin(reasons, [0., 1.]).all()
                np.testing.assert_array_equal(reasons[:, 1:, 0] > .5, masks[:, 1:])
                np.testing.assert_array_equal(assemble_history(x[:, 467:481], times), x[:, 439:481])
                contacts = short[:, 3:]
                supported = short[:, 0] > 0
                valid = (x[:, 435] > .5) & (x[:, 438] > .5)
                finite = np.isfinite(contacts)
                assert not (finite & ~supported[:, None]).any()
                assert not ((contacts[finite] < 0.) | (contacts[finite] > 1.2+1e-8)).any()
                expected = np.zeros_like(contacts, np.float32)
                expected[finite] = np.clip(1.-contacts[finite]/5.2, 0., 1.)
                admitted = finite & valid[:, None]
                assert np.all(x[:, 467:479][admitted] >= expected[admitted]-1e-7)
                assert np.all(x[admitted.any(1), 479] == 1.)
                supported_floor_frames = admitted.any(1)
                controlled = saved['model_controlled'].astype(bool) if 'model_controlled' in saved.files else np.ones(n, bool)
                if row.get('recovery_test'):
                    np.testing.assert_array_equal(controlled, np.arange(n) >= int(row['frozen_prefix_frames']))
                only_stop = ~masks[:, 1:].any(1)
                near_goal = np.linalg.norm(x[:, 218:220]*12., axis=1) <= .3
                chosen_stop = actions == 0
                moving_executions = np.any(abs(saved['commands']) > 1e-8, axis=1)
                np.testing.assert_allclose(saved['commands'][moving_executions], commands[actions[moving_executions]], rtol=0, atol=2e-8)
                motion_denied = Counter()
                for column in range(1, 8):
                    motion_denied[contract['rejection_groups'][column]] = int(reasons[controlled, 1:, column].sum())
                info = dict(group=group, case_id=row['case_id'], physical_success=bool(row['safe_success']),
                    complete_trace_frames=n, initialization_frames=int((~controlled).sum()), controlled_frames=int(controlled.sum()),
                    controlled_only_stop_frames=int((controlled & only_stop).sum()),
                    controlled_chooses_stop_with_motion_and_goal_far=int((controlled & chosen_stop & ~only_stop & ~near_goal).sum()),
                    controlled_stop_contact_frames=int((controlled & admitted[:, 0]).sum()),
                    controlled_contact_frames=int((controlled & supported_floor_frames).sum()),
                    controlled_per_action_rejection_groups=dict(motion_denied),
                    unknown_is_not_certified_clear=True, source_trace_sha256=digest(raw))
                totals.update({k:v for k,v in info.items() if isinstance(v, int) and not isinstance(v, bool)})
                rows.append(info)
    result = dict(completed=True, at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
        source_report_snapshot=str((HERE/'SOURCE_SNAPSHOT.json').relative_to(ROOT)), source_snapshot_sha256=digest(source_bytes),
        source_report_was_completed=report['completed'], actual_closed_episodes=len(rows), totals=dict(totals), rows=rows,
        emitted_risk_lower_bound_matches_current_short_guard=True, emitted_physical_age_history_exact=True,
        emitted_current_reasons_match_action_permissions=True, sampled_choices_currently_legal=True,
        moving_executions_match_selected_original_command=True,
        initialization_separated_from_controlled_frames=True, new_physical_steps=0, new_ppo_steps=0,
        does_not_prove_network_understands_features_or_recovers=True,
        goal_achieved=False, deployment_approved=False, inputs=paths)
    (HERE/'RESULTS.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ['actual_closed_episodes','totals','emitted_risk_lower_bound_matches_current_short_guard','emitted_physical_age_history_exact']}, ensure_ascii=False))


if __name__ == '__main__':
    audit()
