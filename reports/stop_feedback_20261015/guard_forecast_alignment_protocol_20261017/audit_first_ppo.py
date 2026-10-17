"""Audit a closed actual on-policy episode, without touching the live learner.

This validates execution/history/feedback records. The unchanged-behavior
likelihood replay is a separate gate performed by the learner on a complete
collection batch; finite stored log probabilities alone do not prove it.
"""
from datetime import datetime
import hashlib
import json
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent


def main():
    output = HERE/'PPO_FIRST_EPISODE_AUDIT.json'
    assert not output.exists()
    report_path = HERE.parent/'guard_aligned_mixed_ppo_20261017/RESULTS.json'
    raw = report_path.read_bytes()
    report = json.loads(raw)
    row = report['episodes'][0]
    path = ROOT/row['rollout_path']
    assert hashlib.sha256(path.read_bytes()).hexdigest() == row['rollout_sha256']
    hardware_path = ROOT/'models/stop_feedback_20261015/guard_aligned_migration_s231017/portable/hardware.json'
    commands = np.asarray(json.loads(hardware_path.read_text())['policy_action']['choices'])
    with np.load(path, allow_pickle=False) as data:
        x, masks, actions = data['observations'], data['masks'], data['actions']
        n = len(actions)
        temperature = float(data['temperature'])
        assert temperature == 4.
        np.testing.assert_array_equal(masks, np.column_stack([np.ones(n, bool), x[:,235:244] > .5, x[:,267:269] > .5]))
        assert masks[np.arange(n), actions].all()
        assert not data['execution_changed'].any()
        np.testing.assert_allclose(data['executed_commands'], commands[actions], rtol=0, atol=2e-8)
        np.testing.assert_array_equal(x[1:,222:224], (data['executed_commands'][:-1]/[.2,.5]).astype(np.float32))
        normalized_velocity = np.clip(data['input_measured_velocity']/[.2,.5], -1., 1.)
        np.testing.assert_allclose(x[:,220:222], normalized_velocity, rtol=0, atol=3e-7)
        probes = data['per_action_guard_executed']
        accepted = np.isclose(probes, commands[None], rtol=0, atol=1e-9).all(2)
        np.testing.assert_array_equal(accepted[:,1:], masks[:,1:])
        groups = x[:,303:399].reshape(n,12,8)
        np.testing.assert_array_equal(groups.sum(2), np.ones((n,12), np.float32))
        np.testing.assert_array_equal(groups[:,1:,0] > .5, masks[:,1:])
        contacts = data['short_guard_contacts']
        valid = (x[:,435] > .5) & (x[:,438] > .5)
        finite = np.isfinite(contacts)
        admitted = finite & valid[:,None]
        expected = np.zeros_like(contacts, np.float32)
        expected[finite] = np.clip(1.-contacts[finite]/5.2,0.,1.)
        assert (x[:,467:479][admitted] >= expected[admitted]-1e-7).all()
        assert np.isfinite(data['old_log_probability']).all() and np.isfinite(data['old_values']).all()
        only_stop = ~masks[:,1:].any(1)
        np.testing.assert_array_equal(actions[only_stop], np.zeros(only_stop.sum(), actions.dtype))
        np.testing.assert_allclose(data['old_log_probability'][only_stop], 0., rtol=0, atol=2e-6)
        terminal = np.zeros(n,bool);terminal[-1] = True
        np.testing.assert_array_equal(data['terminated'], terminal)
        np.testing.assert_array_equal(data['episode_ends'], terminal)
        times = data['labels'][:,0]
        assert (np.diff(times) > 0.).all()
        checked = 0
        for index,now in enumerate(times):
            if now-times[0] < .4-1e-8:
                continue  # Earlier frames belong to the real initialization.
            for part,age in enumerate((.4,.2,0.)):
                found = np.flatnonzero(times[:index+1] <= now-age+1e-8)
                sample = found[-1]
                assert now-age-times[sample] <= .05+1e-8
                np.testing.assert_array_equal(x[index,439+part*14:453+part*14], x[sample,467:481])
            checked += 1
    (HERE/'PPO_FIRST_EPISODE_SOURCE_SNAPSHOT.json').write_bytes(raw)
    result = dict(completed=True, at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
        actual_sampled_decisions=n, physically_executed_action_history_exact=True,
        no_unexpected_execution_rewrites=True, current_rejection_groups_match_permissions=True,
        emitted_short_contact_risk_lower_bound=True, physical_age_history_checked_frames=checked,
        warm_initial_history_not_falsely_replaced_with_zeros=True, sampling_temperature=temperature,
        measured_velocity_matches_existing_clipped_normalization=True,
        complete_finite_task_terminal_not_reset_bootstrap=True,
        only_stop_frames=int(only_stop.sum()), stop_with_motion_frames=int(((actions==0)&~only_stop).sum()),
        unchanged_behavior_likelihood_replay_still_requires_complete_collection_batch=True,
        training_episode_safe_success=row['safe_success'], training_episode_is_not_independent_validation=True,
        new_actor_updates_at_report_snapshot=report['new_actor_optimizer_updates'],
        source_report_snapshot_sha256=hashlib.sha256(raw).hexdigest(),
        inputs={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__), path, hardware_path,
            ROOT/'training/neural_navigation/core.py']},
        goal_achieved=False, deployment_approved=False)
    output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ['actual_sampled_decisions','physically_executed_action_history_exact',
        'no_unexpected_execution_rewrites','physical_age_history_checked_frames','training_episode_safe_success']},ensure_ascii=False))


if __name__ == '__main__':
    main()
