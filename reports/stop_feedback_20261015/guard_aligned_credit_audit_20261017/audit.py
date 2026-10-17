"""Read the actual completed v2 PPO data without loading a model or simulator.

Reconstruct the complete-task advantages that were actually used, rather than
the older per-episode GAE arrays also retained in the raw collection files.
This diagnoses credit and feedback; it does not establish navigation success.
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
BASE = HERE.parent
REPORT = BASE / 'guard_aligned_mixed_ppo_resume_20261017/RESULTS.json'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def support_from_observation(x, legal):
    history = x[:, 439:481].reshape(-1, 3, 14)
    recent_or_waiting = ((history[:, :, 12] > .5).any(1)
                         | (x[:, 433:435] > 0.).any(1))
    valid = ((x[:, 435] > .5) & (x[:, 438] > .5)
             & legal[:, 1:].any(1)
             & (np.linalg.norm(x[:, 218:220] * 12., axis=1) > .3))
    near = np.min(x[:, 144:216], axis=1) * 12. < .8
    restricted = ~legal[:, 1:].all(1)
    forecast = np.max(x[:, 419:431], axis=1) > 0.
    return valid & (recent_or_waiting | near | restricted | forecast)


def complete_returns(rewards, gamma):
    result = np.zeros(len(rewards), np.float32)
    following = 0.
    for i in range(len(rewards) - 1, -1, -1):
        following = float(rewards[i]) + gamma * following
        result[i] = following
    return result


def reason_group(reason):
    if reason in ('None', 'none', 'estimated_goal_reached'):
        return 0
    if reason in ('nonfinite_input', 'stale_or_future_scan', 'scan_unavailable', 'odometry_jump'):
        return 1
    if reason in ('localization_uncertain', 'learned_route_incomplete'):
        return 2
    if reason == 'static_braking_footprint':
        return 3
    if reason in ('laser_braking_footprint', 'escape_not_increasing_clearance'):
        return 4
    if reason in ('predicted_dynamic_contact', 'predictive_command_contract'):
        return 5
    if reason in ('reverse_without_rear_visibility', 'reverse_duration_limit',
                  'unsafe_turn_blocked', 'unsafe_translation_blocked'):
        return 6
    return 7


def describe(values):
    values = np.asarray(values)
    return dict(count=int(len(values)), negative=int((values < -1e-7).sum()),
                positive=int((values > 1e-7).sum()),
                mean=float(values.mean()) if len(values) else None,
                minimum=float(values.min()) if len(values) else None,
                maximum=float(values.max()) if len(values) else None)


def main():
    assert not (HERE / 'RESULTS.json').exists(), 'Keep completed audit evidence immutable'
    raw = REPORT.read_bytes()
    report = json.loads(raw)
    assert report['completed'] and report['new_ppo_steps'] == 1993
    hardware = ROOT / 'models/stop_feedback_20261015/guard_aligned_migration_s231017/portable/hardware.json'
    commands = np.asarray(json.loads(hardware.read_text())['policy_action']['choices'])
    inputs = [Path(__file__), REPORT, hardware,
              BASE / 'native625_learning_diagnostic_20261017/batch_credit_ppo.py',
              ROOT / 'training/experiments/native625_onpolicy.py',
              ROOT / 'training/experiments/context_release_actor.py',
              ROOT / 'training/experiments/release_actor.py',
              ROOT / 'training/experiments/long_risk_actor.py',
              ROOT / 'training/experiments/guard_reason_release_actor.py',
              ROOT / 'training/neural_navigation/decision_feedback.py',
              ROOT / 'training/neural_navigation/aligned_risk_observation.py']
    batch_results, episode_results, arrays = [], [], []
    checked_history = contact_frames = 0
    for batch_meta in report['batches']:
        number = batch_meta['batch']
        rows = [r for r in report['episodes'] if r['batch'] == number]
        assert len(rows) == 5
        assert len({r['behavior_actor_sha256'] for r in rows}) == 1
        assert len({r['behavior_critic_sha256'] for r in rows}) == 1
        episodes = []
        for row in rows:
            path = ROOT / row['rollout_path']
            assert digest(path) == row['rollout_sha256'], str(path)
            inputs.append(path)
            with np.load(path, allow_pickle=False) as saved:
                z = {key: saved[key].copy() for key in saved.files}
            x, legal, actions = z['observations'], z['masks'], z['actions']
            n = len(actions)
            assert x.shape == (n, 625) and np.isfinite(x).all()
            np.testing.assert_array_equal(legal, np.column_stack([
                np.ones(n, bool), x[:, 235:244] > .5, x[:, 267:269] > .5]))
            assert legal[np.arange(n), actions].all()
            changed = z['execution_changed']
            assert np.all(z['stop_reasons'][changed] == 'estimated_goal_reached')
            np.testing.assert_allclose(z['executed_commands'][~changed], commands[actions[~changed]],
                                       rtol=0, atol=2e-8)
            np.testing.assert_array_equal(x[1:, 222:224],
                (z['executed_commands'][:-1] / [.2, .5]).astype(np.float32))
            np.testing.assert_allclose(x[:, 220:222],
                np.clip(z['input_measured_velocity'] / [.2, .5], -1., 1.), rtol=0, atol=3e-7)
            accepted = np.isclose(z['per_action_guard_executed'], commands[None],
                                  rtol=0, atol=1e-9).all(2)
            np.testing.assert_array_equal(accepted[:, 1:], legal[:, 1:])
            groups = x[:, 303:399].reshape(n, 12, 8)
            expected_groups = np.zeros_like(groups)
            for i in range(n):
                for j in range(12):
                    group = reason_group(str(z['per_action_guard_reasons'][i, j]))
                    expected_groups[i, j, 0 if accepted[i, j] else (group or 7)] = 1.
            np.testing.assert_array_equal(groups, expected_groups)
            valid = (x[:, 435] > .5) & (x[:, 438] > .5)
            contact = z['short_guard_contacts']
            finite = np.isfinite(contact)
            admitted = finite & valid[:, None]
            lower = np.zeros_like(contact, np.float32)
            lower[finite] = np.clip(1. - contact[finite] / 5.2, 0., 1.)
            assert (x[:, 467:479][admitted] >= lower[admitted] - 1e-7).all()
            contact_frames += int(admitted.any(1).sum())
            times = z['labels'][:, 0]
            assert (np.diff(times) > 0.).all()
            for i, now in enumerate(times):
                if now - times[0] < .4 - 1e-8:
                    continue
                for part, age in enumerate((.4, .2, 0.)):
                    history_i = np.flatnonzero(times[:i+1] <= now-age+1e-8)[-1]
                    assert now-age-times[history_i] <= .05+1e-8
                    np.testing.assert_array_equal(x[i, 439+part*14:453+part*14],
                                                  x[history_i, 467:481])
                checked_history += 1
            only_stop = ~legal[:, 1:].any(1)
            assert (actions[only_stop] == 0).all()
            np.testing.assert_allclose(z['old_log_probability'][only_stop], 0., rtol=0, atol=2e-6)
            assert np.isfinite(z['old_log_probability']).all() and np.isfinite(z['old_values']).all()
            terminal = np.zeros(n, bool)
            terminal[-1] = True
            np.testing.assert_array_equal(z['terminated'], terminal)
            np.testing.assert_array_equal(z['episode_ends'], terminal)
            eligible = support_from_observation(x, legal) & ~only_stop
            assert int(eligible.sum()) == row['eligible_actor_decisions']
            assert float(z['temperature']) == 4.
            returns = complete_returns(z['rewards'], .99)
            episodes.append((row, z, eligible, returns, returns-z['old_values']))
        raw_adv = np.concatenate([part[4] for part in episodes])
        keep = np.concatenate([part[2] for part in episodes])
        mean, std = float(raw_adv[keep].mean()), float(raw_adv[keep].std())
        original = batch_meta['normalization']
        np.testing.assert_allclose([mean, std],
            [original['raw_advantage_mean'], original['raw_advantage_std']], rtol=0, atol=1e-7)
        assert int(keep.sum()) == original['eligible_samples']
        result = dict(batch=number, complete_decisions=len(raw_adv), eligible_decisions=int(keep.sum()),
                      normalization_reproduced=True, raw_advantage_mean=mean, raw_advantage_std=std,
                      eligible_stop=[], eligible_motion=[])
        for row, z, eligible, returns, advantage in episodes:
            x, actions, legal = z['observations'], z['actions'], z['masks']
            normalized = np.zeros(len(actions), np.float32)
            normalized[eligible] = (advantage[eligible]-mean)/(std+1e-8)
            stop = actions == 0
            motion_available = legal[:, 1:].any(1)
            waiting = eligible & stop
            categories = dict(eligible_stop_with_motion=waiting,
                              eligible_motion=eligible & ~stop,
                              only_stop_available=~motion_available,
                              outside_support_stop_with_motion=~eligible & motion_available & stop,
                              outside_support_motion=~eligible & ~stop)
            reason_counts = Counter(str(v) for v in z['per_action_guard_reasons'][~legal])
            item = dict(batch=number, source=row['rollout_path'], safe_success=row['safe_success'],
                        release_frame=row['release_frame'], full_navigation_case=row['full_navigation_case'],
                        sampled_decisions=len(actions),
                        category_counts={k:int(v.sum()) for k,v in categories.items()},
                        eligible_stop_advantage=describe(normalized[waiting]),
                        eligible_motion_advantage=describe(normalized[eligible & ~stop]),
                        stopped_with_motion_available=int((stop & motion_available).sum()),
                        outside_support_stop_sensor_valid=int((categories['outside_support_stop_with_motion']
                            & (x[:,435] > .5) & (x[:,438] > .5)).sum()),
                        rejected_action_reasons=dict(reason_counts),
                        terminal_credit_at_first_decision=float(row['reward_terms']['terminal'] * .99**(len(actions)-1)),
                        recorded_old_gae_not_used_as_complete_batch_advantage=True)
            episode_results.append(item)
            result['eligible_stop'].extend(normalized[waiting].tolist())
            result['eligible_motion'].extend(normalized[eligible & ~stop].tolist())
            arrays.append(dict(batch=np.full(len(actions), number), actions=actions,
                               eligible=eligible, complete_returns=returns,
                               raw_advantages=advantage, normalized_advantages=normalized))
        for name in ('eligible_stop', 'eligible_motion'):
            result[name] = describe(result[name])
        batch_results.append(result)
    assert sum(r['sampled_decisions'] for r in episode_results) == report['new_ppo_steps']
    output_arrays = {k:np.concatenate([r[k] for r in arrays]) for k in arrays[0]}
    np.savez_compressed(HERE / 'reconstructed_credit.npz', **output_arrays)
    (HERE / 'TRAINING_REPORT_SNAPSHOT.json').write_bytes(raw)
    result = dict(completed=True, at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
        actual_decisions=report['new_ppo_steps'], complete_episodes=len(episode_results),
        exact_current_guard_reason_groups=True, executed_history_and_masks_verified=True,
        short_contact_floor_verified_frames=contact_frames,
        physical_history_age_checked_frames=checked_history,
        complete_batch_advantage_normalization_reproduced=True,
        no_actor_credit_assigned_to_only_stop_states=True,
        batches=batch_results, episodes=episode_results,
        inputs={str(p.relative_to(ROOT)):digest(p) for p in inputs},
        output_credit_sha256=digest(HERE / 'reconstructed_credit.npz'),
        model_not_loaded=True, new_optimizer_updates=0, new_physical_episodes=0,
        sampling_behavior_log_probability_replay_performed_by_original_learner=True,
        this_audit_does_not_independently_replay_neural_log_probabilities=True,
        signs_are_learning_signal_not_causal_action_quality_labels=True,
        goal_achieved=False, deployment_approved=False)
    (HERE / 'RESULTS.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ['actual_decisions','complete_episodes','batches',
          'physical_history_age_checked_frames']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
