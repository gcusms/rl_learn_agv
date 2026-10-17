"""Compare shared inputs, then independently reconstruct the declared v1->v2 risk change.

The initial auditor incorrectly required identity between two explicitly different
42-dimensional risk contracts. Keep that audit and its failure; this script never
changes physical results, commands, permissions, deadlines, or acceptance gates.
"""
from datetime import datetime
import hashlib
import json
from pathlib import Path
from zoneinfo import ZoneInfo
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
BASE = Path(__file__).resolve().parent.parent
OUT = BASE / 'deployment_acceptance_supplement_20261017'


def read(path):
    return json.loads(path.read_text())


def main():
    result = read(OUT / 'RESULTS.json')
    assert len(result['rows']) == 2, 'Both physical checks must have actually finished'
    current = read(ROOT / result['model'] / 'portable/manifest.json')
    old = read(ROOT / 'models/stop_feedback_20261015/normal_suite_wait_pair_v2_s231016/wait_continuation/portable/manifest.json')
    a, b = current['observation_contract'], old['observation_contract']
    assert a['schema'] == 'risk_and_measured_release_context_v2_guard_aligned'
    assert b['schema'] == 'risk_and_measured_release_context_v1'
    assert a['parent_contract']['schema'] == 'long_action_risk_history_v2_guard_lower_bound'
    assert b['parent_contract']['schema'] == 'long_action_risk_history_v1'
    cfg = current['action_contract']['predictive_constraint']
    short_denominator = cfg['horizon_s'] + cfg['period_s']
    long_denominator = a['parent_contract']['horizon_s'] + a['parent_contract']['forecast_period_s']
    assert a['parent_contract']['history_ages_s'] == [.4, .2, 0.]
    records, prefixes = [], []
    for row in result['rows']:
        job = read(OUT / 'jobs' / (row['case_id'] + '.json'))
        n = row['frozen_prefix_frames']
        trace = ROOT / row['trace_path']
        assert hashlib.sha256(trace.read_bytes()).hexdigest() == row['trace_sha256']
        with np.load(trace, allow_pickle=False) as actual, np.load(ROOT / job['spec']['trace'], allow_pickle=False) as ref:
            essential = ['scans', 'odometry', 'measured_velocity', 'commands', 'choices', 'masks', 'human_positions']
            for name in essential:
                np.testing.assert_array_equal(actual[name][:n], ref[name][:n])
            np.testing.assert_allclose(actual['labels'][:n + 1, :4], ref['labels'][:n + 1, :4], rtol=0, atol=2e-8)
            obs, old_obs = actual['observations'][:n], ref['observations'][:n]
            assert obs.shape == old_obs.shape == (n, 625)
            np.testing.assert_array_equal(obs[:, :439], old_obs[:, :439])
            np.testing.assert_array_equal(obs[:, 481:], old_obs[:, 481:])
            np.testing.assert_allclose(np.diff(actual['labels'][:n, 0]), .2, rtol=0, atol=1e-8)
            # Current v1 prediction is in the last14 of the old42D history.
            # Its only declared v2 change is a lower bound from the measured
            # short-guard contact scores already present in unchanged input.
            frames = old_obs[:, 467:481].astype(np.float64).copy()
            for i in range(n):
                valid = obs[i, 435] > .5 and obs[i, 438] > .5
                contacts = obs[i, 419:431].astype(np.float64)
                supported = contacts > 0
                if valid and supported.any():
                    bound = 1 - (1 - contacts[supported]) * short_denominator / long_denominator
                    frames[i, :12][supported] = np.maximum(frames[i, :12][supported], bound)
                    frames[i, 12] = 1
                    frames[i, 13] = max(frames[i, 13], float(obs[i, 436]))
            expected = np.zeros((n, 3, 14), np.float64)
            expected[:, 2] = frames
            expected[1:, 1] = frames[:-1]
            expected[2:, 0] = frames[:-2]
            # Float32 score storage loses <1ulp when recovering a contact time.
            np.testing.assert_allclose(obs[:, 439:481], expected.reshape(n, 42), rtol=0, atol=2e-7)
            assert not actual['model_controlled'][:n].any()
            assert actual['model_controlled'][n:].all()
            waits = (actual['choices'][n:] == 0) & actual['masks'][n:, 1:].any(axis=1)
            only_stop = ~actual['masks'][n:, 1:].any(axis=1)
            assert int(waits.sum()) == row['post_release_wait_with_choices_frames']
            assert int(only_stop.sum()) == row['post_release_only_stop_frames']
            changed = np.flatnonzero(np.any(obs != old_obs, axis=0))
            records.append(dict(case_id=row['case_id'], prefix_frames=n,
                                exact_sensor_motion_permission_arrays=essential,
                                exact_same_semantics_observation_dimensions=583,
                                changed_observation_columns=changed.tolist(),
                                changed_columns_only_declared_439_to_480=True,
                                v2_risk_reconstructed_from_v1_and_measured_short_guard=True,
                                maximum_v2_risk_reconstruction_error=float(np.max(np.abs(obs[:, 439:481] - expected.reshape(n, 42)))),
                                physical_prefix_tolerance_m_rad=2e-8,
                                learned_control_after_release_verified=True,
                                wait_counts_independently_verified=True,
                                not_full_mission_score=True))
            prefixes.append(dict(observations=obs.copy(), choices=actual['choices'][:n].copy()))
    common = min(len(p['choices']) for p in prefixes)
    different = np.flatnonzero(prefixes[0]['choices'][:common] != prefixes[1]['choices'][:common])
    common = int(different[0]) if len(different) else common
    if common:
        np.testing.assert_array_equal(prefixes[0]['observations'][:common], prefixes[1]['observations'][:common])
    audit = dict(completed=True, at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
                 correction='Original auditor required full625 identity across explicitly different v1/v2 contracts; original auditor/failure preserved. Physical gates unchanged.',
                 original_observation_schema=b['schema'], current_observation_schema=a['schema'],
                 records=records, shared_current_prefix_full625_exact_frames=common,
                 independent_source_initialization_checks_passed=True,
                 new_ppo_steps=0, new_training_updates=0,
                 no_model_runtime_guard_dynamics_deadline_changes=True,
                 original106_failure_not_waived=True, deployment_approved=False)
    (OUT / 'VERSIONED_TRACE_AUDIT.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(audit, ensure_ascii=False))


if __name__ == '__main__':
    main()
