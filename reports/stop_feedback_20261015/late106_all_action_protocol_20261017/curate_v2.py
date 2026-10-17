"""Prepare matching v2 imitation data only after a verified original-config recovery.

The offline teacher is not a deployed controller and its samples are not PPO
experience. Failed, alternative-hardware and initialization choices are never
admitted. This utility does not fit a model or relax any navigation gate.
"""
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
from zoneinfo import ZoneInfo

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT/'reports/stop_feedback_20261015'
HERE = Path(__file__).resolve().parent
ALIGNMENT = BASE/'guard_forecast_alignment_protocol_20261017'
OUT = BASE/'late106_verified_v2_data_20261017'


def eligible(row):
    return bool(row.get('hardware') == 'original_memory'
        and row.get('safe_success') and row.get('physical_arrival')
        and not any(row.get(key, True) for key in ('false_arrival', 'collisions', 'physical_guards'))
        and np.isfinite(row.get('minimum_dynamic_gap_m', np.nan))
        and row['minimum_dynamic_gap_m'] >= .04-1e-4
        and np.isfinite(row.get('final_xy_error_m', np.nan)) and row['final_xy_error_m'] < .2
        and np.isfinite(row.get('post_release_seconds', np.nan)) and 0. < row['post_release_seconds'] <= 34.+1e-7
        and row.get('original_prefix_sensor_and_physics_exact')
        and row.get('snapshot_two_restores_exact'))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def conflicts(observations, actions):
    previous, rows = {}, []
    for index, (x, action) in enumerate(zip(observations, actions)):
        key = np.ascontiguousarray(x).tobytes()
        if key in previous and previous[key][1] != int(action):
            rows.append(dict(first=previous[key][0], second=index,
                first_action=previous[key][1], second_action=int(action)))
        else:
            previous.setdefault(key, (index, int(action)))
    return rows


def main():
    source = BASE/'late106_all_action_20261017/RESULTS.json'
    report = json.loads(source.read_text())
    if not report['completed']:
        raise ValueError('Exact teacher diagnosis is still running; no dataset created')
    rows = [row for row in report['rows'] if eligible(row)]
    if not rows:
        print(json.dumps(dict(dataset_created=False, reason='No complete safe original-hardware teacher recovery',
            alternative_hardware_requires_its_own_matched_dataset=True, new_optimizer_updates=0)))
        return
    if OUT.exists():
        raise ValueError('Do not overwrite prior data evidence')
    sys.path.insert(0, str(ALIGNMENT))
    from causal_history import CausalHistory
    from forecast_alignment import fuse_current_frame

    paths = [Path(__file__), source, ALIGNMENT/'aligned_demonstrations.npz',
        ALIGNMENT/'causal_history.py', ALIGNMENT/'forecast_alignment.py']
    original = ROOT/'models/stop_feedback_20261015/normal_suite_wait_pair_v2_s231016/wait_continuation/portable'
    parent = ROOT/'models/stop_feedback_20261015/guard_aligned_migration_s231017/portable'
    for name in ('hardware.json', 'docking_contract.json'):
        assert digest(original/name) == digest(parent/name), name
        paths += [original/name, parent/name]
    hardware = json.loads((parent/'hardware.json').read_text())
    commands = np.asarray(hardware['policy_action']['choices'])
    with np.load(ALIGNMENT/'aligned_demonstrations.npz', allow_pickle=False) as saved:
        assert len(saved['actions']) == 3799 and int((saved['actions'] == 0).sum()) == 557
        parts = [dict(observations=saved['observations'].copy(), actions=saved['actions'].copy(),
            masks=saved['masks'].copy(), sensor_times=saved['sensor_times'].copy(),
            source_id=np.zeros(len(saved['actions']), np.int32), frame_index=saved['frame_index'].copy())]
    teacher_rows = []
    for episode, row in enumerate(rows, 1):
        path = ROOT/row['trace_path']
        assert digest(path) == row['trace_sha256']
        paths.append(path)
        with np.load(path, allow_pickle=False) as saved:
            old, actions, masks = saved['observations'], saved['choices'], saved['masks']
            times, short = saved['labels'][:, 0], saved['constraint_diagnostics']
            n = len(times)
            selected = saved['teacher_controlled']
            assert old.shape == (n, 625) and short.shape == (n, 15)
            assert 530 < n <= 701 and np.all(np.diff(times) > 0.) and times[-1] <= 140.+1e-7
            assert saved['labels'][-1, 7] < .2
            np.testing.assert_allclose(times[-1], 106.+row['post_release_seconds'], rtol=0, atol=2e-8)
            np.testing.assert_array_equal(selected, np.arange(n) >= 530)
            np.testing.assert_array_equal(masks, np.column_stack([np.ones(n, bool), old[:, 235:244] > .5, old[:, 267:269] > .5]))
            assert masks[np.arange(n), actions].all()
            np.testing.assert_allclose(saved['commands'], commands[actions], rtol=0, atol=2e-8)
            raw_history, fused_history = CausalHistory(), CausalHistory()
            new = old.copy()
            for index, timestamp in enumerate(times):
                np.testing.assert_array_equal(raw_history.append(old[index, 467:481], timestamp), old[index, 439:481])
                valid = bool(old[index, 435] > .5 and old[index, 438] > .5)
                frame = fuse_current_frame(old[index, 467:481], short[index, 3:], int(short[index, 0]),
                    timestamp, timestamp, input_valid=valid, scan_valid=valid)
                new[index, 439:481] = fused_history.append(frame, timestamp)
            np.testing.assert_array_equal(new[:, :439], old[:, :439])
            np.testing.assert_array_equal(new[:, 481:], old[:, 481:])
            count = int(selected.sum())
            parts.append(dict(observations=new[selected], actions=actions[selected], masks=masks[selected],
                sensor_times=times[selected], source_id=np.full(count, episode, np.int32), frame_index=np.flatnonzero(selected)))
            teacher_rows.append(dict(trace=str(path.relative_to(ROOT)), sha256=digest(path),
                initialization_frames_excluded=530, admitted_teacher_rows=count, hardware='original_memory'))
    arrays = {key: np.concatenate([part[key] for part in parts]) for key in parts[0]}
    ambiguous = conflicts(arrays['observations'], arrays['actions'])
    OUT.mkdir()
    archive = OUT/'demonstrations.npz'
    np.savez_compressed(archive, **arrays)
    result = dict(completed=True, at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
        preparation_only=True, source_references=3799, teacher_rows=teacher_rows,
        total_rows=len(arrays['actions']), stop_labels=int((arrays['actions'] == 0).sum()),
        conflicting_exact_inputs=ambiguous, data_preflight_passed=not ambiguous,
        original_hardware_and_4cm_margin=True, matching_guard_aligned_v2_features=True,
        failed_trajectories_and_initialization_excluded=True, teacher_is_imitation_not_on_policy_ppo=True,
        no_teacher_in_deployment=True, new_optimizer_updates=0, new_ppo_steps=0, new_physical_steps=0,
        learned_recovery_not_yet_proven=True, goal_achieved=False, deployment_approved=False,
        archive_sha256=digest(archive), inputs={str(p.relative_to(ROOT)): digest(p) for p in paths})
    (OUT/'RESULTS.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('total_rows', 'stop_labels', 'data_preflight_passed', 'conflicting_exact_inputs')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
