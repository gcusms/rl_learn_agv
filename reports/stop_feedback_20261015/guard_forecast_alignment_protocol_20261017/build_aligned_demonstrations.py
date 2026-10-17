"""Causally rebuild v2 features from prior successful development references.

This creates imitation data only. Source failed episodes, frozen future seeds,
and initialization-prefix choices are excluded from the training rows. No
policy or simulator is run, and no model is trained or promoted here.
"""
from datetime import datetime
import hashlib
import json
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

from causal_history import CausalHistory
from forecast_alignment import contract, fuse_current_frame
from neural_navigation.experiment_budget import ExperimentBudget


ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / 'reports/stop_feedback_20261015'
HERE = Path(__file__).resolve().parent


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    output = HERE / 'ALIGNED_DEMONSTRATIONS.json'
    archive_path = HERE / 'aligned_demonstrations.npz'
    assert not output.exists() and not archive_path.exists()
    source = BASE / 'normal_suite_wait_validation_v2_20261016/RESULTS.json'
    report = json.loads(source.read_text())
    assert report['completed']
    assert report['selected_model'] == 'models/stop_feedback_20261015/normal_suite_wait_pair_v2_s231016/wait_continuation'
    rows = [r for group in report['groups'].values() for r in group['rows'] if r['safe_success']]
    assert len(rows) == 29
    assert not any(r['case_id'].startswith('pre_ppo_holdout_') for r in rows)
    budget = ExperimentBudget(.1, memory_mib=550, directory=HERE)
    paths = [Path(__file__), HERE/'forecast_alignment.py', HERE/'causal_history.py', source,
             ROOT/'training/neural_navigation/long_risk_observation.py']
    parts, source_rows = [], []
    for episode, row in enumerate(rows):
        assert not any(row[k] for k in ['collisions', 'false_arrival', 'physical_guards', 'timeout'])
        path = ROOT / row['trace_path']
        assert digest(path) == row['trace_sha256']
        paths.append(path)
        with np.load(path, allow_pickle=False) as saved:
            old = saved['observations']
            diagnostics = saved['constraint_diagnostics']
            times = saved['labels'][:, 0]
            actions, masks = saved['choices'], saved['masks']
            assert old.shape == (len(times), 625) and diagnostics.shape == (len(times), 15)
            assert masks[np.arange(len(times)), actions].all()
            selected = np.ones(len(times), bool)
            prefix = int(row.get('frozen_prefix_frames', 0))
            if row.get('recovery_test'):
                assert prefix > 0 and 'model_controlled' in saved
                selected = saved['model_controlled'].copy()
                np.testing.assert_array_equal(selected, np.arange(len(times)) >= prefix)
            raw_history, fused_history = CausalHistory(), CausalHistory()
            new = old.copy()
            for index, timestamp in enumerate(times):
                # First prove this source is a continuous history with the
                # declared frame timing; deduplicated unordered inputs cannot
                # be safely treated as a physical temporal sequence.
                reconstructed = raw_history.append(old[index, 467:481], float(timestamp))
                np.testing.assert_array_equal(reconstructed, old[index, 439:481])
                valid = bool(old[index, 435] > .5 and old[index, 438] > .5)
                fused = fuse_current_frame(old[index, 467:481], diagnostics[index, 3:],
                    int(diagnostics[index, 0]), float(timestamp), float(timestamp),
                    input_valid=valid, scan_valid=valid)
                new[index, 439:481] = fused_history.append(fused, float(timestamp))
                if index % 100 == 0:
                    budget.tick()
            np.testing.assert_array_equal(new[:, :439], old[:, :439])
            np.testing.assert_array_equal(new[:, 481:], old[:, 481:])
            count = int(selected.sum())
            parts.append(dict(observations=new[selected], original_observations=old[selected],
                actions=actions[selected], masks=masks[selected], sensor_times=times[selected],
                episode_id=np.full(count, episode, np.int32), frame_index=np.flatnonzero(selected)))
            source_rows.append(dict(case_id=row['case_id'], source=str(path.relative_to(ROOT)),
                sha256=digest(path), replayed_frames=len(times), excluded_initialization_frames=prefix,
                admitted_training_frames=count, changed_feature_frames=int(np.any(new[selected] != old[selected], axis=1).sum()),
                reused_development_reference_not_new_success=True))
    arrays = {key: np.concatenate([part[key] for part in parts]) for key in parts[0]}
    duplicate_choices = {}
    conflicts = []
    for index, (observation, action) in enumerate(zip(arrays['observations'], arrays['actions'])):
        key = observation.tobytes()
        if key in duplicate_choices and duplicate_choices[key] != int(action):
            conflicts.append(index)
        else:
            duplicate_choices[key] = int(action)
    # Conflicting exact inputs must be inspected before any imitation fitting.
    approved = not conflicts
    np.savez_compressed(archive_path, **arrays)
    budget.tick()
    result = dict(completed=True, at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
        contract=contract(), source_model=report['selected_model'], source_rows=source_rows,
        source_episodes=len(rows), admitted_training_rows=len(arrays['actions']),
        exact_unique_observations=len(duplicate_choices), conflicting_exact_inputs=conflicts,
        data_preflight_passed=approved,
        stop_labels=int((arrays['actions'] == 0).sum()),
        invalid_action_labels=0, initialization_commands_excluded=True,
        raw_old_history_exactly_reconstructed=True, only_risk_history_slice_changed=True,
        frozen_seed_evaluation_data_excluded=True, failed_episodes_excluded=True,
        archive=str(archive_path.relative_to(ROOT)), archive_sha256=digest(archive_path),
        new_physical_steps=0, new_ppo_steps=0, new_optimizer_updates=0,
        no_new_model_or_runtime_integration=True, no_navigation_gain_claim=True,
        goal_achieved=False, deployment_approved=False,
        inputs={str(p.relative_to(ROOT)): digest(p) for p in set(paths)}, resource=budget.statistics())
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({k: result[k] for k in ['source_episodes','admitted_training_rows','stop_labels',
        'conflicting_exact_inputs','data_preflight_passed','resource']}))


if __name__ == '__main__':
    main()
