"""Reconstruct a complete successful reference after a full-mission regression.

Only recorded scan, odometry, velocity and timestamps enter the runtime. The
reference's complete 439D inputs, choices, masks and commands must reproduce.
The resulting 625D trace is diagnostic data, never a new candidate success.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import torch

from experiments.guard_reason_release_actor import GuardReasonReleaseActor
from experiments.long_risk_actor import digest
from experiments.refit_phase_tail_pair import write
from neural_navigation.experiment_budget import ExperimentBudget
from neural_navigation.masked_actions import ACTIONS, masks_from_observation
from neural_navigation.release_runtime import ReleaseRuntime


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'reports/stop_feedback_20261015'


def arrays(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key].copy() for key in archive.files}


def main(output):
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    budget = ExperimentBudget(.1, memory_mib=1400, directory=output)
    tasks_path = BASE / 'context_feedback_longfit_protocol_20261016/tasks.json'
    old_report_path = BASE / 'recovery_followup_broad_regression_20261016/RESULTS.json'
    new_report_path = BASE / 'sensor_retention_validation_20261016/RESULTS.json'
    tasks = json.loads(tasks_path.read_text())
    source_report = json.loads(old_report_path.read_text())
    new_report = json.loads(new_report_path.read_text())
    assert source_report['completed'] and new_report['completed']
    case_id = 'train_failure_w11_e7'
    case = next(c for c in tasks['normal18'] if c['id'] == case_id)
    old_row = next(r for r in source_report['full18_rows'] if r['case_id'] == case_id)
    new_row = next(r for r in new_report['groups']['normal_other16']['rows'] if r['case_id'] == case_id)
    assert old_row == next(r for r in tasks['reference']['normal18'] if r['case_id'] == case_id)
    for row in (old_row, new_row):
        assert row['safe_success'] and row['physical_arrival'] and row['protocol_valid']
        assert not any(row[k] for k in ('collisions', 'false_arrival', 'physical_guards', 'timeout'))
    old_path, new_path = (ROOT / row['trace_path'] for row in (old_row, new_row))
    for row, path in ((old_row, old_path), (new_row, new_path)):
        assert digest(path) == row['trace_sha256']
    model = ROOT / 'models/stop_feedback_20261015/sensor_retention_pair_s231016/retained500'
    bundle = model / 'portable'
    teacher_bundle = ROOT / 'models/stop_feedback_20261015/recovery_followup_bc400_s231016/bc_portable'
    teacher_path = teacher_bundle / 'actor.pt'
    dataset_path = BASE / 'sensor_retention_protocol_20261016/demonstrations.npz'
    manifest = json.loads((bundle / 'manifest.json').read_text())
    files = [Path(__file__), tasks_path, old_report_path, new_report_path, old_path,
        new_path, model / 'parent_actor.pt', model / 'checkpoint.pt', teacher_path,
        teacher_bundle / 'hardware.json', dataset_path, bundle / 'manifest.json']
    files += [bundle / name for name in manifest['files']]
    for folder in ['neural_navigation', 'navrl']:
        files += list((ROOT / 'training' / folder).rglob('*.py'))
    for name in ['guard_reason_release_actor', 'context_release_actor', 'release_actor', 'long_risk_actor']:
        files.append(ROOT / ('training/experiments/' + name + '.py'))
    files += [p for p in (ROOT / 'training' / case['scene']).iterdir() if p.is_file()]
    hashes = {str(p.relative_to(ROOT)): digest(p) for p in set(files)}
    write(output / 'PLAN.json', dict(inputs=hashes, case=case,
        maximum_recorded_frames=old_row['frames'], cpu_soft_cores=.1,
        old_source_is_complete_safe_full_mission=True,
        original439_prefix_masks_choices_commands_must_reproduce=True,
        raw_sensor_inputs_only=True, no_truth_or_future_inputs=True,
        no_new_physics_or_optimization=True, new_ppo_steps=0, deployment_approved=False))
    old, new = arrays(old_path), arrays(new_path)
    assert old['observations'].shape == (186, 439)
    assert old['labels'][0, 0] == new['labels'][0, 0] == 0.
    fork = int(np.flatnonzero(old['choices'][:len(old['choices'])] != new['choices'][:len(old['choices'])])[0])
    for key in ['scans', 'odometry', 'measured_velocity', 'masks']:
        np.testing.assert_allclose(old[key][:fork+1], new[key][:fork+1], rtol=0, atol=2e-8, equal_nan=True)
    np.testing.assert_array_equal(old['observations'][:fork+1], new['observations'][:fork+1, :439])
    np.testing.assert_allclose(old['commands'][:fork], new['commands'][:fork], rtol=0, atol=2e-8)
    reference = torch.jit.load(str(teacher_path), map_location='cpu').eval()
    candidate = GuardReasonReleaseActor(torch.jit.load(str(model / 'parent_actor.pt'), map_location='cpu')).eval()
    saved = torch.load(model / 'checkpoint.pt', map_location='cpu', weights_only=False)
    candidate.load_state_dict(saved['actor'])
    index = [0]

    class ReferenceReplay:
        def predict(self, observation):
            with torch.no_grad():
                action = reference(torch.from_numpy(np.asarray(observation[:439], np.float32)[None])).numpy()[0]
            np.testing.assert_array_equal(action, ACTIONS[old['choices'][index[0]]])
            return action.copy()

    runtime = ReleaseRuntime(bundle, ROOT / 'training' / case['scene'] / 'decision_map.yaml',
        case['start'], case['goal'], 'shortcut')
    runtime.policy.policy = ReferenceReplay()
    observations = []
    maximum = 0.
    for frame in range(len(old['choices'])):
        index[0] = frame
        result = runtime.step(old['scans'][frame], old['odometry'][frame],
            old['measured_velocity'][frame], float(old['labels'][frame, 0]), capture_observation=True)
        obs = np.asarray(result['policy_observation'], np.float32)
        np.testing.assert_allclose(obs[:439], old['observations'][frame], rtol=0, atol=2e-6)
        np.testing.assert_array_equal(masks_from_observation(obs[:282]), old['masks'][frame])
        np.testing.assert_array_equal(result['proposed_action'], ACTIONS[old['choices'][frame]])
        np.testing.assert_allclose(result['command'], old['commands'][frame], rtol=0, atol=2e-8)
        maximum = max(maximum, float(np.max(abs(obs[:439] - old['observations'][frame]))))
        observations.append(obs)
        if frame % 25 == 0:
            write(output / 'STATUS.json', dict(status='complete_reference_sensor_replay', matched_frames=frame+1,
                maximum_frames=len(old['choices'])))
        budget.tick()
    assert result['stop_reason'] == 'estimated_goal_reached'
    observations = np.asarray(observations)
    np.testing.assert_array_equal(observations[:fork+1], new['observations'][:fork+1])
    with torch.no_grad():
        x = torch.from_numpy(observations)
        features, _ = candidate.features(x)
        logits = candidate.logits(x).masked_fill(~candidate.masks(x), -1e8)
        probability = logits.softmax(-1)
        choices = np.abs(candidate(x).numpy()[:, None] - ACTIONS[None]).max(2).argmin(1)
        # Only an offline component ablation; this never issues a command.
        f = features[fork:fork+1]
        old_projection = candidate.head[0].previous(f[:, :312])
        reason_projection = candidate.head[0].current_reason(f[:, 312:])
        ablated = candidate.head[1:](old_projection) + 2. * f[:, 76:88]
        full = candidate.head[1:](old_projection + reason_projection) + 2. * f[:, 76:88]
        data = arrays(dataset_path)
        source_features, _ = candidate.features(torch.from_numpy(data['observations']))
        distances = torch.linalg.vector_norm(source_features - f, dim=1).numpy()
    nearest = np.argsort(distances)[:8]
    trace = dict(old)
    trace['original439_observations'] = trace['observations']
    trace['observations'] = observations
    path = output / 'complete_reference625.npz'
    np.savez_compressed(path, **trace)
    original_choice = int(old['choices'][fork])
    stopped_choice = int(new['choices'][fork])
    report = dict(completed=True, at=datetime.now(timezone.utc).isoformat(),
        source_frames=len(observations), maximum439_input_difference=maximum,
        complete_reference_masks_actions_commands_reproduced=True,
        source_trace_path=str(path.relative_to(ROOT)), source_trace_sha256=digest(path),
        old_complete_safe_source_row=old_row, new_complete_safe_but_inefficient_row=new_row,
        source_is_reused_not_new_success=True, automatic_training_admission=False,
        first_identical_input_fork=dict(frame=fork, seconds=float(old['labels'][fork, 0]),
            full625_observation_identical=True, all_raw_sensor_inputs_and_prior_execution_identical=True,
            reference_choice=original_choice, candidate_choice=stopped_choice,
            legal_actions=np.flatnonzero(old['masks'][fork]).tolist(),
            current_forecast=observations[fork, 419:431].tolist(),
            legal_idle_seconds=float(observations[fork, 434] * 5),
            source_label_probability=float(probability[fork, original_choice]),
            candidate_stop_probability=float(probability[fork, 0]),
            current_reason_ablation_choice=int(ablated.argmax(1)[0]),
            full_choice=int(full.argmax(1)[0]),
            reason_projection_l2=float(torch.linalg.vector_norm(reason_projection)),
            nearest_training_inputs=[dict(index=int(i), distance=float(distances[i]),
                choice=int(data['choices'][i]), source_id=int(data['source_ids'][i])) for i in nearest]),
        candidate_mismatch_on_recorded_reference_inputs=int(np.count_nonzero(choices != old['choices'])),
        no_counterfactual_navigation_success_inferred=True,
        new_physical_steps=0, new_optimizer_updates=0, new_ppo_steps=0,
        goal_achieved=False, deployment_approved=False, resource=budget.statistics())
    for name, expected in hashes.items():
        assert digest(ROOT / name) == expected, name
    report['input_hashes_verified'] = len(hashes)
    budget.tick()
    report['resource'] = budget.statistics()
    write(output / 'RESULTS.json', report)
    write(output / 'STATUS.json', dict(status='complete'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    try:
        main(args.output)
    except Exception as error:
        if args.output.is_dir():
            write(args.output / 'FAILURE.json', dict(error=repr(error), automatically_restarted=False))
        raise
