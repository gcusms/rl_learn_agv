"""Replay completed traces and measure recovery-head feedback sensitivity.

No counterfactual action is executed or admitted as a training label. Ablating
head inputs while retaining the original support and hard mask is only an
information-use diagnostic, not a safe maneuver or retraining comparison.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import torch

from experiments.long_risk_actor import digest
from experiments.release_actor import ReleaseActor
from neural_navigation.decision_feedback import FEEDBACK_REASONS
from neural_navigation.experiment_budget import ExperimentBudget
from neural_navigation.masked_actions import ACTIONS
from neural_navigation.release_observation import augment
from neural_navigation.sensors import LidarSpec


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'reports/stop_feedback_20261015'
ABLATIONS = {
    'full_context': None,
    'previous_stop_reason_none': slice(54, 74),
    'without_idle_timers': slice(74, 76),
    'without_long_risk_history': slice(0, 42),
    'without_mask_features_hard_mask_retained': slice(42, 54),
    'without_current_ranges': slice(88, 160),
    'without_current_scan_quality': slice(160, 304),
    'without_waypoint_and_goal': slice(304, 308),
    'without_measured_velocity_and_execution': slice(308, 312),
}


def write(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def action_ids(commands):
    commands = np.asarray(commands)
    distances = np.abs(commands[:, None] - ACTIONS[None]).max(2)
    choices = distances.argmin(1)
    assert np.all(distances[np.arange(len(choices)), choices] < 1e-6)
    return choices


def main(args):
    output, model = args.output.resolve(), args.model.resolve()
    pilot_path, regression_path = args.pilot.resolve(), args.regression.resolve()
    pilot, regression = [json.loads(p.read_text()) for p in [pilot_path, regression_path]]
    assert pilot['completed'] and regression['completed']
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    budget = ExperimentBudget(.15, memory_mib=2048, directory=output)
    rows = [('pilot', row, True) for row in pilot['rows'] if row['variant'] == 'candidate']
    rows += [(group, row, True) for group, value in regression['groups'].items()
        if not value.get('reused', False) for row in value['rows']]
    older_path = BASE / 'release_phase_validation_20261016/RESULTS.json'
    older = json.loads(older_path.read_text())
    rows += [('older_recorded_context', next(row for row in older['rows']
        if row['case_id'] == 'long_risk_sensor_05_1' and row['variant'] == 'recovery_phase'), False)]
    files = [Path(__file__), pilot_path, regression_path, older_path,
        model / 'checkpoint.pt', model / 'parent_actor.pt', model / 'portable/manifest.json',
        model / 'portable/actor.pt', model / 'portable/hardware.json',
        ROOT / 'training/experiments/release_actor.py',
        ROOT / 'training/experiments/long_risk_actor.py',
        ROOT / 'training/neural_navigation/decision_feedback.py']
    files += [ROOT / row['trace_path'] for _, row, _ in rows]
    hashes = {str(p.relative_to(ROOT)): digest(p) for p in set(files)}
    write(output / 'PLAN.json', dict(at=datetime.now(timezone.utc).isoformat(), inputs=hashes,
        ablations=list(ABLATIONS), fixed_support_parent_and_hard_mask=True,
        counterfactuals_not_executed_or_admitted_as_labels=True,
        no_simulator_truth_in_policy_input=True, new_optimizer_updates=0, new_physical_steps=0))
    actor = ReleaseActor(torch.jit.load(str(model / 'parent_actor.pt'), map_location='cpu')).eval()
    actor.load_state_dict(torch.load(model / 'checkpoint.pt', map_location='cpu', weights_only=False)['actor'])
    portable = torch.jit.load(str(model / 'portable/actor.pt'), map_location='cpu').eval()
    lidar = LidarSpec(json.loads((model / 'portable/hardware.json').read_text())['lidar'])
    reports, replayed = [], 0
    for group, row, expected_current in rows:
        path = ROOT / row['trace_path']
        assert digest(path) == row['trace_sha256'], str(path)
        with np.load(path, allow_pickle=False) as archive:
            data = {key:archive[key].copy() for key in archive.files}
        observations = data['observations']
        if observations.shape[1] == 481:
            observations = np.stack([augment(obs, scan, lidar)
                for obs, scan in zip(observations, data['scans'])])
        assert observations.shape == (len(data['choices']), 625)
        np.testing.assert_array_equal(observations[:, 481:],
            np.stack([augment(obs[:481], scan, lidar)[481:]
                for obs, scan in zip(observations, data['scans'])]))
        controlled = data.get('model_controlled', np.ones(len(observations), dtype=bool)).astype(bool)
        results = {name:dict(choices=[], pstop=[]) for name in ABLATIONS}
        parents, active_parts = [], []
        for start in range(0, len(observations), 64):
            tensor = torch.from_numpy(observations[start:start+64])
            with torch.no_grad():
                features, parent = actor.features(tensor)
                masks, active = actor.masks(tensor), actor.support(tensor)
                actual = actor(tensor)
                np.testing.assert_array_equal(portable(tensor).numpy(), actual.numpy())
                np.testing.assert_array_equal(masks.numpy(), data['masks'][start:start+len(tensor)])
                parents.extend(action_ids(parent.numpy()).tolist())
                active_parts.extend(active.numpy().tolist())
                for name, removed in ABLATIONS.items():
                    changed = features.clone()
                    if removed is not None:
                        changed[:, removed] = 0.
                    if name == 'previous_stop_reason_none':
                        changed[:, 54 + FEEDBACK_REASONS.index('none')] = 1.
                    logits = (actor.head(changed) + 2. * changed[:, 76:88]).masked_fill(~masks, -1e8)
                    choice = logits.argmax(1)
                    command = torch.where(active[:, None], actor.actions[choice], parent)
                    results[name]['choices'].extend(action_ids(command.numpy()).tolist())
                    results[name]['pstop'].extend(logits.softmax(1)[:, 0].numpy().tolist())
                    if name == 'full_context':
                        np.testing.assert_array_equal(command.numpy(), actual.numpy())
            budget.tick()
        parent, active = np.asarray(parents), np.asarray(active_parts, dtype=bool)
        for value in results.values():
            for key in value:
                value[key] = np.asarray(value[key])
        full, pstop = results['full_context']['choices'], results['full_context']['pstop']
        if expected_current:
            np.testing.assert_array_equal(full[controlled], data['choices'][controlled])
            replayed += int(controlled.sum())
        eligible = active & controlled
        legal_motion = data['masks'][:, 1:].any(1)
        wait = controlled & (full == 0) & legal_motion
        previous_reason = observations[:, 399:419].argmax(1)
        reason_counts = Counter(FEEDBACK_REASONS[i] for i in previous_reason[wait])
        sensitivity = {}
        for name, values in results.items():
            difference = np.abs(values['pstop'] - pstop)
            sensitivity[name] = dict(
                head_active_action_changes=int(((values['choices'] != full) & eligible).sum()),
                mean_active_absolute_pstop_change=float(difference[eligible].mean()) if eligible.any() else None,
                maximum_active_absolute_pstop_change=float(difference[eligible].max()) if eligible.any() else None,
                voluntary_wait_choices_on_same_recorded_context=int(((values['choices'] == 0)
                    & controlled & legal_motion).sum()))
        focus = set(np.flatnonzero(controlled)[:1].tolist())
        for subset in [wait, eligible & (parent != full), eligible & (full != 0)
                & (observations[:, 434] >= .4-1e-7)]:
            focus.update(np.flatnonzero(subset)[:4].tolist())
        examples = []
        for i in sorted(focus):
            examples.append(dict(sensor_time_s=float(data['labels'][i, 0]),
                selected_choice=int(full[i]), frozen_parent_choice=int(parent[i]),
                legal_motion_choices=int(data['masks'][i, 1:].sum()),
                previous_executed_reason=FEEDBACK_REASONS[previous_reason[i]],
                previous_idle_with_choices_s=float(observations[i, 434]*5),
                current_prediction_valid=bool(observations[i, 435]>.5),
                choices_under_head_ablations={name:int(value['choices'][i]) for name,value in results.items()},
                pstop_under_head_ablations={name:float(value['pstop'][i]) for name,value in results.items()}))
        reports.append(dict(group=group, case_id=row['case_id'], trace_sha256=row['trace_sha256'],
            current_model_trace=expected_current, frames=len(observations), controlled_frames=int(controlled.sum()),
            active_controlled_frames=int(eligible.sum()),
            observed_physical_result={key:row[key] for key in ['safe_success','seconds','travel_m','collisions','timeout']},
            new_model_voluntary_wait_on_recorded_context=int(wait.sum()),
            only_stop_controlled_frames=int((controlled & ~legal_motion).sum()),
            previous_executed_reason_during_wait=dict(reason_counts),
            changed_vs_parent_controlled=int(((full != parent)&controlled).sum()),
            parent_motion_but_head_stop=int((controlled & (parent != 0)&(full == 0)).sum()),
            parent_stop_but_head_motion=int((controlled & (parent == 0)&(full != 0)).sum()),
            sensitivity=sensitivity, examples=examples))
        write(output / 'STATUS.json', dict(status='offline_feedback_sensitivity',completed_traces=len(reports)))
    for name, expected in hashes.items():
        assert digest(ROOT / name) == expected, name
    budget.tick()
    write(output / 'RESULTS.json',dict(completed=True,at=datetime.now(timezone.utc).isoformat(),
        replayed_current_model_controlled_frames=replayed,reports=reports,
        limitations=['Ablations may be outside the physical data distribution.',
            'Original support and hard masks were fixed, so this tests the head channels only.',
            'Changed outputs establish sensitivity, not correct explanation or safe counterfactual execution.',
            'Older contexts are off-policy observations; their counterfactual choices are not physical results.'],
        new_optimizer_updates=0,new_physical_steps=0,new_ppo_steps=0,
        resource=budget.statistics(),goal_achieved=False,deployment_approved=False))
    write(output / 'STATUS.json',dict(status='complete',diagnostic_only=True))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['model','pilot','regression','output']:
        parser.add_argument('--'+name,type=Path,required=True)
    main(parser.parse_args())
