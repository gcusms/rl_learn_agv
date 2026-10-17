"""Inspect residual supervision errors and common-prefix decisions, read-only."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import torch

from experiments.guard_reason_release_actor import GuardReasonReleaseActor
from experiments.long_risk_actor import digest
from neural_navigation.experiment_budget import ExperimentBudget
from neural_navigation.masked_actions import ACTIONS


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'reports/stop_feedback_20261015'


def load_actor(path):
    actor = GuardReasonReleaseActor(torch.jit.load(str(path / 'parent_actor.pt'), map_location='cpu')).eval()
    actor.load_state_dict(torch.load(path / 'checkpoint.pt', map_location='cpu', weights_only=False)['actor'])
    return actor


def choice(actions):
    distances = np.abs(actions[:, None] - ACTIONS[None]).max(2)
    assert (distances.min(1) < 2e-6).all()
    return distances.argmin(1)


def main(args):
    model, control, dataset, output = [getattr(args, key).resolve()
        for key in ['model', 'control', 'dataset', 'output']]
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    budget = ExperimentBudget(.1, memory_mib=1024, directory=output)
    result = json.loads((model / 'RESULTS.json').read_text())
    assert result['completed'] and digest(model / 'checkpoint.pt') == result['checkpoint_sha256']
    assert digest(dataset) == result['source_sha256']
    first_path = BASE / 'normal_wait_diagnosis_20261016/RESULTS.json'
    first = json.loads(first_path.read_text())
    replay = BASE / 'normal_retention_replay_20261016'
    late = BASE / 'context_broad_diagnostic_20261016/physical/hard3_w11_training_stall_t106.npz'
    inputs = [Path(__file__), model / 'checkpoint.pt', model / 'RESULTS.json',
        control / 'checkpoint.pt', dataset, first_path, replay / 'RESULTS.json', late]
    inputs += list(replay.glob('*.npz'))
    hashes = {str(p.relative_to(ROOT)): digest(p) for p in inputs}
    actor, previous = load_actor(model), load_actor(control)
    with np.load(dataset, allow_pickle=False) as archive:
        arrays = {key: archive[key].copy() for key in archive.files}
    observations, masks, targets = [arrays[key] for key in ['observations', 'masks', 'choices']]
    x = torch.from_numpy(observations)
    with torch.no_grad():
        features, baseline = actor.features(x)
        active = actor.support(x).numpy()
        predictions = choice(actor(x).numpy())
        probability = actor.logits(x).masked_fill(~torch.from_numpy(masks), -1e8).softmax(1).numpy()
    wrong = np.flatnonzero(predictions != targets)
    np.testing.assert_array_equal(wrong, result['wrong_source_indices'])
    features = features.numpy()
    residuals = []
    for i in wrong:
        distances = np.linalg.norm(features - features[i], axis=1)
        distances[(targets != predictions[i]) | (~active)] = np.inf
        nearest = int(distances.argmin())
        residuals.append(dict(index=int(i), source_id=int(arrays['source_ids'][i]),
            sensor_time_s=float(arrays['sensor_times'][i]), target_choice=int(targets[i]),
            predicted_choice=int(predictions[i]), parent_choice=int(choice(baseline[i:i + 1].numpy())[0]),
            target_probability=float(probability[i, targets[i]]),
            predicted_probability=float(probability[i, predictions[i]]),
            legal_actions=np.flatnonzero(masks[i]).tolist(),
            nearest_active_source_with_predicted_label=dict(index=nearest,
                feature_l2_distance=float(distances[nearest]), source_id=int(arrays['source_ids'][nearest]),
                sensor_time_s=float(arrays['sensor_times'][nearest]),
                differing_feature_indices=np.flatnonzero(features[i] != features[nearest]).tolist())))
    rows = []
    for prior in first['rows']:
        path = replay / ('normal_retention_' + prior['case_id'] + '.npz')
        with np.load(path, allow_pickle=False) as archive:
            obs, expected = archive['observations'].copy(), archive['choices'].copy()
        with torch.no_grad():
            tensor = torch.from_numpy(obs)
            before, after = [choice(a(tensor).numpy()) for a in [previous, actor]]
        index = prior['first_different_choice']['frame']
        rows.append(dict(case_id=prior['case_id'], frame=index,
            reference_choice=int(expected[index]), control_choice=int(before[index]),
            candidate_choice=int(after[index]), reference_frames=len(obs),
            candidate_wrong_reference_frames=np.flatnonzero(after != expected).tolist(),
            control_wrong_reference_frame_count=int((before != expected).sum()),
            no_navigation_success_inferred=True))
        budget.tick()
    with np.load(late, allow_pickle=False) as archive:
        obs, masks = archive['observations'][archive['model_controlled']].copy(), archive['masks'][archive['model_controlled']].copy()
    with torch.no_grad():
        tensor = torch.from_numpy(obs)
        before, after = [choice(a(tensor).numpy()) for a in [previous, actor]]
    legal = masks[:, 1:].any(1)
    late_audit = dict(recorded_frames=len(obs), control_legal_wait=int(((before == 0) & legal).sum()),
        candidate_legal_wait=int(((after == 0) & legal).sum()),
        changed_decisions=int((before != after).sum()), no_counterfactual_executed=True)
    for name, expected in hashes.items():
        assert digest(ROOT / name) == expected, name
    budget.tick()
    report = dict(completed=True, at=datetime.now(timezone.utc).isoformat(), inputs=hashes,
        correct_source_labels=int((predictions == targets).sum()), residual_errors=residuals,
        first_divergences=rows, old_actual106_input_audit=late_audit,
        no_new_physics_labels_or_training=True, new_ppo_steps=0,
        source_fit_gate_remains_failed=not result['exported_for_physical_validation'],
        deployment_approved=False, resource=budget.statistics())
    (output / 'RESULTS.json').write_text(json.dumps(report, indent=2) + '\n')
    (output / 'driver_snapshot.py').write_text(Path(__file__).read_text())
    print(json.dumps({k: v for k, v in report.items() if k != 'inputs'}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ['model', 'control', 'dataset', 'output']:
        parser.add_argument('--' + key, type=Path, required=True)
    main(parser.parse_args())
