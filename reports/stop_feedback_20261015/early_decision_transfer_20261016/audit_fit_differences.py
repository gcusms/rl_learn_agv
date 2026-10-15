"""Explain the completed offline candidate's original-label disagreements.

No environment, replay commands, optimizer steps or threshold changes. Nearby
observations are a numerical diagnostic, never proof of state equivalence.
"""
from datetime import datetime
import hashlib
import json
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import torch

from experiments.stop_recovery_policy import StopRecoveryPPO
from neural_navigation.decision_feedback import REASON_SLICE, RISK_SLICE, STAT_SLICE, FEEDBACK_REASONS
from neural_navigation.experiment_budget import ExperimentBudget


ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    destination = OUT / 'FIT_DIFFERENCES.json'
    if destination.exists():
        raise ValueError('Do not overwrite a completed audit')
    folder = ROOT / 'models/stop_feedback_20261015/early_transfer_bc800_s231015'
    status = json.loads((folder / 'status.json').read_text())
    result = json.loads((folder / 'RESULTS.json').read_text())
    if status['status'] != 'offline_bc_complete' or not result['completed']:
        raise ValueError('Completed, independently saved offline fit required')
    torch.set_num_threads(1)
    budget = ExperimentBudget(.05, memory_mib=2048, directory=OUT)
    with np.load(folder / 'demonstrations.npz', allow_pickle=False) as data:
        obs = data['observations'].copy()
        masks = data['masks'].astype(bool)
        labels = data['choices'].copy()
    sources = json.loads((OUT / 'RESULTS.json').read_text())['sources']
    source_indices = np.concatenate([np.full(source['recovery_frames'], i) for i, source in enumerate(sources)])
    if len(labels) != 2355 or len(source_indices) != len(labels):
        raise ValueError('The audited ordered 2355-label dataset is required')
    reference_path = ROOT / 'models/stop_feedback_20261015/w11_recovery_bc800_s231015/policy_bc.zip'
    predictions = {}
    for name, path in [('reference', reference_path), ('candidate', folder / 'policy_bc.zip')]:
        model = StopRecoveryPPO.load(path, device='cpu')
        values = []
        budget.tick()
        with torch.no_grad():
            for first in range(0, len(labels), 125):
                values.append(model.policy.get_distribution(torch.from_numpy(obs[first:first+125]),
                    masks[first:first+125]).distribution.probs.numpy())
                budget.tick()
        predictions[name] = np.concatenate(values)
        del model
    before, after = predictions['reference'], predictions['candidate']
    current = after.argmax(-1)
    wrong = np.flatnonzero(current != labels)
    if len(wrong) != len(labels) - result['correct_examples']:
        raise ValueError('Reloaded checkpoint disagrees with the completed fit report')
    examples = []
    new_indices = np.arange(2089, len(labels))
    for i in wrong:
        source_n = int(source_indices[i])
        source = sources[source_n]
        same_mask = np.all(masks[new_indices] == masks[i], axis=-1)
        opposite_label = labels[new_indices] != labels[i]
        selected = new_indices[same_mask & opposite_label]
        nearby = None
        if len(selected):
            delta = obs[selected] - obs[i]
            j = int(selected[np.argmin(np.linalg.norm(delta, axis=-1))])
            distance = obs[j] - obs[i]
            nearby = dict(index=j, target=int(labels[j]),
                source_id=sources[int(source_indices[j])]['id'],
                observation_l2=float(np.linalg.norm(distance)),
                observation_max_abs=float(np.abs(distance).max()),
                base300_l2=float(np.linalg.norm(distance[:300])),
                feedback136_l2=float(np.linalg.norm(distance[300:])),
                exactly_equal=bool(np.array_equal(obs[i], obs[j])),
                candidate_target_probability=float(after[j, labels[j]]))
        reason_slice = obs[i, 300+REASON_SLICE.start:300+REASON_SLICE.stop]
        stats = obs[i, 300+STAT_SLICE.start:300+STAT_SLICE.stop]
        example = dict(index=int(i), original_example=bool(i < 2089),
            source_id=source['id'], source_report=source['report'],
            source_kept_offset=int(i-np.flatnonzero(source_indices == source_n)[0]),
            target=int(labels[i]), candidate_choice=int(current[i]),
            reference_choice=int(before[i].argmax()),
            target_probability_before=float(before[i, labels[i]]),
            target_probability_after=float(after[i, labels[i]]),
            chosen_probability_after=float(after[i, current[i]]),
            legal_actions=np.flatnonzero(masks[i]).tolist(),
            candidate_choice_legal=bool(masks[i, current[i]]),
            motion_became_stop=bool(labels[i] != 0 and current[i] == 0),
            stop_became_motion=bool(labels[i] == 0 and current[i] != 0),
            previous_executed_reason=FEEDBACK_REASONS[int(reason_slice.argmax())],
            prior_executed_idle_s=float(stats[2]*5.),
            prior_executed_idle_with_motion_s=float(stats[3]*5.),
            predictor_valid=bool(stats[4] > .5),
            nearest_pooled_lidar_m=float(obs[i, 144:216].min()*12.),
            target_forecast_risk=float(obs[i, 300+RISK_SLICE.start+labels[i]]),
            selected_forecast_risk=float(obs[i, 300+RISK_SLICE.start+current[i]]),
            nearest_new_different_label_same_mask=nearby)
        examples.append(example)
        budget.tick()
    changed_new = np.arange(len(labels)) >= 2089
    positive_transfer = changed_new & (before.argmax(-1) == 0) & (labels != 0) & (current == labels)
    audit = dict(completed=True, at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
        complete_fit_result=result, disagreements=examples,
        original_disagreements=sum(bool(row['original_example']) for row in examples),
        motion_to_wait=sum(row['motion_became_stop'] for row in examples),
        wait_to_motion=sum(row['stop_became_motion'] for row in examples),
        new_demonstrated_stop_to_motion_corrected=int(positive_transfer.sum()),
        original_admission_gate_still_failed=True,
        additional_environment_steps=0, weights_or_runtime_changed=False,
        source_sha256=digest(Path(__file__)),
        inputs_sha256={str(path.relative_to(ROOT)): digest(path) for path in [
            folder/'policy_bc.zip', folder/'RESULTS.json', folder/'demonstrations.npz',
            OUT/'RESULTS.json', reference_path]},
        resource=budget.statistics(),
        limitations=[
            'Different legal actions can be useful; offline label disagreement is not physical failure.',
            'Numerically nearby observations do not prove observational aliasing or incompatible labels.',
            'Correcting seven demonstrated stop decisions does not prove closed-loop recovery.',
            'The predeclared all-label quick-test admission remains failed; no threshold is relaxed here.'
        ], goal_achieved=False, deployment_approved=False)
    destination.write_text(json.dumps(audit, indent=2)+'\n')
    print(json.dumps(dict(completed=True, disagreements=[{
        key: row[key] for key in ['index','source_id','target','candidate_choice',
                                  'motion_became_stop','target_probability_after']}
        for row in examples], corrected_new_stop_to_motion=int(positive_transfer.sum()),
        original_admission_gate_still_failed=True, resource=audit['resource']), indent=2), flush=True)


if __name__ == '__main__':
    main()
