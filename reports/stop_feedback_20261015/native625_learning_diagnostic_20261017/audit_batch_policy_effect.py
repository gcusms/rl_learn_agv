"""Measure actual batch-PPO policy changes without any new environment steps."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from neural_navigation.experiment_budget import ExperimentBudget
from neural_navigation.masked_actions import ACTIONS, masks_from_observation


ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / 'reports/stop_feedback_20261015'
HERE = Path(__file__).resolve().parent


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    output = HERE / 'BATCH_POLICY_EFFECT_AUDIT.json'
    assert not output.exists()
    torch.set_num_threads(1)
    budget = ExperimentBudget(.1, memory_mib=1600, directory=HERE)
    parent = ROOT / 'models/stop_feedback_20261015/normal_suite_wait_pair_v2_s231016/wait_continuation/portable/actor.pt'
    candidate = ROOT / 'models/stop_feedback_20261015/native625_batch_credit_s231017/portable/actor.pt'
    batch = BASE / 'native625_batch_credit_pilot_20261017/collection_batch.npz'
    before_trace = BASE / 'normal_suite_wait_validation_v2_20261016/physical/hard3_w11_training_stall_t106.npz'
    after_trace = BASE / 'native625_batch_credit_validation_20261017/physical/hard3_w11_training_stall_t106.npz'
    files = [Path(__file__), parent, candidate, batch, before_trace, after_trace,
        ROOT / 'training/neural_navigation/masked_actions.py']
    with np.load(batch, allow_pickle=False) as saved:
        collection = saved['observations'].copy()
        actions = saved['actions'].copy()
        advantage = saved['advantages'].copy()
    with np.load(before_trace, allow_pickle=False) as old, np.load(after_trace, allow_pickle=False) as new:
        paired = {key: bool(np.array_equal(old[key], new[key], equal_nan=True)) for key in
            ['observations', 'masks', 'choices', 'commands', 'scans', 'odometry', 'measured_velocity', 'labels']}
        late = new['observations'][530:].copy()
    datasets = {'collection': collection, 'actual106_suffix': late}
    predictions = {}
    for name, path in [('parent', parent), ('candidate', candidate)]:
        actor = torch.jit.load(str(path), map_location='cpu').eval()
        predictions[name] = {}
        with torch.no_grad():
            for label, x in datasets.items():
                choices, probabilities, support = [], [], []
                for start in range(0, len(x), 64):
                    part = x[start:start+64]
                    tensor = torch.from_numpy(part)
                    masks = torch.from_numpy(masks_from_observation(part[:, :282]))
                    logits = actor.logits(tensor).masked_fill(~masks, -1e8) / 4.
                    supported = actor.support(tensor)
                    output_actions = actor(tensor).numpy()
                    selected = np.abs(output_actions[:, None] - ACTIONS[None]).max(2).argmin(1)
                    probs = logits.softmax(1).numpy()
                    frozen = ~supported.numpy()
                    probs[frozen] = 0.
                    probs[np.flatnonzero(frozen), selected[frozen]] = 1.
                    choices.extend(selected.tolist())
                    probabilities.extend(probs.tolist())
                    support.extend(supported.numpy().tolist())
                    budget.tick()
                predictions[name][label] = dict(choices=np.asarray(choices),
                    probabilities=np.asarray(probabilities), support=np.asarray(support))
        del actor
    rows = {}
    for label, x in datasets.items():
        old, new = predictions['parent'][label], predictions['candidate'][label]
        masks = masks_from_observation(x[:, :282])
        eligible = old['support'] & masks[:, 1:].any(1)
        assert np.array_equal(old['support'], new['support'])
        p, q = old['probabilities'][eligible], new['probabilities'][eligible]
        row = dict(frames=len(x), actor_eligible_frames=int(eligible.sum()),
            greedy_changes=int((old['choices'] != new['choices']).sum()),
            parent_greedy_stop_frames=int((old['choices'] == 0).sum()),
            candidate_greedy_stop_frames=int((new['choices'] == 0).sum()),
            mean_parent_stop_probability=float(p[:, 0].mean()),
            mean_candidate_stop_probability=float(q[:, 0].mean()),
            mean_kl_parent_to_candidate=float((p * (np.log(p.clip(1e-30)) - np.log(q.clip(1e-30)))).sum(1).mean()),
            maximum_probability_change=float(abs(p-q).max()),
            probability_temperature=4., logged_observations_only_not_counterfactual_physics=True)
        if label == 'collection':
            ids = np.flatnonzero(eligible)
            before = old['probabilities'][ids, actions[ids]]
            after = new['probabilities'][ids, actions[ids]]
            ratio = after / before
            gains = np.minimum(advantage[ids] * ratio, advantage[ids] * ratio.clip(.8, 1.2)) - advantage[ids]
            row.update(logged_complete_batch_clipped_surrogate_gain=float(gains.mean()),
                positive_advantage_samples=int((advantage[ids] > 0).sum()),
                negative_advantage_samples=int((advantage[ids] < 0).sum()))
        rows[label] = row
    result = dict(completed=True, datasets=rows, actual106_paired_trace_arrays_identical=paired,
        actor_probabilities_changed=rows['collection']['maximum_probability_change'] > 0,
        no_new_physics=True, no_optimizer_updates=True, new_ppo_steps=0,
        no_claim_that_surrogate_gain_proves_navigation_gain=True,
        inputs={str(p.relative_to(ROOT)): digest(p) for p in files}, resource=budget.statistics(),
        goal_achieved=False, deployment_approved=False)
    with output.open('x') as handle:
        json.dump(result, handle, indent=2)
        handle.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k != 'inputs'}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
