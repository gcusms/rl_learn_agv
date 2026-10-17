"""Inspect native625 sampling on recorded late states; no physical trial."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from experiments.guard_reason_release_actor import GuardReasonReleaseActor
from experiments.native625_onpolicy import Native625OnPolicy
from neural_navigation.experiment_budget import ExperimentBudget
from neural_navigation.masked_actions import ACTIONS


ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / 'reports/stop_feedback_20261015'
HERE = Path(__file__).resolve().parent


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    output = HERE / 'LATE_EXPLORATION_PROFILE.json'
    assert not output.exists(), 'Retain the original pretraining profile'
    torch.set_num_threads(1)
    model = ROOT / 'models/stop_feedback_20261015/normal_suite_wait_pair_v2_s231016/wait_continuation'
    fit = json.loads((model / 'RESULTS.json').read_text())
    assert digest(model / 'checkpoint.pt') == fit['checkpoint_sha256']
    actual = BASE / 'normal_suite_wait_validation_v2_20261016/physical/hard3_w11_training_stall_t106.npz'
    source = BASE / 'context_broad_diagnostic_20261016/physical/hard3_w11_training_stall_t106.npz'
    budget = ExperimentBudget(.1, memory_mib=1100, directory=HERE)
    actor = GuardReasonReleaseActor(torch.jit.load(str(model / 'parent_actor.pt'), map_location='cpu')).eval()
    actor.load_state_dict(torch.load(model / 'checkpoint.pt', map_location='cpu', weights_only=False)['actor'])
    learner = Native625OnPolicy(actor, temperature=4.)
    with np.load(actual, allow_pickle=False) as trace:
        indices = np.flatnonzero(trace['model_controlled'])
        x, masks, choices = [trace[name][indices].copy() for name in ['observations', 'masks', 'choices']]
    assert len(x) == 170
    with torch.no_grad():
        np.testing.assert_array_equal(actor(torch.from_numpy(x)).numpy(), ACTIONS[choices])
        terms = learner.evaluate(x, masks)
        probabilities = terms['distribution'].probs.numpy()
    move = 1.-probabilities[:, 0]
    statistics = dict(recorded_frames=len(x), eligible_actor_frames=int(terms['eligible'].sum()),
        only_stop_frames=int((~masks[:, 1:].any(1)).sum()),
        greedy_stop_frames=int((choices==0).sum()),
        move_probability_quantiles=dict(zip(['minimum','p25','median','p75','maximum'],
            np.quantile(move,[0.,.25,.5,.75,1.]).tolist())),
        sum_move_probability_over_fixed_recorded_states=float(move.sum()),
        probability_sum_is_not_a_prediction_of_actual_new_rollout=True,
        legal_action_sets=np.unique(masks,axis=0).astype(int).tolist())
    with np.load(source, allow_pickle=False) as trace:
        releases = np.array([450,480,530,572])
        initial_x = trace['observations'][releases].copy()
        initial_masks = trace['masks'][releases].copy()
    with torch.no_grad():
        initial = learner.evaluate(initial_x, initial_masks)
        initial_probabilities = initial['distribution'].probs.numpy()
    rows = [dict(prefix_frames=int(frame), release_s=float(frame)*.2,
        eligible_actor=bool(initial['eligible'][i]),
        legal_actions=np.flatnonzero(initial_masks[i]).tolist(),
        sampling_probabilities=initial_probabilities[i].tolist()) for i,frame in enumerate(releases)]
    budget.tick()
    files = [Path(__file__), actual, source, model / 'checkpoint.pt', model / 'parent_actor.pt',
        model / 'RESULTS.json', ROOT / 'training/experiments/native625_onpolicy.py',
        ROOT / 'training/experiments/guard_reason_release_actor.py']
    result = dict(completed=True, model=str(model.relative_to(ROOT)), temperature=4.,
        late_actual_recorded_states=statistics, curriculum_initial_states=rows,
        inputs={str(path.relative_to(ROOT)):digest(path) for path in files},
        resource=budget.statistics(), no_new_physics=True, no_optimizer_updates=True,
        reference_head_created_for_inspection_only=True,
        greedy_export_rule_unchanged=True, new_ppo_steps=0,
        sampling_opportunity_not_proof_of_recovery=True, goal_achieved=False, deployment_approved=False)
    temporary = output.with_suffix('.tmp')
    temporary.write_text(json.dumps(result, indent=2)+'\n')
    temporary.replace(output)
    print(json.dumps(statistics), flush=True)
    print(json.dumps(rows), flush=True)


if __name__ == '__main__':
    main()
