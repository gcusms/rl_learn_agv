"""Read-only audit of complete stage-one rollouts and saved PPO checkpoints."""
from collections import Counter
import gc
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from experiments.guard_reason_release_actor import GuardReasonReleaseActor
from experiments.native625_onpolicy import Native625OnPolicy
from neural_navigation.decision_feedback import GROUPS
from neural_navigation.experiment_budget import ExperimentBudget


ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / 'reports/stop_feedback_20261015'
HERE = Path(__file__).resolve().parent
TRAINING = BASE / 'native625_suite_ppo_20261016'
MODELS = ROOT / 'models/stop_feedback_20261015/native625_suite_ppo_s231017'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def runs(values):
    edges = np.diff(np.r_[False, values, False].astype(np.int8))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def main():
    output = HERE / 'FIRST_STAGE_AUDIT.json'
    assert not output.exists(), 'Do not replace an existing immutable audit'
    torch.set_num_threads(1)
    budget = ExperimentBudget(.1, memory_mib=1200, directory=HERE)
    report = json.loads((TRAINING / 'RESULTS.json').read_text())
    rows = report['episodes'][:8]
    assert len(rows) == 8 and all(r['release_frames'] == 450 for r in rows)
    snapshot = HERE / 'STAGE_ONE_ROWS.json'
    assert not snapshot.exists()
    snapshot.write_text(json.dumps(rows, indent=2) + '\n')
    plan = json.loads((TRAINING / 'PLAN.json').read_text())
    parent_validation = json.loads((BASE / 'normal_suite_wait_validation_v2_20261016/RESULTS.json').read_text())
    parent = ROOT / parent_validation['selected_model']
    assert digest(parent / 'checkpoint.pt') == plan['inputs'][str((parent / 'checkpoint.pt').relative_to(ROOT))]
    actor = GuardReasonReleaseActor(torch.jit.load(str(parent / 'parent_actor.pt'), map_location='cpu')).eval()
    learner = Native625OnPolicy(actor, temperature=4.)
    files = [Path(__file__), snapshot, TRAINING / 'PLAN.json', parent / 'checkpoint.pt', parent / 'parent_actor.pt']
    initial = None
    results = []
    with torch.no_grad():
        for index, row in enumerate(rows):
            checkpoint = parent / 'checkpoint.pt' if index == 0 else MODELS / f'episode_{index:02d}.pt'
            state = torch.load(checkpoint, map_location='cpu', weights_only=False)
            learner.actor.load_state_dict(state['actor'])
            if index:
                learner.critic.load_state_dict(state['critic'])
            files.append(checkpoint)
            trace = ROOT / row['rollout_path']
            assert digest(trace) == row['rollout_sha256']
            files.append(trace)
            with np.load(trace, allow_pickle=False) as archive:
                data = {k: archive[k].copy() for k in archive.files}
            x, mask, actions = [data[k] for k in ['observations', 'masks', 'actions']]
            if initial is None:
                initial = x[0].copy()
            np.testing.assert_array_equal(initial, x[0])
            probabilities, values, eligibility = [], [], []
            for start in range(0, len(x), 64):
                terms = learner.evaluate(x[start:start+64], mask[start:start+64], actions[start:start+64])
                probabilities.extend(terms['log_probability'].numpy())
                values.extend(terms['values'].numpy())
                eligibility.extend(terms['eligible'].numpy())
                budget.tick()
            log_error = float(abs(np.asarray(probabilities)-data['old_log_probability']).max())
            value_error = float(abs(np.asarray(values)-data['old_values']).max())
            np.testing.assert_allclose(probabilities, data['old_log_probability'], rtol=0, atol=2e-5)
            np.testing.assert_allclose(values, data['old_values'], rtol=0, atol=2e-5)
            assert int(np.sum(eligibility)) == row['eligible_actor_decisions']
            groups = x[:, 303:399].reshape(-1, 12, 8).argmax(2)
            np.testing.assert_array_equal(groups[:, 1:] == 0, mask[:, 1:])
            blocked = ~mask[:, 1:].any(1)
            spans = runs(blocked)
            sustained = [(a,b) for a,b in spans if b-a >= 25]
            at = sustained[0][0] if sustained else None
            reason_counts = Counter(GROUPS[int(g)] for g in groups[blocked, 1:].ravel())
            first_terms = learner.evaluate(initial[None])
            terminal_reward = row['reward_terms']['terminal']
            discounted = float(sum(float(r) * .99 ** i for i, r in enumerate(data['rewards'])))
            result = dict(episode=row['episode'], safe_success=row['safe_success'],
                frames=len(x), only_stop_frames=int(blocked.sum()),
                voluntary_wait_frames=row['voluntary_wait_frames'],
                longest_only_stop_seconds=max((b-a for a,b in spans), default=0)*.2,
                first_5s_only_stop_at_s=None if at is None else float(data['labels'][at,0]),
                first_5s_only_stop_after_decisions=None if at is None else int(at),
                rejection_group_counts_when_only_stop=dict(reason_counts),
                first20_actions=actions[:20].tolist(), initial_advantage=float(data['advantages'][0]),
                initial_value=float(data['old_values'][0]), discounted_episode_return=discounted,
                initial_monte_carlo_advantage=discounted-float(data['old_values'][0]),
                terminal_direct_gae_contribution_at_first_action=float(terminal_reward*(.99*.95)**(len(x)-1)),
                terminal_discounted_return_contribution_at_first_action=float(terminal_reward*.99**(len(x)-1)),
                initial_action_probabilities=first_terms['distribution'].probs[0].tolist(),
                maximum_old_log_probability_error=log_error, maximum_old_value_error=value_error,
                final_xy_error_m=row['final_xy_error_m'])
            results.append(result)
            del state, data
            budget.tick()
        final_path = MODELS / 'episode_08.pt'
        final = torch.load(final_path, map_location='cpu', weights_only=False)
        learner.actor.load_state_dict(final['actor'])
        files.append(final_path)
        final_probabilities = learner.evaluate(initial[None])['distribution'].probs[0].tolist()
        # The exact anchor path is selected from the pinned plan, never from a newer source.
        candidates = [ROOT / k for k in plan['inputs'] if k.endswith('.npz') and '/physical/' not in k]
        assert len(candidates) == 1, candidates
        source = candidates[0]
        assert digest(source) == plan['inputs'][str(source.relative_to(ROOT))]
        files.append(source)
        with np.load(source, allow_pickle=False) as archive:
            anchors, labels = archive['observations'].copy(), archive['choices'].copy()
        predictions = []
        for start in range(0,len(anchors),128):
            output_actions = learner.actor(torch.from_numpy(anchors[start:start+128])).numpy()
            predictions.extend(abs(output_actions[:,None]-learner.actor.actions.numpy()[None]).max(2).argmin(1))
            budget.tick()
        matches = np.asarray(predictions) == labels
        retention = dict(labels=len(labels), matches=int(matches.sum()),
            stop_labels=int((labels==0).sum()), stop_mismatches=int((~matches & (labels==0)).sum()),
            motion_mismatches=int((~matches & (labels!=0)).sum()),
            wrong_indices=np.flatnonzero(~matches).tolist(), source_fit_is_not_physical_acceptance=True)
    gc.collect()
    budget.tick()
    result = dict(completed=True, episodes=results, initial_state_byte_exact_all_episodes=True,
        final_stage_initial_action_probabilities=final_probabilities, stage8_source_retention=retention,
        training_safe_arrivals=sum(r['safe_success'] for r in rows),
        stage_training_not_independent_validation=True, no_causal_claim_from_advantage_contrast_alone=True,
        inputs={str(p.relative_to(ROOT)):digest(p) for p in set(files)}, resource=budget.statistics(),
        no_new_physics=True, no_optimizer_updates=True, new_ppo_steps=0,
        goal_achieved=False, deployment_approved=False)
    output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['inputs','episodes']},ensure_ascii=False),flush=True)


if __name__ == '__main__':
    main()
