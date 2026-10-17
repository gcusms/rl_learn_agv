"""Compare a completed paired fit on fixed recorded states, without physics."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from experiments.guard_reason_release_actor import GuardReasonReleaseActor
from neural_navigation.experiment_budget import ExperimentBudget
from neural_navigation.masked_actions import ACTIONS


ROOT = Path(__file__).resolve().parents[3]
OUTPUT = Path(__file__).resolve().parent
BASE = ROOT / 'reports/stop_feedback_20261015'
FITS = ROOT / 'models/stop_feedback_20261015/normal_suite_wait_pair_v2_s231016'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    report_path = OUTPUT / 'PAIR_OFFLINE_AUDIT.json'
    if report_path.exists():
        raise ValueError('Audit is immutable; do not overwrite or rerun silently')
    fit = json.loads((FITS / 'RESULTS.json').read_text())
    assert fit['completed'], 'Wait for both fixed-update fits to finish'
    torch.set_num_threads(1)
    budget = ExperimentBudget(.1, memory_mib=1100, directory=OUTPUT)
    dataset = OUTPUT / 'demonstrations.npz'
    with np.load(dataset, allow_pickle=False) as source:
        observations, targets = source['observations'].copy(), source['choices'].copy()
    states = [
        ('dynamic_first_fork_8_6s', BASE / 'full_mission_retention_validation_20261016/physical/normal_other16_independent_dynamic_000.npz', 43),
        ('actual_learner_wait_35s', BASE / 'learner_wait_continuation_20261016/continuation625.npz', 0),
        ('old_late_wait_106s', BASE / 'context_broad_diagnostic_20261016/physical/hard3_w11_training_stall_t106.npz', 530),
    ]
    examples = []
    for name, path, index in states:
        with np.load(path, allow_pickle=False) as trace:
            examples.append(trace['observations'][index].copy())
    examples = torch.from_numpy(np.asarray(examples))
    models = [ROOT / 'models/stop_feedback_20261015/full_mission_retention_pair_s231016/retained500']
    models += [FITS / row['model'] for row in fit['rows']]
    inputs = [Path(__file__), dataset, FITS / 'RESULTS.json'] + [s[1] for s in states]
    results = []
    for folder in models:
        metadata = json.loads((folder / 'RESULTS.json').read_text())
        assert digest(folder / 'checkpoint.pt') == metadata['checkpoint_sha256']
        inputs += [folder / name for name in ['RESULTS.json', 'checkpoint.pt', 'parent_actor.pt']]
        actor = GuardReasonReleaseActor(torch.jit.load(str(folder / 'parent_actor.pt'), map_location='cpu')).eval()
        actor.load_state_dict(torch.load(folder / 'checkpoint.pt', map_location='cpu', weights_only=False)['actor'])
        with torch.no_grad():
            decisions = actor(torch.from_numpy(observations)).numpy()
            matches = (decisions == ACTIONS[targets]).all(1)
            features, baseline = actor.features(examples)
            supported, masks = actor.support(examples), actor.masks(examples)
            greedy = actor(examples).numpy()
            logits = actor.head(features) + 2. * features[:, 76:88]
            probabilities = [torch.softmax((logits / temperature).masked_fill(~masks, -1e8), dim=1)
                for temperature in [1., 4.]]
        if folder.parent == FITS:
            assert int(matches.sum()) == metadata['correct_source_labels']
        rows = []
        for index, (name, path, source_index) in enumerate(states):
            rows.append(dict(state=name, recorded_trace=str(path.relative_to(ROOT)),
                recorded_index=source_index, support=bool(supported[index]),
                legal_actions=np.flatnonzero(masks[index].numpy()).tolist(),
                greedy_choice=int(np.abs(ACTIONS-greedy[index]).max(1).argmin()),
                parent_choice=int(np.abs(ACTIONS-baseline[index].numpy()).max(1).argmin()),
                temperature1_head_probabilities=probabilities[0][index].numpy().tolist(),
                temperature4_head_probabilities=probabilities[1][index].numpy().tolist(),
                probabilities_apply_only_when_support_true=True))
        results.append(dict(model=str(folder.relative_to(ROOT)),
            source_correct=int(matches.sum()), source_total=len(matches),
            original4949_correct=int(matches[:4949].sum()),
            all_reference_stop_labels_retained=bool(matches[targets==0].all()),
            wrong_source_indices=np.flatnonzero(~matches).tolist(), states=rows))
        del actor
        budget.tick()
    hashes = {str(path.relative_to(ROOT)): digest(path) for path in set(inputs)}
    result = dict(completed=True, rows=results, inputs=hashes, resource=budget.statistics(),
        recorded_states_not_counterfactual_navigation=True, no_new_physics=True,
        no_optimizer_updates=True, new_ppo_steps=0, goal_achieved=False, deployment_approved=False)
    temporary = report_path.with_suffix('.tmp')
    temporary.write_text(json.dumps(result, indent=2)+'\n')
    temporary.replace(report_path)
    print(json.dumps([dict(model=row['model'], source_correct=row['source_correct'],
        choices={s['state']:s['greedy_choice'] for s in row['states']}) for row in results]), flush=True)


if __name__ == '__main__':
    main()
