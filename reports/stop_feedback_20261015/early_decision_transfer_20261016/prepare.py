"""Audit two complete PPO successes as optional early-decision examples.

This creates data only. Existing paired training, guard settings and deployment
are untouched. Reused development tasks are explicitly retired as holdouts.
"""
from collections import Counter
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
from zoneinfo import ZoneInfo

import numpy as np
import torch

from experiments.recovery_demonstrations import load_demonstrations
from experiments.stop_recovery_policy import StopRecoveryPPO
from neural_navigation.experiment_budget import ExperimentBudget


ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
BASE = Path('reports/stop_feedback_20261015')
BC = Path('models/stop_feedback_20261015/w11_recovery_bc800_s231015')
SOURCE = BASE / 'proactive_augmented_ppo8000_navigation_20261016'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def main():
    os.chdir(ROOT)
    if (OUT / 'RESULTS.json').exists() or (OUT / 'sources').exists():
        raise ValueError('A fresh data audit is required')
    torch.set_num_threads(1)
    budget = ExperimentBudget(.05, memory_mib=2500, directory=OUT)
    report_path = SOURCE / 'RESULTS.json'
    report = json.loads(report_path.read_text())
    if not report['completed']:
        raise ValueError('Only completed physical reports may supply examples')
    selected = ['recovery_holdout_00', 'recovery_holdout_01']
    rows = []
    inputs = {str(report_path): digest(report_path), str(OUT.relative_to(ROOT) / 'prepare.py'): digest(Path(__file__))}
    sources = OUT / 'sources'
    sources.mkdir()
    for case_id in selected:
        matches = [r for r in report['rows'] if r['case_id'] == case_id]
        if len(matches) != 1:
            raise ValueError('Exactly one declared source trajectory required')
        row = matches[0].copy()
        if (not row['safe_success'] or not row['physical_arrival'] or row['false_arrival']
                or row['collisions'] or row['physical_guards'] or row['timeout']):
            raise ValueError('Do not imitate a failed source')
        original = SOURCE / (case_id + '_' + row['variant'] + '.npz')
        if digest(original) != row['trace_sha256']:
            raise ValueError('Source trace differs from its physical outcome')
        name = 'development_' + case_id + '_ppo_success'
        target = sources / (name + '.npz')
        shutil.copy2(original, target)
        row.update(specification=dict(id=name), historical_category=row['category'],
            category='reused_development_training_candidate',
            source_report=str(report_path), source_trace=str(original),
            retired_as_unseen_or_holdout_task=True)
        rows.append(row)
        inputs[str(original)] = digest(original)
    wrapper = dict(completed=True, rows=rows, source_report=str(report_path),
        source_report_sha256=digest(report_path),
        not_a_new_evaluation=True, no_timed_teacher=True,
        scope='Already evaluated PPO successes, selected for an optional new training dataset.')
    write(sources / 'RESULTS.json', wrapper)
    specifications_path = BASE / 'w11_success_augmented_protocol_20261016/demonstrations.json'
    specifications = json.loads(specifications_path.read_text())
    specifications += [dict(report=str((sources / 'RESULTS.json').relative_to(ROOT)),
        id=row['specification']['id']) for row in rows]
    write(OUT / 'demonstrations.json', specifications)
    for path in [specifications_path, BC / 'policy_bc.zip', BC / 'demonstrations.npz',
                 Path('training/experiments/recovery_demonstrations.py'),
                 Path('training/experiments/proactive_recovery.py')]:
        inputs[str(path)] = digest(path)
    write(OUT / 'PLAN.json', dict(created_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
        hypothesis='Complete successful early maneuvers may teach avoidance before a close-range deadlock.',
        selected_development_cases=selected, inputs=inputs, soft_cpu_cores=.05,
        training_started=False, new_physical_steps=0, current_comparison_changed=False,
        deployment_approved=False))
    policy = StopRecoveryPPO.load(BC / 'policy_bc.zip', device='cpu')
    budget.tick()
    observations, masks, choices, used = load_demonstrations(specifications, policy.policy)
    budget.tick()
    with np.load(BC / 'demonstrations.npz', allow_pickle=False) as old:
        n = len(old['choices'])
        for key, value in [('observations', observations), ('masks', masks), ('choices', choices)]:
            np.testing.assert_array_equal(value[:n], old[key])
        if n != 2089 or np.count_nonzero(old['choices'] == 0) != 37:
            raise ValueError('The exact verified 2089-label initialization is required')
    if len(choices) <= n:
        raise ValueError('The new sources supplied no trainable examples')
    seen = {}
    conflicts = []
    for index, (observation, mask, choice) in enumerate(zip(observations, masks, choices)):
        key = hashlib.sha256(observation.tobytes() + mask.tobytes()).digest()
        earlier = seen.setdefault(key, (int(choice), index))
        if earlier[0] != int(choice):
            conflicts.append(dict(first_index=earlier[1], later_index=index,
                first_choice=earlier[0], later_choice=int(choice)))
    new_observations, new_masks, new_choices = observations[n:], masks[n:], choices[n:]
    with torch.no_grad():
        probabilities = policy.policy.get_distribution(torch.from_numpy(new_observations), new_masks).distribution.probs.numpy()
    budget.tick()
    greedy = probabilities.argmax(-1)
    transitions = Counter()
    for before, target in zip(greedy, new_choices):
        transitions['stop_to_motion' if before == 0 and target != 0 else
            'motion_to_wait' if before != 0 and target == 0 else
            'different_motion' if before != target else 'same_choice'] += 1
    np.savez_compressed(OUT / 'candidate_demonstrations.npz', observations=observations,
        masks=masks, choices=choices)
    result = dict(completed=True, training_started=False, environment_steps=0,
        existing_examples=n, existing_wait_examples_retained=37, existing_prefix_bit_identical=True,
        additional_examples=len(choices)-n, total_examples=len(choices),
        new_stop_examples=int(np.count_nonzero(new_choices == 0)),
        new_movement_examples=int(np.count_nonzero(new_choices)),
        new_label_counts={str(k): v for k, v in Counter(map(int, new_choices)).items()},
        new_target_changes_from_bc=dict(transitions), exact_observation_conflicts=conflicts,
        minimum_current_target_probability=float(probabilities[np.arange(len(new_choices)), new_choices].min()),
        sources=used, all_new_sources_complete_and_physically_safe=True,
        candidate_data_sha256=digest(OUT / 'candidate_demonstrations.npz'),
        input_hashes=inputs, resource=budget.statistics(),
        inference_teacher_added=False, current_training_or_guard_changed=False,
        training_admitted=False,
        reason_training_not_automatically_admitted='Finish the independent paired PPO and geometry evidence before selecting the next factor.',
        limitations=['Two reused successful development traces do not establish generalization.',
            'Old files named holdout00/01 become training tasks if this dataset is used.',
            'A legal demonstrated action and label disagreement do not establish that the new model can complete navigation.',
            'No new network training, simulator steps or real robot commands were performed.'],
        goal_achieved=False, deployment_approved=False)
    write(OUT / 'RESULTS.json', result)
    print(json.dumps({k: v for k, v in result.items() if k not in ('sources', 'input_hashes')}, indent=2))


if __name__ == '__main__':
    main()
