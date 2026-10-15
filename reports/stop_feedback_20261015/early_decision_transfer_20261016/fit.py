"""Fit the audited data from the exact original migration, with no environment.

This is a separate offline initialization candidate, not extra PPO experience.
It cannot invoke reset/step or issue commands. Physical validation remains a
separate resource-gated experiment; this script never promotes a model.
"""
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import zipfile
from zoneinfo import ZoneInfo

import numpy as np
import torch

from experiments.experiment_provenance import record_provenance
from experiments.recovery_demonstrations import initialize_from_demonstrations
from experiments.stop_recovery_policy import StopRecoveryPPO, export_recovery
from experiments.train_prefix_recovery import frozen_base_digest
from neural_navigation.experiment_budget import ExperimentBudget


ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stamp():
    return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()


def write(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def main():
    os.chdir(ROOT)
    plan = json.loads((OUT / 'BC_PLAN.json').read_text())
    target = Path(plan['model_output'])
    if target.exists():
        raise ValueError('A fresh model output is required')
    for name, expected in plan['inputs'].items():
        if digest(Path(name)) != expected:
            raise ValueError('Audited initialization source changed: ' + name)
    audit = json.loads((OUT / 'RESULTS.json').read_text())
    if (not audit['completed'] or audit['total_examples'] != 2355
            or audit['exact_observation_conflicts'] or not audit['existing_prefix_bit_identical']
            or not audit['all_new_sources_complete_and_physically_safe']):
        raise ValueError('Complete safe and conflict-free candidate data is required')
    target.mkdir(parents=True)
    torch.set_num_threads(1)
    budget = ExperimentBudget(plan['soft_cpu_cores'], memory_mib=2048, directory=target)
    write(target / 'status.json', dict(status='offline_bc_training', at=stamp(),
        maximum_supervised_updates=plan['maximum_supervised_updates'], additional_ppo_steps=0,
        environment_created=False, physical_validation_completed=False))
    with zipfile.ZipFile(plan['initial_checkpoint']) as a, zipfile.ZipFile(plan['reference_migrated_checkpoint']) as b:
        same = {name: hashlib.sha256(a.read(name)).hexdigest()
            for name in ('policy.pth', 'policy.optimizer.pth')}
        for name, expected in same.items():
            if hashlib.sha256(b.read(name)).hexdigest() != expected:
                raise ValueError('Initial policy or optimizer differs from original BC migration')
    write(target / 'INITIALIZATION_PARITY.json', dict(completed=True,
        compared_before_supervised_updates=True, policy_and_optimizer_byte_identical=True,
        initial_components_sha256=same, baseline_supervision_changed_only_by_extra_examples=True))
    model = StopRecoveryPPO.load(plan['initial_checkpoint'], device='cpu')
    if model.num_timesteps != 0 or model._n_updates != 0 or model.get_env() is not None:
        raise ValueError('A pre-BC, pre-PPO checkpoint without a physical environment is required')
    budget.tick()
    before = {name: value.detach().clone() for name, value in model.policy.state_dict().items()}
    base_digest = frozen_base_digest(model.policy)
    with np.load(OUT / 'candidate_demonstrations.npz', allow_pickle=False) as data:
        obs, masks, labels = [data[name].copy() for name in ('observations', 'masks', 'choices')]
    np.savez_compressed(target / 'demonstrations.npz', observations=obs, masks=masks, choices=labels)
    record_provenance(target, plan['parent_bundle'], extra_sources=(Path(__file__),))
    initialized = initialize_from_demonstrations(model, obs, masks, labels, budget,
        maximum_updates=plan['maximum_supervised_updates'])
    for name, value in model.policy.state_dict().items():
        if '.residual.' not in name and not torch.equal(value, before[name]):
            raise ValueError('A parameter outside the recovery residual changed: ' + name)
    if frozen_base_digest(model.policy) != base_digest:
        raise ValueError('Frozen base navigation actor changed')
    model.demonstration_initialization = dict(initialized, source_episodes=audit['sources'])
    model.save(target / 'policy_bc.zip')
    write(target / 'INITIALIZATION.json', initialized)
    export_recovery(model, plan['parent_bundle'], target / 'bc_portable')
    budget.tick()
    with torch.no_grad():
        distribution = model.policy.get_distribution(torch.from_numpy(obs), masks).distribution.probs.numpy()
    correct = distribution.argmax(-1) == labels
    original_n = audit['existing_examples']
    waits = labels == 0
    result = dict(completed=True, at=stamp(), supervised_updates=initialized['updates'],
        additional_ppo_steps=0, environment_steps=0, physical_validation_completed=False,
        environment_created=False, total_examples=len(labels), correct_examples=int(correct.sum()),
        original_examples=original_n, correct_original_examples=int(correct[:original_n].sum()),
        original_wait_examples=37, correct_original_wait_examples=int((correct[:original_n] & waits[:original_n]).sum()),
        extra_examples=len(labels)-original_n, correct_extra_examples=int(correct[original_n:].sum()),
        stop_examples=int(waits.sum()), correct_stop_examples=int(correct[waits].sum()),
        residual_parameters_only_changed=True, frozen_base_sha256=base_digest,
        policy_sha256=digest(target / 'policy_bc.zip'),
        export_manifest_sha256=digest(target / 'bc_portable/manifest.json'),
        export_contains_runtime_teacher=False, resource=budget.statistics(),
        old_holdout00_and01_now_training_tasks=True, new_ppo_or_geometry_comparison_changed=False,
        goal_achieved=False, deployment_approved=False)
    torch.save(dict(torch_rng=torch.get_rng_state(), numpy_rng=np.random.get_state(),
        python_rng=random.getstate()), target / 'random_state.pt')
    write(target / 'RESULTS.json', result)
    write(target / 'config.json', dict(plan=plan, dataset_audit=str(OUT / 'RESULTS.json'),
        supervised_only=True, training_cases_retired_as_holdouts=True))
    write(target / 'status.json', dict(status='offline_bc_complete', **result))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        print(json.dumps(dict(failed=True, error=repr(error), automatically_restarted=False)), flush=True)
        raise
