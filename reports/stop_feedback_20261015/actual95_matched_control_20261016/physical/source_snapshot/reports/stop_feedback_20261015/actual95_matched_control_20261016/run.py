"""Continue the old dataset for 400 identical updates, then test actual stalls.

The treatment adds 66 examples and also trains for 400 extra updates. This
control matches that training dose, so improvement is not attributed to new
examples merely because the treatment received additional optimization.
"""
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
BASE = ROOT / 'reports/stop_feedback_20261015'
PARENT = ROOT / 'models/stop_feedback_20261015/early_transfer_bc800_s231015'
MODEL = ROOT / 'models/stop_feedback_20261015/actual95_control_bc400_s231016'
BUNDLE = ROOT / 'models/stop_feedback_20261015/baseline/portable'


def stamp():
    return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2) + '\n')


def main():
    if MODEL.exists() or (OUT / 'PLAN.json').exists():
        raise FileExistsError('Do not restart a matched optimization control')
    treatment = json.loads((BASE / 'actual95_transfer_20261016/RESULTS.json').read_text())
    assert treatment['completed'] and treatment['fitting']['updates'] == 400
    paths = [Path(__file__), PARENT / 'policy_bc.zip', PARENT / 'demonstrations.npz',
        BASE / 'actual95_transfer_20261016/PLAN.json', BASE / 'actual95_transfer_20261016/RESULTS.json',
        BASE / 'actual_stop_validation_20261016/actual_prefix.json',
        ROOT / 'training/experiments/recovery_demonstrations.py',
        ROOT / 'training/experiments/stop_recovery_policy.py',
        ROOT / 'training/experiments/evaluate_feedback_recovery.py']
    manifest = json.loads((BUNDLE / 'manifest.json').read_text())
    paths += [BUNDLE / name for name in ['manifest.json', *manifest['files']]]
    inputs = {str(p.relative_to(ROOT)): digest(p) for p in paths}
    plan = dict(at=stamp(), inputs=inputs, parent=str(PARENT.relative_to(ROOT)),
        target=str(MODEL.relative_to(ROOT)), old_examples=2355, new_examples=0,
        additional_supervised_updates=400, matching_treatment_seed=231016,
        identical_learning_rate_and_balanced_loss=True, temperature_calibration_used=False,
        physical_tests=[95., 106.], after_release_model_only=True,
        resource=dict(soft_cpu_cores=.15, hard_cpu_percent=20, memory_high_mib=3072, memory_max_mib=4096),
        deployment_approved=False, goal_achieved=False, new_ppo_steps=0)
    write('PLAN.json', plan)
    def verify():
        for name, expected in inputs.items():
            if digest(ROOT / name) != expected:
                raise ValueError('A matched-control input changed: ' + name)
    import numpy as np
    import torch
    from experiments.experiment_provenance import record_provenance
    from experiments.recovery_demonstrations import initialize_from_demonstrations
    from experiments.stop_recovery_policy import StopRecoveryPPO, export_recovery
    from experiments.train_prefix_recovery import frozen_base_digest
    from neural_navigation.experiment_budget import ExperimentBudget
    torch.set_num_threads(1)
    torch.manual_seed(231016)
    np.random.seed(231016)
    MODEL.mkdir()
    budget = ExperimentBudget(.15, memory_mib=3072, directory=MODEL)
    model = StopRecoveryPPO.load(PARENT / 'policy_bc.zip', device='cpu')
    assert model.num_timesteps == 0
    with np.load(PARENT / 'demonstrations.npz', allow_pickle=False) as z:
        observations, masks, choices = [z[k].copy() for k in ['observations', 'masks', 'choices']]
    assert len(choices) == 2355 and int((choices == 0).sum()) == 55
    frozen = frozen_base_digest(model.policy)
    np.savez_compressed(MODEL / 'demonstrations.npz', observations=observations, masks=masks, choices=choices)
    record_provenance(MODEL, BUNDLE, [Path(__file__)])
    model.save(MODEL / 'policy_initial.zip')
    write('STATUS.json', dict(status='matched_supervised_control', at=stamp()))
    history = []
    updates = 0
    # The shared initializer may stop early at p(target)>=0.9. Continue until
    # exactly the treatment's 400 Adam updates, with the same full batch/LR.
    while updates < 400:
        verify()
        fitted = initialize_from_demonstrations(model, observations, masks, choices, budget,
            maximum_updates=400 - updates)
        updates += fitted['updates']
        history.append(fitted)
    assert frozen_base_digest(model.policy) == frozen and model.num_timesteps == 0
    model.policy.set_training_mode(False)
    with torch.no_grad():
        predicted = model.policy.get_distribution(torch.from_numpy(observations), masks).distribution.probs.argmax(-1).numpy()
    stop_preserved = int((predicted[choices == 0] == 0).sum())
    fitting = dict(completed=True, at=stamp(), updates=updates, history=history,
        dataset_unchanged=True, examples=len(choices), old_correct=int((predicted == choices).sum()),
        necessary_stops_preserved=stop_preserved, frozen_base_sha256=frozen,
        new_ppo_steps=0, temperature_calibration_used=False, deployment_approved=False)
    (MODEL / 'INITIALIZATION.json').write_text(json.dumps(fitting, indent=2) + '\n')
    (MODEL / 'config.json').write_text(json.dumps(plan, indent=2) + '\n')
    model.demonstration_initialization = dict(parent_supervised_updates=800, additional_supervised_updates=400,
        unchanged_parent_demonstrations=True)
    model.save(MODEL / 'policy_bc.zip')
    torch.save(dict(torch_rng=torch.get_rng_state(), numpy_rng=np.random.get_state(),
        optimizer=model.policy.optimizer.state_dict()), MODEL / 'continuation_state.pt')
    if stop_preserved != 55:
        write('RESULTS.json', dict(completed=True, physical_tests_skipped=True, fitting=fitting,
            reason='Necessary-stop labels regressed', deployment_approved=False, goal_achieved=False))
        return
    export_recovery(model, BUNDLE, MODEL / 'bc_portable')
    card_path = MODEL / 'bc_portable/MODEL_CARD.json'
    card = json.loads(card_path.read_text())
    card.update(method='Matched 400 supervised update control with the unchanged2355 dataset',
        additional_supervised_updates=400, additional_ppo_steps=0, navigation_acceptance_completed=False)
    card_path.write_text(json.dumps(card, indent=2) + '\n')
    from experiments.evaluate_feedback_recovery import evaluate_recovery
    output = OUT / 'physical'
    output.mkdir()
    record_provenance(output, MODEL / 'bc_portable', [Path(__file__)])
    original = json.loads((BASE / 'actual_stop_validation_20261016/actual_prefix.json').read_text())
    result = dict(completed=False, fitting=fitting, rows=[], new_ppo_steps=0,
        deployment_approved=False, goal_achieved=False)
    for release in [95, 106]:
        verify()
        prefix = dict(original, id='w11_training_stall_t' + str(release), release_s=float(release))
        write('STATUS.json', dict(status='matched_control_physical_test', at=stamp(), case=prefix['id']))
        row, trace = evaluate_recovery(prefix, MODEL / 'bc_portable', budget)
        path = output / (prefix['id'] + '.npz')
        np.savez_compressed(path, **trace)
        row.update(variant='actual95_matched_control_bc400', trace_sha256=digest(path), no_expert_after_release=True)
        result['rows'].append(row)
        result['resource'] = budget.statistics()
        (output / 'RESULTS.json').write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps(row), flush=True)
    result['completed'] = True
    (output / 'RESULTS.json').write_text(json.dumps(result, indent=2) + '\n')
    write('RESULTS.json', result)
    write('STATUS.json', dict(status='complete', at=stamp(), new_ppo_steps=0, goal_achieved=False))


if __name__ == '__main__':
    os.chdir(ROOT)
    try:
        main()
    except Exception as error:
        write('STATUS.json', dict(status='failed', at=stamp(), error=repr(error), automatically_restarted=False))
        raise
