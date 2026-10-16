"""Fit one added actual recovery trajectory after its physical job finishes.

This experiment changes demonstrated data only. The separate wait-temperature
candidate is not used. Failed 106 s interventions are not positive examples.
"""
from collections import Counter
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
BASE = ROOT / 'reports/stop_feedback_20261015'
PARENT = ROOT / 'models/stop_feedback_20261015/early_transfer_bc800_s231015'
BUNDLE = ROOT / 'models/stop_feedback_20261015/baseline/portable'
TARGET = ROOT / 'models/stop_feedback_20261015/actual95_transfer_bc400_s231016'
DEPENDENCY = 'rl-actual95-left-probe-20261016.service'
SOURCE_ID = 'actual95_left_then_pass_1_6s'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stamp():
    return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()


def write(name, values):
    (OUT / name).write_text(json.dumps(values, indent=2) + '\n')


def state():
    text = subprocess.check_output(['systemctl', '--user', 'show', DEPENDENCY,
        '-p', 'ActiveState', '-p', 'Result', '-p', 'MainPID', '-p', 'InvocationID'], text=True)
    return dict(line.split('=', 1) for line in text.splitlines())


def verify(plan):
    for name, expected in plan['inputs'].items():
        if digest(ROOT / name) != expected:
            raise ValueError('A pinned learning input changed: ' + name)


def main():
    if TARGET.exists() or (OUT / 'PLAN.json').exists():
        raise FileExistsError('A new fitting experiment requires a fresh output')
    dependency = state()
    plan = dict(at=stamp(), dependency=DEPENDENCY, dependency_invocation=dependency['InvocationID'],
        selected_source_id=SOURCE_ID,
        source_selection='First completed actual95 physical success, explicitly selected after observing its outcome.',
        target=str(TARGET.relative_to(ROOT)), parent=str(PARENT.relative_to(ROOT)),
        only_changed_factor='Add the complete safe actual95 suffix to the exact 2355 demonstrations.',
        temperature_calibration_used=False, maximum_additional_supervised_updates=400,
        new_ppo_steps=0, soft_cpu_cores=.35, hard_cpu_percent=50,
        physical_prefix_and_expert_motion_not_counted_as_ppo=True,
        deployment_approved=False, inputs={})
    for p in [Path(__file__), PARENT / 'policy_bc.zip', PARENT / 'demonstrations.npz',
            BASE / 'early_decision_transfer_20261016/demonstrations.json',
            ROOT / 'training/experiments/recovery_demonstrations.py',
            ROOT / 'training/experiments/proactive_recovery.py',
            ROOT / 'training/experiments/stop_recovery_policy.py',
            ROOT / 'training/experiments/train_stop_feedback.py']:
        plan['inputs'][str(p.relative_to(ROOT))] = digest(p)
    manifest = json.loads((BUNDLE / 'manifest.json').read_text())
    assert digest(BUNDLE / 'manifest.json') == 'f412add64da251ec875e5e3f7b402e38d8153bf3b82425fafd99ef3ccc15ce4f'
    for name in ['manifest.json', *manifest['files']]:
        p = BUNDLE / name
        plan['inputs'][str(p.relative_to(ROOT))] = digest(p)
    write('PLAN.json', plan)
    while True:
        current = state()
        if current['InvocationID'] not in ('', plan['dependency_invocation']):
            raise RuntimeError('The preceding physical probe was restarted')
        if current['ActiveState'] in ('inactive', 'failed'):
            if current['ActiveState'] != 'inactive' or current['Result'] != 'success' or current['MainPID'] != '0':
                raise RuntimeError('The preceding physical probe did not finish successfully')
            break
        write('STATUS.json', dict(status='waiting_for_completed_physical_source', at=stamp(),
            dependency=current, new_training_started=False))
        time.sleep(5)
    verify(plan)
    report_path = BASE / 'actual_t95_left_physical_20261016/RESULTS.json'
    report = json.loads(report_path.read_text())
    if not report['completed']:
        raise ValueError('The full physical source experiment must be complete')
    selected = [r for r in report['rows'] if r['specification']['id'] == SOURCE_ID]
    if len(selected) != 1 or not selected[0]['safe_success']:
        raise ValueError('The selected actual failure recovery has no complete safe success')
    source_trace = report_path.parent / (SOURCE_ID + '.npz')
    if digest(source_trace) != selected[0]['trace_sha256']:
        raise ValueError('Completed source trace hash mismatch')
    import numpy as np
    import torch
    from experiments.experiment_provenance import record_provenance
    from experiments.recovery_demonstrations import load_demonstrations, initialize_from_demonstrations
    from experiments.stop_recovery_policy import StopRecoveryPPO, export_recovery
    from experiments.train_prefix_recovery import frozen_base_digest
    from neural_navigation.experiment_budget import ExperimentBudget
    torch.set_num_threads(1)
    torch.manual_seed(231016)
    np.random.seed(231016)
    TARGET.mkdir()
    (TARGET / 'status.json').write_text(json.dumps(dict(status='checking_demonstrations', new_ppo_steps=0)) + '\n')
    budget = ExperimentBudget(.35, memory_mib=3072, directory=TARGET)
    model = StopRecoveryPPO.load(PARENT / 'policy_bc.zip', device='cpu')
    model.policy.set_training_mode(False)
    if model.num_timesteps != 0:
        raise ValueError('The declared source must be the zero-PPO diagnostic BC')
    specifications = json.loads((BASE / 'early_decision_transfer_20261016/demonstrations.json').read_text())
    specifications += [dict(report=str(report_path.relative_to(ROOT)), id=SOURCE_ID,
        minimum_sensor_timestamp_s=95.)]
    write('demonstrations.json', specifications)
    write('STATUS.json', dict(status='checking_demonstrations', at=stamp()))
    observations, masks, choices, sources = load_demonstrations(specifications, model.policy)
    budget.tick()
    with np.load(PARENT / 'demonstrations.npz', allow_pickle=False) as z:
        n = len(z['choices'])
        assert n == 2355 and int((z['choices'] == 0).sum()) == 55
        for k, actual in [('observations', observations), ('masks', masks), ('choices', choices)]:
            np.testing.assert_array_equal(actual[:n], z[k])
    if len(choices) <= n:
        raise ValueError('The complete actual95 recovery supplied no new supported decisions')
    seen = {}
    for i, (x, mask, label) in enumerate(zip(observations, masks, choices)):
        key = hashlib.sha256(x.tobytes() + mask.tobytes()).digest()
        previous = seen.setdefault(key, (int(label), i))
        if previous[0] != int(label):
            raise ValueError('Exactly identical observations have conflicting demonstration labels')
    x = torch.from_numpy(observations)
    with torch.no_grad():
        before = model.policy.get_distribution(x, masks).distribution.probs.clone().numpy()
    base = frozen_base_digest(model.policy)
    assert base == 'e5502406c4ee8e42c4da09dd6affa022cb76e7f9b01316b1eaf0d2551431938e'
    np.savez_compressed(TARGET / 'demonstrations.npz', observations=observations, masks=masks, choices=choices)
    source_hashes = {str(report_path.relative_to(ROOT)): digest(report_path),
        str(source_trace.relative_to(ROOT)): digest(source_trace)}
    config = dict(plan, inputs=dict(plan['inputs'], **source_hashes),
        data=dict(old_examples=n, new_examples=len(choices) - n, total_examples=len(choices),
            old_prefix_bit_identical=True, old_necessary_stop_labels=55,
            new_label_counts=dict(Counter(map(int, choices[n:]))), exact_input_conflicts=0),
        demonstration_sources=sources, recovery_contract=model.recovery_contract,
        source_report_completed=True, source_physical_prefix_excluded=True,
        frozen_base_sha256=base, reward_changed=False, actions_masks_guard_unchanged=True)
    (TARGET / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
    record_provenance(TARGET, BUNDLE, [Path(__file__)])
    model.save(TARGET / 'policy_initial.zip')
    write('STATUS.json', dict(status='supervised_fitting', at=stamp(),
        total_demonstrations=len(choices), maximum_additional_updates=400, new_ppo_steps=0))
    fitting = initialize_from_demonstrations(model, observations, masks, choices, budget, maximum_updates=400)
    assert frozen_base_digest(model.policy) == base
    assert model.num_timesteps == 0
    model.policy.set_training_mode(False)
    with torch.no_grad():
        after = model.policy.get_distribution(x, masks).distribution.probs.clone().numpy()
    old_stop = choices[:n] == 0
    old_correct = before[:n].argmax(-1) == choices[:n]
    lost = np.flatnonzero(old_correct & (after[:n].argmax(-1) != choices[:n]))
    first_source_target = int(choices[n])
    first_source_chosen = int(after[n].argmax())
    result = dict(completed=True, at=stamp(), new_ppo_steps=0,
        fitting=fitting, old_examples=n, total_examples=len(choices),
        new_source=str(source_trace.relative_to(ROOT)), source_hashes=source_hashes,
        new_examples=len(choices) - n, old_prefix_bit_identical=True,
        old_correct_before=int(old_correct.sum()), old_correct_after=int((after[:n].argmax(-1) == choices[:n]).sum()),
        parent_correct_labels_lost_indices=lost.tolist(),
        old_necessary_stops_preserved=int((after[:n].argmax(-1)[old_stop] == 0).sum()),
        old_necessary_stop_count=int(old_stop.sum()),
        new_correct_before=int((before[n:].argmax(-1) == choices[n:]).sum()),
        new_correct_after=int((after[n:].argmax(-1) == choices[n:]).sum()),
        first_actual95_target=first_source_target, first_actual95_choice=first_source_chosen,
        new_target_probability_minimum=float(after[np.arange(n, len(choices)), choices[n:]].min()),
        frozen_base_sha256=base, temperature_calibration_used=False,
        model_navigation_success_not_yet_evaluated=True, actual106_escape_not_proven=True,
        no_large_ppo_or_deployment_admission=True, goal_achieved=False, deployment_approved=False,
        resource=budget.statistics())
    model.demonstration_initialization = dict(fitting, parent_checkpoint=str(PARENT.relative_to(ROOT)),
        parent_supervised_updates=800, additional_supervised_updates=fitting['updates'], sources=sources)
    model.save(TARGET / 'policy_bc.zip')
    if result['old_necessary_stops_preserved'] == result['old_necessary_stop_count']:
        export_recovery(model, BUNDLE, TARGET / 'bc_portable')
        card_path = TARGET / 'bc_portable/MODEL_CARD.json'
        card = json.loads(card_path.read_text())
        card.update(method='Additional complete actual95 safe demonstrations; supervised residual fitting only',
            additional_supervised_updates=fitting['updates'], additional_ppo_steps=0,
            temperature_calibration_used=False, navigation_acceptance_completed=False)
        card_path.write_text(json.dumps(card, indent=2) + '\n')
        result['admitted_for_bounded_physical_diagnosis_only'] = True
    else:
        result['admitted_for_bounded_physical_diagnosis_only'] = False
    torch.save(dict(torch_rng=torch.get_rng_state(), numpy_rng=np.random.get_state(),
        optimizer=model.policy.optimizer.state_dict()), TARGET / 'continuation_state.pt')
    (TARGET / 'RESULTS.json').write_text(json.dumps(result, indent=2) + '\n')
    (TARGET / 'status.json').write_text(json.dumps(dict(status='complete', new_ppo_steps=0,
        additional_supervised_updates=fitting['updates'], deployment_approved=False)) + '\n')
    write('RESULTS.json', result)
    write('STATUS.json', dict(status='supervised_fitting_complete', at=stamp(), new_ppo_steps=0,
        admitted_for_bounded_physical_diagnosis_only=result['admitted_for_bounded_physical_diagnosis_only']))
    print(json.dumps({k: v for k, v in result.items() if k not in ['fitting']}, indent=2), flush=True)


if __name__ == '__main__':
    os.chdir(ROOT)
    try:
        main()
    except Exception as error:
        write('STATUS.json', dict(status='failed', at=stamp(), error=repr(error), automatically_restarted=False))
        raise
