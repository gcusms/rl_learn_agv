"""Check 2.4 s waiting transfer at the actual95 failure, after normal checks.

The exact475-frame physical history plus12 actual zero commands initializes
both models. Only after that common initialization do learned actions count.
No timed command enters deployment; this does not repair the actual106 stall.
"""
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
DEPENDENCY = 'rl-actual95-learned-validation-20261016.service'
MODELS = {'new_data': 'actual95_transfer_bc400_s231016', 'matched_old_data': 'actual95_control_bc400_s231016'}
HOLD_S = 2.4


def stamp():
    return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2) + '\n')


def state():
    data = subprocess.check_output(['systemctl', '--user', 'show', DEPENDENCY,
        '-p', 'ActiveState', '-p', 'Result', '-p', 'MainPID', '-p', 'InvocationID'], text=True)
    return dict(line.split('=', 1) for line in data.splitlines())


def verify(plan):
    for name, expected in plan['inputs'].items():
        if digest(ROOT / name) != expected:
            raise ValueError('A frozen wait-transfer input changed: ' + name)


def main():
    if (OUT / 'PLAN.json').exists():
        raise FileExistsError('Do not restart the wait-transfer comparison')
    dependency = state()
    paths = [Path(__file__), BASE / 'actual_stop_validation_20261016/actual_prefix.json',
        BASE / 'actual95_transfer_20261016/RESULTS.json',
        ROOT / 'training/experiments/evaluate_feedback_recovery.py',
        ROOT / 'training/experiments/evaluate_docking_navigation.py']
    for model_name in MODELS.values():
        folder = ROOT / 'models/stop_feedback_20261015' / model_name
        paths += [folder / 'policy_bc.zip']
        manifest = json.loads((folder / 'bc_portable/manifest.json').read_text())
        paths += [folder / 'bc_portable' / name for name in ['manifest.json', *manifest['files']]]
    plan = dict(at=stamp(), inputs={str(p.relative_to(ROOT)): digest(p) for p in paths},
        dependency=DEPENDENCY, dependency_invocation=dependency['InvocationID'], dependency_at_prepare=dependency,
        models=MODELS, hold_s=HOLD_S, expected_extra_zero_commands=12, matched_initialization=True,
        prior_history_release_s=95., model_release_s=95. + HOLD_S,
        condition='Run after actual95 candidate recovery and all four original-task regression checks pass.',
        fixed_times_used_only_for_physical_initialization=True, no_expert_after_model_release=True,
        not_a_106s_escape_or_unseen_map_test=True, temperature_calibration_used=False,
        larger_ppo_training_admitted=False, new_ppo_steps=0, deployment_approved=False,
        resource=dict(soft_cpu_cores=.35, hard_cpu_percent=50, memory_high_mib=3072, memory_max_mib=4096))
    write('PLAN.json', plan)
    while True:
        current = state()
        if current['InvocationID'] not in ('', plan['dependency_invocation']):
            raise ValueError('The preceding main validation was restarted')
        if current['ActiveState'] in ('inactive', 'failed'):
            if current['ActiveState'] != 'inactive' or current['Result'] != 'success' or current['MainPID'] != '0':
                raise RuntimeError('The preceding validation did not finish successfully')
            break
        write('STATUS.json', dict(status='waiting_for_main_slot_and_normal_regression', at=stamp(),
            dependency=current, physical_simulation_started=False))
        time.sleep(5)
    verify(plan)
    previous_path = BASE / 'actual95_learned_validation_20261016/RESULTS.json'
    previous = json.loads(previous_path.read_text())
    actual95 = [r for r in previous.get('recovered_rows', []) if r['case_id'] == 'w11_training_stall_t95']
    if (not previous['completed'] or len(actual95) != 1 or not actual95[0]['safe_success']
            or len(previous.get('normal_rows', [])) != 4
            or not all(r['safe_success'] for r in previous['normal_rows'])
            or not previous.get('normal_pair', {}).get('all_regression_checks_passed')):
        write('RESULTS.json', dict(completed=True, skipped=True, at=stamp(),
            reason='Finish actual recovery and original-task safety/efficiency checks before testing transfer.',
            physical_simulation_started=False, goal_achieved=False))
        return
    import numpy as np
    import torch
    from unittest.mock import patch
    import experiments.evaluate_feedback_recovery as recovery
    from experiments.paired_navigation_gate import compare
    from neural_navigation.decision_feedback_runtime import StopFeedbackRuntime
    from neural_navigation.experiment_budget import ExperimentBudget
    from neural_navigation.masked_actions import ACTIONS
    torch.set_num_threads(1)
    prefix = json.loads((BASE / 'actual_stop_validation_20261016/actual_prefix.json').read_text())
    prefix.update(id='w11_actual95_wait2_4', release_s=95.)
    write('ADMISSION.json', dict(at=stamp(), preceding_main_terminal=current,
        preceding_complete_report_sha256=digest(previous_path), no_third_parallel_physical_job=True))
    output = OUT / 'physical'
    output.mkdir()
    budget = ExperimentBudget(.35, memory_mib=3072, directory=output)
    results = dict(completed=False, rows=[], new_ppo_steps=0, temperature_calibration_used=False,
        not_an_actual106_repair=True, goal_achieved=False, deployment_approved=False)
    first_initialization = None
    for variant, model_name in MODELS.items():
        verify(plan)
        created = []
        class HoldActor:
            def __init__(self, actor, core):
                self.actor, self.core, self.held_frames = actor, core, 0
            def predict(self, observation):
                if 95. - 1e-8 <= self.core.elapsed < 95. + HOLD_S - 1e-8:
                    self.held_frames += 1
                    return ACTIONS[0].copy()
                return self.actor.predict(observation)
        class Runtime(StopFeedbackRuntime):
            def __init__(self, *a, **kw):
                super().__init__(*a, **kw)
                actor = HoldActor(self.policy.policy, self.core)
                self.policy.policy = actor
                created.append(actor)
        write('STATUS.json', dict(status='physical_wait_transfer', at=stamp(), variant=variant))
        folder = ROOT / 'models/stop_feedback_20261015' / model_name / 'bc_portable'
        with patch.object(recovery, 'StopFeedbackRuntime', Runtime):
            row, trace = recovery.evaluate_recovery(prefix, folder, budget)
        assert len(created) == 1 and created[0].held_frames == 12
        times = trace['labels'][:, 0]
        hold = (times >= 95. - 1e-8) & (times < 95. + HOLD_S - 1e-8)
        assert int(hold.sum()) == 12
        np.testing.assert_array_equal(trace['commands'][hold], np.zeros((12, 2)))
        trace['initialization_hold'] = hold
        trace['model_controlled'] &= ~hold
        n = int(np.flatnonzero(trace['model_controlled'])[0])
        assert n == 487 and abs(times[n] - 97.4) < 1e-8
        initialization = {key: trace[key][:n + (key == 'labels')].copy()
            for key in ['labels', 'scans', 'odometry', 'measured_velocity', 'commands']}
        if first_initialization is None:
            first_initialization = initialization
        else:
            # Only physical pose columns of labels are initialization checks;
            # actor predictions and unrelated diagnostics are not ground truth.
            for key, value in initialization.items():
                a, b = value, first_initialization[key]
                if key == 'labels':
                    a, b = a[:, :4], b[:, :4]
                np.testing.assert_allclose(a, b, rtol=0, atol=2e-8, equal_nan=True)
        waits = (trace['choices'][n:] == 0) & trace['masks'][n:, 1:].any(-1)
        row.update(variant=variant, model_controlled_frames=int(trace['model_controlled'].sum()),
            source_frozen_prefix_frames=475, additional_initialization_hold_frames=12,
            release_sensor_timestamp_s=97.4, post_release_seconds=row['seconds'] - 97.4,
            post_release_wait_with_choices_frames=int(waits.sum()),
            post_release_only_stop_frames=int((~trace['masks'][n:, 1:].any(-1)).sum()),
            initial_observed_legal_wait_s=float(trace['observations'][n, 434] * 5.),
            all_model_commands_after_common_initialization=True,
            not_an_actual106_escape=True)
        path = output / (variant + '.npz')
        np.savez_compressed(path, **trace)
        row['trace_sha256'] = digest(path)
        results['rows'].append(row)
        results['resource'] = budget.statistics()
        (output / 'RESULTS.json').write_text(json.dumps(results, indent=2) + '\n')
        print(json.dumps(row), flush=True)
    results.update(completed=True, at=stamp(), paired_physical_initialization_verified=True,
        comparison=compare([results['rows'][1]], [results['rows'][0]]))
    (output / 'RESULTS.json').write_text(json.dumps(results, indent=2) + '\n')
    write('RESULTS.json', results)
    write('STATUS.json', dict(status='complete', at=stamp(), goal_achieved=False))


if __name__ == '__main__':
    os.chdir(ROOT)
    try:
        main()
    except Exception as error:
        write('STATUS.json', dict(status='failed', at=stamp(), error=repr(error), automatically_restarted=False))
        raise
