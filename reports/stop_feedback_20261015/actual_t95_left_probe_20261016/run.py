"""Check earlier recovery from the unchanged actual 95 s failure prefix."""
import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
BASE = ROOT / 'reports/stop_feedback_20261015'
MAIN_UNITS = ['rl-actual-stop-validation-main-20261016.service',
    'rl-committed-right-probe-20261016.service']


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def stamp():
    return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()


def write(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2) + '\n')


def main_slot():
    states = {}
    for unit in MAIN_UNITS:
        data = subprocess.check_output(['systemctl', '--user', 'show', unit,
            '-p', 'ActiveState', '-p', 'Result', '-p', 'MainPID'], text=True)
        state = dict(line.split('=', 1) for line in data.splitlines())
        if state != dict(ActiveState='inactive', Result='success', MainPID='0'):
            raise RuntimeError('Preceding physical main slot is not terminal: ' + unit)
        states[unit] = state
    return states


def verify(plan):
    for name, expected in plan['inputs'].items():
        if digest(ROOT / name) != expected:
            raise ValueError('A pinned physical-experiment input changed: ' + name)


def prepare():
    if (OUT / 'PLAN.json').exists():
        raise FileExistsError('Do not redefine the experiment after preparation')
    states = main_slot()
    original = json.loads((BASE / 'actual_stop_validation_20261016/PLAN.json').read_text())
    main = json.loads((BASE / 'actual_stop_validation_20261016/RESULTS_main.json').read_text())
    assert main['completed'] and main['fixed35_groups_completed']
    previous = json.loads((BASE / 'committed_right_actual_t106_20261016/RESULTS.json').read_text())
    assert previous['completed'] and not any(r['safe_success'] for r in previous['rows'])
    prefix = json.loads((BASE / 'actual_stop_validation_20261016/actual_prefix.json').read_text())
    prefix.update(id='w11_actual_failure_t95', release_s=95.)
    write('prefix.json', prefix)
    specifications = [dict(id='actual95_left_then_pass_' + str(turn_s).replace('.', '_') + 's',
        windows=[dict(start_s=95., end_s=95. + turn_s, choice=4),
            dict(start_s=95. + turn_s, end_s=95. + turn_s + pass_s, choice=1)])
        for turn_s, pass_s in [(1.6, 6.), (3.4, 9.)]]
    write('specifications.json', specifications)
    inputs = dict(original['inputs'])
    for p in [Path(__file__), OUT / 'prefix.json', OUT / 'specifications.json',
            BASE / 'actual_stop_validation_20261016/RESULTS_main.json',
            BASE / 'committed_right_actual_t106_20261016/RESULTS.json']:
        inputs[str(p.relative_to(ROOT))] = digest(p)
    plan = dict(at=stamp(), inputs=inputs, bundle=original['bundle'],
        preceding_main_states=states, output=str((BASE / 'actual_t95_left_physical_20261016').relative_to(ROOT)),
        predeclared_experiments=2, physical_prefix_release_s=95.,
        purpose='Test whether a real earlier failure state has a complete safe alternative.',
        not_a_solution_to_later106s_stall=True, not_a_runtime_rule=True,
        actions_masks_body_sensors_guard_and_timeout_unchanged=True,
        new_ppo_steps=0, automatic_label_or_training_admission=False,
        soft_cpu_cores=.35, hard_cpu_percent=50,
        deployment_approved=False, goal_achieved=False)
    verify(plan)
    write('PLAN.json', plan)
    print(json.dumps(dict(prepared=True, pins=len(inputs), physical_steps=0)))


def run():
    plan = json.loads((OUT / 'PLAN.json').read_text())
    verify(plan)
    states = main_slot()
    output = ROOT / plan['output']
    if output.exists():
        raise FileExistsError('A physical experiment may not be restarted into the same output')
    import numpy as np
    import torch
    from unittest.mock import patch
    import experiments.evaluate_feedback_recovery as recovery
    from experiments.probe_feedback_alternatives import ProbeActor, intervention_windows
    from neural_navigation.decision_feedback_runtime import StopFeedbackRuntime
    from neural_navigation.experiment_budget import ExperimentBudget
    torch.set_num_threads(1)
    output.mkdir()
    prefix = json.loads((OUT / 'prefix.json').read_text())
    specifications = json.loads((OUT / 'specifications.json').read_text())
    write('ADMISSION.json', dict(at=stamp(), verified_terminal_main_slots=states,
        new_parallel_physical_main_jobs=1, trained_model_or_deployment_admitted=False))
    budget = ExperimentBudget(.35, memory_mib=3072, directory=output)
    report = dict(completed=False, new_ppo_steps=0, not_a_trained_policy=True,
        not_a_later106s_escape=True, no_automatic_training_admission=True,
        deployment_approved=False, rows=[])
    for specification in specifications:
        verify(plan)
        write('STATUS.json', dict(status='running', at=stamp(), case=specification['id']))
        created = []
        class Runtime(StopFeedbackRuntime):
            def __init__(self, *a, **kw):
                super().__init__(*a, **kw)
                actor = ProbeActor(self.policy.policy, self.core, specification)
                self.policy.policy = actor
                created.append(actor)
        with patch.object(recovery, 'StopFeedbackRuntime', Runtime):
            row, trace = recovery.evaluate_recovery(prefix, ROOT / plan['bundle'], budget)
        assert len(created) == 1
        times = trace['labels'][:, 0]
        expert = np.zeros(len(times), dtype=bool)
        for window in intervention_windows(specification):
            expert |= (times >= window['start_s'] - 1e-8) & (times < window['end_s'] - 1e-8)
        assert int(expert.sum()) == len(created[0].entries)
        trace['expert_controlled'] = expert
        trace['model_controlled'] &= ~expert
        path = output / (specification['id'] + '.npz')
        np.savez_compressed(path, **trace)
        row.update(case_id=specification['id'], specification=specification,
            intervention=created[0].entries, not_a_trained_policy=True,
            expert_controlled_frames=int(expert.sum()),
            model_controlled_frames=int(trace['model_controlled'].sum()),
            trace_sha256=digest(path), no_automatic_training_admission=True)
        report['rows'].append(row)
        report['resource'] = budget.statistics()
        (output / 'RESULTS.json').write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps({k: v for k, v in row.items() if k != 'intervention'}), flush=True)
    report['completed'] = True
    (output / 'RESULTS.json').write_text(json.dumps(report, indent=2) + '\n')
    write('RESULTS.json', dict(completed=True, at=stamp(), report=str((output / 'RESULTS.json').relative_to(ROOT)),
        safe_complete_probes=sum(bool(r['safe_success']) for r in report['rows']),
        new_ppo_steps=0, goal_achieved=False, deployment_approved=False))
    write('STATUS.json', dict(status='complete', at=stamp()))


if __name__ == '__main__':
    os.chdir(ROOT)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'run'])
    args = parser.parse_args()
    try:
        prepare() if args.action == 'prepare' else run()
    except Exception as error:
        write('STATUS.json', dict(status='failed', at=stamp(), error=repr(error), automatically_restarted=False))
        raise
