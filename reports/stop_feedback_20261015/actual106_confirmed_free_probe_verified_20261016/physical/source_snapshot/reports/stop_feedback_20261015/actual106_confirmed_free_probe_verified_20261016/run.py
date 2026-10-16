"""Queue bounded component-handover diagnostics after normal regressions.

All530 recorded frames use the unchanged original guard. At106s, transfer
only measured near-field memory to a three-confirmation free-evidence mode.
This is an offline component-handover experiment, not a deployment trigger.
Try the saved network first; any subsequent explicit action windows remain
expert probes and may never be reported as learned-model successes.
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
BUNDLE = ROOT / 'models/stop_feedback_20261015/actual95_transfer_bc400_s231016/bc_portable'
DEPENDENCY = 'rl-actual95-broad-regression-20261016.service'


def stamp():
    return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(name, data):
    path = OUT / name
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, indent=2) + '\n')
    temporary.replace(path)


def state():
    output = subprocess.check_output(['systemctl', '--user', 'show', DEPENDENCY,
        '-p', 'ActiveState', '-p', 'Result', '-p', 'MainPID', '-p', 'InvocationID'], text=True)
    return dict(line.split('=', 1) for line in output.splitlines())


def main():
    if (OUT / 'PLAN.json').exists():
        raise FileExistsError('Do not restart a component-handover probe')
    dependency = state()
    audit_path = BASE / 'actual106_memory_context_verified_20261016/RESULTS.json'
    audit = json.loads(audit_path.read_text())
    assert audit['completed'] and audit['all700_saved_actor_proposals_exactly_reproduced']
    assert all(audit['fixtures'].values())
    assert audit['frames_with_new_geometry_permissions'] > 0
    local = audit['local_geometry_checks']
    assert local and min(min(row['minimum_truth_gap_m'].values()) for row in local) >= .04
    original = json.loads((BASE / 'actual_stop_validation_20261016/actual_prefix.json').read_text())
    prefix = dict(original, id='w11_actual106_component_handover', release_s=106.)
    write('prefix.json', prefix)
    specifications = [dict(id='model_only', windows=[])]
    for turn_s, pass_s in [(1.6, 6.), (3.4, 9.)]:
        specifications.append(dict(id='expert_right_' + str(turn_s).replace('.', '_'),
            windows=[dict(start_s=110., end_s=110. + turn_s, choice=5),
                     dict(start_s=110. + turn_s, end_s=110. + turn_s + pass_s, choice=1)]))
    write('specifications.json', specifications)
    manifest = json.loads((BUNDLE / 'manifest.json').read_text())
    paths = [Path(__file__), audit_path,
        BASE / 'actual106_memory_context_20261016/replay.py',
        BASE / 'actual106_memory_context_20261016/replay.npz',
        BASE / 'actual106_memory_context_verified_20261016/recover.py',
        BASE / 'actual95_broad_regression_20261016/run.py',
        BASE / 'actual95_broad_regression_20261016/PLAN.json',
        OUT / 'prefix.json', OUT / 'specifications.json', ROOT / prefix['trace'], BUNDLE / 'manifest.json']
    paths += [BUNDLE / name for name in manifest['files']]
    for folder in ['navrl', 'neural_navigation', 'experiments']:
        paths += list((ROOT / 'training' / folder).rglob('*.py'))
    paths += list((ROOT / 'training' / prefix['case']['scene']).glob('*'))
    inputs = {str(p.relative_to(ROOT)): digest(p) for p in set(paths) if p.is_file()}
    plan = dict(at=stamp(), inputs=inputs, dependency_unit=DEPENDENCY,
        dependency_invocation_id=dependency['InvocationID'], dependency_at_prepare=dependency,
        resource=dict(soft_cpu_cores=.35, hard_cpu_percent=50, memory_high_mib=3072, memory_max_mib=4096),
        recorded_physical_prefix_frames=530, activation_s=106.,
        hypothetical_memory=dict(free_evidence_margin_m=.03, free_evidence_frames=3, free_evidence_max_gap_s=.4),
        activation_is_experimental_component_handover_not_a_deployment_rule=True,
        action_commands_body_4cm_margin_trajectory_rules_and_models_unchanged=True,
        memory_evidence_and_resulting_action_permissions_differ=True,
        tested_first_without_expert_actions=True, maximum_physical_trials=3,
        stop_after_first_complete_safe_recovery=True,
        full18_and_other_groups_must_not_lose_parent_successes=True,
        local_geometry_is_only_research_admission_not_safety_proof=True,
        no_parallel_physical_job=True, new_ppo_steps=0, new_optimizer_updates=0,
        no_automatic_positive_labels_or_model_promotion=True, goal_achieved=False, deployment_approved=False)
    write('PLAN.json', plan)
    def verify():
        for name, expected in inputs.items():
            if digest(ROOT / name) != expected:
                raise ValueError('A component-handover input changed: ' + name)
    while True:
        current = state()
        if current['InvocationID'] not in ('', plan['dependency_invocation_id']):
            raise RuntimeError('The predecessor was restarted')
        if current['ActiveState'] in ('inactive', 'failed'):
            if current['ActiveState'] != 'inactive' or current['Result'] != 'success' or current['MainPID'] != '0':
                raise RuntimeError('The preceding physical main job did not finish successfully')
            break
        write('STATUS.json', dict(at=stamp(), status='waiting_for_broad_regression',
            physical_simulation_started=False, dependency=current))
        time.sleep(5)
    finished_path = BASE / 'actual95_broad_regression_20261016/RESULTS.json'
    finished = json.loads(finished_path.read_text())
    assert finished['completed'] and finished['broader_regression_completed']
    admissible = bool(finished['full18_pair']['all_regression_checks_passed']
        and finished['groups']['fresh12']['paired']['safety_nonregression']
        and finished['groups']['original_prefixes2']['paired']['safety_nonregression']
        and all(r['safe_success'] for r in finished['groups']['heading2']['rows'])
        and all(r['safe_success'] for r in finished['groups']['w11_t90']['rows']))
    if not admissible:
        write('RESULTS.json', dict(completed=True, skipped=True, at=stamp(),
            reason='Predecessor failed safety/success retention; do not expand this candidate.',
            predecessor_report_sha256=digest(finished_path), new_physical_steps=0,
            goal_achieved=False, deployment_approved=False))
        return
    verify()
    import numpy as np
    import torch
    from unittest.mock import patch
    import experiments.evaluate_feedback_recovery as evaluator
    from experiments.probe_feedback_alternatives import ProbeActor, intervention_windows
    from experiments.experiment_provenance import record_provenance
    from neural_navigation.decision_feedback_runtime import StopFeedbackRuntime
    from neural_navigation.experiment_budget import ExperimentBudget
    torch.set_num_threads(1)
    physical = OUT / 'physical'
    physical.mkdir()
    record_provenance(physical, BUNDLE, [Path(__file__)])
    write('ADMISSION.json', dict(at=stamp(), verified_predecessor_terminal=current,
        predecessor_report_sha256=digest(finished_path), physically_complete_normal18=True,
        no_parent_success_lost_in_extra_recovery_or_heading_groups=True,
        offline_geometry_hypothesis_only=True, no_training_or_deployment_admitted=True))
    budget = ExperimentBudget(.35, memory_mib=3072, directory=physical)
    report = dict(completed=False, rows=[], deployment_approved=False, goal_achieved=False,
                  new_ppo_steps=0, new_optimizer_updates=0, not_a_runtime_release=True)
    for specification in specifications:
        verify()
        created = []
        class Runtime(StopFeedbackRuntime):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.research_memory_activation = None
                self.probe = None
                if specification['windows']:
                    self.probe = ProbeActor(self.policy.policy, self.core, specification)
                    self.policy.policy = self.probe
                created.append(self)

            def step(self, scan, odometry, velocity, timestamp, *args, **kwargs):
                if self.research_memory_activation is None and timestamp >= 106. - 1e-8:
                    self.guard.memory.config = dict(self.guard.memory.config, **plan['hypothetical_memory'])
                    self.guard.memory.free_confirmations.clear()
                    self.research_memory_activation = float(timestamp)
                return super().step(scan, odometry, velocity, timestamp, *args, **kwargs)
        write('STATUS.json', dict(at=stamp(), status='physical_component_handover',
            case_id=specification['id'], physical_simulation_started=True))
        with patch.object(evaluator, 'StopFeedbackRuntime', Runtime):
            row, trace = evaluator.evaluate_recovery(prefix, BUNDLE, budget)
        assert len(created) == 1 and abs(created[0].research_memory_activation - 106.) < 1e-8
        times = trace['labels'][:, 0]
        expert = np.zeros(len(times), bool)
        for window in (intervention_windows(specification) if specification['windows'] else []):
            expert |= (times >= window['start_s'] - 1e-8) & (times < window['end_s'] - 1e-8)
        trace['expert_controlled'] = expert
        trace['model_controlled'] &= ~expert
        path = physical / (specification['id'] + '.npz')
        np.savez_compressed(path, **trace)
        row.update(case_id=specification['id'], specification=specification,
            memory_changed_after_original_physical_prefix=True,
            memory_activation_s=created[0].research_memory_activation,
            expert_controlled_frames=int(expert.sum()), model_controlled_frames=int(trace['model_controlled'].sum()),
            no_expert_actions=not specification['windows'],
            intervention=[] if created[0].probe is None else created[0].probe.entries,
            trace_sha256=digest(path), not_a_model_success_if_expert_used=True,
            original_guard_result_not_comparable_as_an_unchanged_mask_trial=True)
        report['rows'].append(row)
        report['resource'] = budget.statistics()
        write('RESULTS.json', report)
        print(json.dumps({k: v for k, v in row.items() if k != 'intervention'}), flush=True)
        if row['safe_success']:
            report['remaining_trials_skipped_after_safe_recovery'] = True
            break
    verify()
    report.update(completed=True, at=stamp(), resource=budget.statistics(),
        complete_safe_trials=sum(r['safe_success'] for r in report['rows']),
        model_only_complete_safe_trials=sum(r['safe_success'] and r['no_expert_actions'] for r in report['rows']),
        scope='Experimental memory component handover on a reused actual106 state. Expert trials are diagnostic, not learned-model results or admitted training samples.')
    write('RESULTS.json', report)
    write('STATUS.json', dict(at=stamp(), status='complete', goal_achieved=False))


if __name__ == '__main__':
    os.chdir(ROOT)
    try:
        main()
    except Exception as error:
        write('FAILURE.json', dict(at=stamp(), error=repr(error), automatically_restarted=False,
                                  goal_achieved=False, deployment_approved=False))
        raise
