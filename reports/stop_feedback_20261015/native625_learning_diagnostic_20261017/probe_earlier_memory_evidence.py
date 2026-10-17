"""Serial component diagnostic: earlier finite-ray memory clearing, frozen actor.

Earlier106s/old-actor tests failed. This experiment uses the broadly evaluated
BC4500 actor and explicitly compares earlier90s and97.6s initialization states.
The activation time is an experimental handover, never a deployment rule or a
feature available to the policy. No scripted actions occur after handover.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import time


ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / 'reports/stop_feedback_20261015'
HERE = Path(__file__).resolve().parent
DEPENDENCIES = {
    'rl-native625-batch-credit-check-20261017.service': '040e38b20f3e48848ebe4f4e36840fd3',
    'rl-native625-batch-credit-pair-20261017.service': '11169c90e4064b13a3a64de2ca8e17e1',
}
CHANGE = dict(free_evidence_margin_m=.03, free_evidence_frames=3, free_evidence_max_gap_s=.4)


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def main(output):
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    tasks_path = BASE / 'context_feedback_longfit_protocol_20261016/tasks.json'
    late_path = BASE / 'parent_wait_protocol_20261016/specifications.json'
    parent_path = BASE / 'normal_suite_wait_validation_v2_20261016/RESULTS.json'
    prior_path = BASE / 'actual106_confirmed_free_probe_verified_20261016/RESULTS.json'
    audit_path = HERE / 'MEMORY_GEOMETRY_AUDIT.json'
    tests_path = HERE / 'MEMORY_FREE_EVIDENCE_TESTS.json'
    tasks, late, parent = read(tasks_path), read(late_path), read(parent_path)
    audit, tests, prior = read(audit_path), read(tests_path), read(prior_path)
    assert parent['completed'] and parent['normal18_pair']['all_regression_checks_passed']
    assert tests['passed'] and tests['tests_run'] == 9
    assert audit['completed'] and audit['original_rejections_reproduced']
    assert prior['completed'] and prior['complete_safe_trials'] == 0
    bundle = ROOT / parent['selected_model'] / 'portable'
    manifest = read(bundle / 'manifest.json')
    assert not manifest['deployment_approved']
    raw_prefix = BASE / 'context_broad_diagnostic_20261016/physical/hard3_w11_training_stall_t106.npz'
    early = dict(id='memory_evidence_actual90', case=tasks['actual106']['case'],
        trace=str(raw_prefix), release_s=90.)
    late95 = next(s for s in late if s['id'] == 'legal_wait_parent_t95')
    assert abs(late95['release_s'] - 97.6) < 1e-6
    specifications = [early, late95, tasks['actual106']]
    files = [Path(__file__), tasks_path, late_path, parent_path, prior_path, audit_path, tests_path,
        raw_prefix, bundle / 'manifest.json']
    files += [bundle / name for name in manifest['files']]
    files += [ROOT / s['trace'] for s in specifications]
    for folder in ['experiments', 'neural_navigation', 'navrl']:
        files += list((ROOT / 'training' / folder).rglob('*.py'))
    files += [ROOT / 'training/neural_navigation/static_tracking_kernel.cpp']
    for scene in {s['case']['scene'] for s in specifications}:
        files += [p for p in (ROOT / 'training' / scene).iterdir() if p.is_file()]
    hashes = {str(p.relative_to(ROOT)): digest(p) for p in set(files)}
    write(output / 'PLAN.json', dict(inputs=hashes, exact_dependencies=DEPENDENCIES,
        maximum_new_physical_episodes=3, specifications=specifications,
        memory_evidence_change=CHANGE, original_config=read(bundle / 'hardware.json')['near_field_memory'],
        frozen_actor_localizer_router_and_docking=True, no_expert_actions_after_release=True,
        experimental_component_handover_not_deployment_trigger=True,
        unchanged_body_4cm_margin_actions_dynamics_unknown_blind_memory_and_deadline=True,
        new_permissions_possible_only_from_finite_ray_evidence=True,
        no_truth_or_time_based_memory_expiration=True, no_parallel_physical_job=True,
        stop_on_any_safety_failure=True, new_ppo_steps=0, new_optimizer_updates=0,
        no_automatic_training_labels_or_runtime_release=True, deployment_approved=False))
    while True:
        states = {}
        for unit, expected in DEPENDENCIES.items():
            result = subprocess.run(['systemctl', '--user', 'show', unit,
                '--property=MainPID,ActiveState,InvocationID,ExecMainStatus'],
                capture_output=True, text=True, check=True)
            state = dict(line.split('=', 1) for line in result.stdout.splitlines() if '=' in line)
            if state['InvocationID'] not in ('', expected):
                raise RuntimeError('Pinned predecessor was replaced or restarted: ' + unit)
            states[unit] = state
        if all(s['MainPID'] == '0' and s['ActiveState'] in ('inactive', 'failed') for s in states.values()):
            break
        write(output / 'STATUS.json', dict(status='waiting_for_exact_physical_predecessors',
            dependencies=states, new_physical_episodes=0, new_ppo_steps=0))
        time.sleep(5.)
    assert all(s['ActiveState'] == 'inactive' and s['ExecMainStatus'] == '0' for s in states.values())
    previous_path = BASE / 'native625_batch_credit_validation_20261017/RESULTS.json'
    pair_path = BASE / 'native625_batch_credit_frozen_validation_20261017/RESULTS.json'
    previous, pair = read(previous_path), read(pair_path)
    assert previous['completed'] and pair['completed']
    if previous.get('all_development_retention_passed') and pair.get('all_frozen_checks_passed'):
        write(output / 'RESULTS.json', dict(completed=True, physical_started=False,
            reason='New PPO passed both declared gates; an additional component change is unnecessary',
            new_physical_episodes=0, new_ppo_steps=0, goal_achieved=False, deployment_approved=False))
        write(output / 'STATUS.json', dict(status='skipped_previous_candidate_passed'))
        return
    assert previous.get('physical_validation_started')
    for group in previous.get('groups', {}).values():
        assert all(not any(r[k] for k in ['collisions', 'false_arrival', 'physical_guards']) for r in group['rows'])
    reference_early = previous['groups']['parent_actual90']['rows'][0]
    assert reference_early['safe_success']
    references = {r['case_id']: r for g in parent['groups'].values() for r in g['rows']}
    references[early['id']] = reference_early
    hashes.update({str(p.relative_to(ROOT)): digest(p) for p in [previous_path, pair_path]})
    for name, expected in hashes.items():
        assert digest(ROOT / name) == expected, name
    write(output / 'DEPENDENCIES.json', dict(inputs=hashes, terminal_states=states,
        prior_model_only106_trial_failed=True, baseline_bundle=str(bundle.relative_to(ROOT))))

    import numpy as np
    import torch
    from unittest.mock import patch
    import experiments.evaluate_docking_navigation as evaluator
    from experiments.release_recovery import evaluate_recovery
    from neural_navigation.experiment_budget import ExperimentBudget
    from neural_navigation.goal_pose_runtime import check_files, resolve_runtime

    check_files(bundle, manifest)
    torch.set_num_threads(1)
    budget = ExperimentBudget(.35, memory_mib=2800, directory=output)
    runtime_type = resolve_runtime(manifest['required_runtime'])
    env_type = evaluator.StressNavigationEnv
    physical = output / 'physical'
    physical.mkdir()
    report = dict(completed=False, physical_started=True, rows=[], new_physical_episodes=0,
        new_ppo_steps=0, new_optimizer_updates=0, frozen_actor=True,
        experimental_memory_component_handover=True, goal_achieved=False, deployment_approved=False)

    def save():
        budget.tick()
        report.update(at=datetime.now(timezone.utc).isoformat(), resource=budget.statistics())
        write(output / 'RESULTS.json', report)

    def finish(reason):
        for name, expected in hashes.items():
            assert digest(ROOT / name) == expected, name
        report.update(completed=True, reason=reason, inputs_verified=len(hashes),
            complete_safe_gains=[r['case_id'] for r in report['rows'] if r['safe_success']
                and not references[r['case_id']]['safe_success']])
        save()
        write(output / 'STATUS.json', dict(status='complete', reason=reason,
            completed_episodes=report['new_physical_episodes']))

    save()
    for specification in specifications:
        environments, runtimes = [], []

        class MeasuredEnvironment(env_type):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                environments.append(self)

        class EvidenceHandoverRuntime(runtime_type):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.memory_evidence_activation = None
                runtimes.append(self)

            def step(self, scan, odometry, velocity, timestamp, *args, **kwargs):
                if self.memory_evidence_activation is None and timestamp >= specification['release_s'] - 1e-8:
                    self.guard.memory.config = dict(self.guard.memory.config, **CHANGE)
                    self.guard.memory.free_confirmations.clear()
                    self.memory_evidence_activation = float(timestamp)
                    # Truth is used only to reset an independent evaluation metric.
                    environments[0].minimum_gap = environments[0]._minimum_gap()
                return super().step(scan, odometry, velocity, timestamp, *args, **kwargs)

        write(output / 'STATUS.json', dict(status='physical_component_diagnostic',
            case_id=specification['id'], completed_episodes=report['new_physical_episodes']))
        with patch('experiments.release_recovery.resolve_runtime', return_value=EvidenceHandoverRuntime), \
                patch.object(evaluator, 'StressNavigationEnv', MeasuredEnvironment):
            row, trace = evaluate_recovery(specification, bundle, budget)
        assert len(environments) == len(runtimes) == 1
        assert abs(runtimes[0].memory_evidence_activation - specification['release_s']) < 1e-6
        minimum_gap = float(environments[0].minimum_gap)
        row.update(minimum_post_release_dynamic_gap_m=minimum_gap,
            safe_success=bool(row['safe_success'] and minimum_gap >= .04 - 1e-4),
            memory_evidence_activation_s=runtimes[0].memory_evidence_activation,
            frozen_model=str(bundle.parent.relative_to(ROOT)),
            component_handover_not_unchanged_mask_comparison=True,
            no_expert_actions_after_release=True, no_positive_labels_automatically_admitted=True)
        path = physical / (specification['id'] + '.npz')
        np.savez_compressed(path, **trace)
        row.update(trace_path=str(path.relative_to(ROOT)), trace_sha256=digest(path))
        reference = references[row['case_id']]
        row['reference'] = {k: reference[k] for k in ['safe_success', 'post_release_seconds', 'travel_m',
            'post_release_wait_with_choices_frames', 'post_release_only_stop_frames']}
        report['rows'].append(row)
        report['new_physical_episodes'] += 1
        save()
        print(json.dumps(row), flush=True)
        if any(row[k] for k in ['collisions', 'false_arrival', 'physical_guards']) or minimum_gap < .04 - 1e-4:
            finish('Safety failure; component experiment stopped without promotion')
            return
        if reference['safe_success'] and (not row['safe_success'] or any(
                row[k] > reference[k]*1.2 + 1e-8 for k in ['post_release_seconds', 'travel_m'])):
            finish('Earlier successful continuation lost or efficiency regressed; no expansion')
            return
    finish('All bounded diagnostics finished; no automatic training or deployment')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    try:
        main(args.output)
    except Exception as error:
        if args.output.is_dir():
            write(args.output / 'FAILURE.json', dict(error=repr(error), automatically_restarted=False))
        raise
