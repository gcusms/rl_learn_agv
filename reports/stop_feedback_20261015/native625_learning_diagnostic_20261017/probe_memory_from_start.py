"""Test finite evidence from initialization, with no timed runtime handover.

This is a bounded prerequisite, not full retention or deployment acceptance.
The supplied physical prefixes are admissibility-checked initialization only.
"""
import argparse
import json
from pathlib import Path
import subprocess
import time

from validate_local_speedup import read, write, digest


ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / 'reports/stop_feedback_20261015'
DEPENDENCY = 'rl-local-speedup-retention-20261017.service'
INVOCATION = '6073559918b94fab9955e98737bf7d81'
VARIANTS = ROOT / 'models/stop_feedback_20261015/memory_evidence_s231017'


def main(output):
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    task_path = BASE / 'context_feedback_longfit_protocol_20261016/tasks.json'
    late_path = BASE / 'parent_wait_protocol_20261016/specifications.json'
    handover_path = BASE / 'earlier_memory_evidence_20261017/RESULTS.json'
    parent_path = BASE / 'normal_suite_wait_validation_v2_20261016/RESULTS.json'
    hard_path = BASE / 'native625_batch_credit_validation_20261017/RESULTS.json'
    tasks, late, handover = read(task_path), read(late_path), read(handover_path)
    assert handover['completed'] and handover['complete_safe_gains'] == ['legal_wait_parent_t95']
    variants = read(VARIANTS / 'VARIANTS.json')
    assert variants['completed'] and variants['new_ppo_steps'] == 0
    hard = [next(s for s in late if s['id'] == 'legal_wait_parent_t95'),
        next(s for s in late if s['id'] == 'legal_wait_parent_t90'), tasks['actual106']]
    normal = [next(c for c in tasks['normal18'] if c['id'] == name)
        for name in ['independent_dynamic_000', 'train_failure_w11_e7']]
    files = [Path(__file__), Path(__file__).with_name('validate_local_speedup.py'), task_path,
        late_path, handover_path, parent_path, hard_path, VARIANTS / 'VARIANTS.json']
    for row in variants['variants']:
        for folder in [ROOT / row['source'], ROOT / row['output']]:
            manifest = read(folder / 'manifest.json')
            files += [folder / 'manifest.json'] + [folder / name for name in manifest['files']]
    files += [ROOT / s['trace'] for s in hard]
    for folder in ['experiments', 'neural_navigation', 'navrl']:
        files += list((ROOT / 'training' / folder).rglob('*.py'))
    files += [ROOT / 'training/neural_navigation/static_tracking_kernel.cpp']
    for scene in {s['case']['scene'] for s in hard} | {c['scene'] for c in normal}:
        files += [p for p in (ROOT / 'training' / scene).iterdir() if p.is_file()]
    hashes = {str(p.relative_to(ROOT)): digest(p) for p in set(files)}
    write(output / 'PLAN.json', dict(inputs=hashes, exact_dependency=DEPENDENCY,
        expected_invocation=INVOCATION, maximum_physical_episodes=5,
        hard_specifications=hard, normal_cases=normal,
        source_selection='PPO1665 only if preceding fixed and frozen retention both pass; otherwise BC4500',
        source_selection_not_based_on_this_config_experiment=True,
        finite_three_frame_evidence_from_runtime_initialization=True,
        no_timed_memory_switch_no_action_substitution_no_truth_input=True,
        prefix_must_remain_admissible_and_physically_exact=True,
        stop_if_original_success_lost_or_safety_violated=True,
        unchanged_625_features_12_actions_footprint_4cm_dynamics_and_blind_memory=True,
        success_still_requires_full_normal_sensor_docking_and_new_seed_retention=True,
        new_ppo_steps=0, new_optimizer_updates=0, deployment_approved=False))
    while True:
        result = subprocess.run(['systemctl', '--user', 'show', DEPENDENCY,
            '--property=MainPID,ActiveState,InvocationID,ExecMainStatus'],
            capture_output=True, text=True, check=True)
        state = dict(line.split('=', 1) for line in result.stdout.splitlines() if '=' in line)
        if state['InvocationID'] not in ('', INVOCATION):
            raise RuntimeError('Pinned predecessor was restarted or replaced')
        if state['MainPID'] == '0' and state['ActiveState'] in ('inactive', 'failed'):
            break
        write(output / 'STATUS.json', dict(status='waiting_for_exact_policy_retention',
            dependency=state, new_physical_episodes=0, new_ppo_steps=0))
        time.sleep(5.)
    assert state['ActiveState'] == 'inactive' and state['ExecMainStatus'] == '0'
    prior_path = BASE / 'native625_local_speedup_retention_20261017/RESULTS.json'
    prior = read(prior_path)
    assert prior['completed']
    choice = 'ppo1665' if prior.get('all_fixed_retention_passed') and prior.get('all_frozen_retention_passed') else 'bc4500'
    variant = next(row for row in variants['variants'] if row['name'] == choice)
    bundle = ROOT / variant['output']
    source = ROOT / variant['source']
    assert digest(bundle / 'actor.pt') == digest(source / 'actor.pt') == variant['actor_sha256']
    parent = read(parent_path)
    references = {r['case_id']: r for g in parent['groups'].values() for r in g['rows']}
    if choice == 'ppo1665':
        references.update({r['case_id']: r for g in prior['groups'].values() for r in g['rows']})
        references.update({r['case_id']: r for r in read(hard_path)['groups']['hard3']['rows']})
    hashes[str(prior_path.relative_to(ROOT))] = digest(prior_path)
    for name, expected in hashes.items():
        assert digest(ROOT / name) == expected, name
    write(output / 'DEPENDENCIES.json', dict(inputs=hashes, selected_frozen_actor=choice,
        source=str(source.relative_to(ROOT)), candidate=str(bundle.relative_to(ROOT)),
        actor_bytes_unchanged=True, predecessor_terminal=state))

    import numpy as np
    import torch
    from unittest.mock import patch
    import experiments.evaluate_docking_navigation as evaluator
    from experiments.release_recovery import evaluate_recovery
    from neural_navigation.experiment_budget import ExperimentBudget
    from neural_navigation.goal_pose_runtime import check_files, resolve_runtime

    check_files(bundle, read(bundle / 'manifest.json'))
    torch.set_num_threads(1)
    budget = ExperimentBudget(.35, memory_mib=2800, directory=output)
    runtime_type = resolve_runtime(read(bundle / 'manifest.json')['required_runtime'])
    env_type = evaluator.StressNavigationEnv
    physical = output / 'physical'
    physical.mkdir()
    report = dict(completed=False, rows=[], new_physical_episodes=0, selected_frozen_actor=choice,
        memory_config_from_initialization=True, no_timed_runtime_memory_switch=True,
        full_retention_not_yet_established=True, new_ppo_steps=0, new_optimizer_updates=0,
        goal_achieved=False, deployment_approved=False)

    def save():
        budget.tick()
        report['resource'] = budget.statistics()
        write(output / 'RESULTS.json', report)

    def finish(reason):
        for name, expected in hashes.items():
            assert digest(ROOT / name) == expected, name
        report.update(completed=True, reason=reason, inputs_verified=len(hashes))
        save()
        write(output / 'STATUS.json', dict(status='complete', reason=reason,
            completed_episodes=report['new_physical_episodes']))

    save()
    for spec, recovery in [(s, True) for s in hard] + [(s, False) for s in normal]:
        environments, runtimes = [], []
        release = spec['release_s'] if recovery else 0.

        class MeasuredEnvironment(env_type):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.evaluation_gap_reset = False
                environments.append(self)

            def step(self, action):
                # This clock resets an evaluation label only; no control or
                # memory configuration depends on it.
                if not self.evaluation_gap_reset and self.elapsed >= release - 1e-8:
                    self.minimum_gap = self._minimum_gap()
                    self.evaluation_gap_reset = True
                return super().step(action)

        class VerifiedRuntime(runtime_type):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                config = self.guard.memory.config
                assert config['free_evidence_frames'] == 3
                assert config['free_evidence_margin_m'] == .03
                assert config['static_footprint_margin_m'] == .04
                runtimes.append(self)

        write(output / 'STATUS.json', dict(status='physical_from_start_config_test', case_id=spec['id'],
            recovery_initialization=recovery, completed_episodes=report['new_physical_episodes']))
        target = 'experiments.release_recovery.resolve_runtime' if recovery else 'neural_navigation.goal_pose_runtime.resolve_runtime'
        try:
            with patch(target, return_value=VerifiedRuntime), patch.object(evaluator, 'StressNavigationEnv', MeasuredEnvironment):
                row, trace = (evaluate_recovery if recovery else evaluator.evaluate)(spec, bundle, budget)
        except ValueError as error:
            if 'frozen prefix command is no longer admissible' not in str(error):
                raise
            report['prefix_incompatible_case'] = dict(case_id=spec['id'], error=str(error),
                no_invalid_command_forced=True)
            finish('Original physical prefix inadmissible under from-start configuration; stop and diagnose')
            return
        assert len(environments) == len(runtimes) == 1
        minimum = float(environments[0].minimum_gap)
        row.update(minimum_post_release_dynamic_gap_m=minimum,
            safe_success=bool(row['safe_success'] and minimum >= .04 - 1e-4),
            memory_verified_from_initialization=True, frozen_actor_unchanged=True)
        path = physical / (spec['id'] + '.npz')
        np.savez_compressed(path, **trace)
        row.update(trace_path=str(path.relative_to(ROOT)), trace_sha256=digest(path))
        ref = references[row['case_id']]
        row['reference'] = {k: ref.get(k) for k in ['safe_success', 'seconds', 'post_release_seconds', 'travel_m']}
        report['rows'].append(row)
        report['new_physical_episodes'] += 1
        save()
        print(json.dumps(row), flush=True)
        if any(row[k] for k in ['collisions', 'false_arrival', 'physical_guards']) or minimum < .04 - 1e-4:
            finish('From-start configuration safety failure; no promotion')
            return
        time_key = 'post_release_seconds' if recovery else 'seconds'
        if ref['safe_success'] and (not row['safe_success'] or any(row[k] > ref[k]*1.2 + 1e-8
                for k in [time_key, 'travel_m'])):
            finish('Existing successful continuation or efficiency lost; no expansion')
            return
        if row['case_id'] == 'legal_wait_parent_t95' and not row['safe_success']:
            finish('Handover gain did not transfer to from-start configuration; no expansion')
            return
    report['bounded_from_start_gate_passed'] = True
    report['all_hard_states_solved'] = all(r['safe_success'] for r in report['rows'][:3])
    finish('Bounded from-start checks finished; full fixed and independent seed retention still required')


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
