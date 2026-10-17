"""Prospective full retention of a from-start finite-evidence configuration.

The preceding five-case prerequisite must pass before any new physics. This
study changes no learned weights and does not waive the unresolved106s state.
Fresh paired seeds are frozen before the configuration results are available.
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
DEPENDENCY = 'rl-memory-from-start-20261017.service'
INVOCATION = '20614a91e7504562af75785bc4e9fe02'
VARIANTS = ROOT / 'models/stop_feedback_20261015/memory_evidence_s231017'


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
    task_path = BASE / 'context_feedback_longfit_protocol_20261016/tasks.json'
    baseline_path = BASE / 'normal_suite_wait_validation_v2_20261016/RESULTS.json'
    ppo_path = BASE / 'native625_local_speedup_retention_20261017/RESULTS.json'
    prerequisite_path = BASE / 'memory_from_start_20261017/RESULTS.json'
    cases_path = HERE / 'CASES.json'
    freeze_path = HERE / 'FREEZE.json'
    tasks, variants, cases = read(task_path), read(VARIANTS / 'VARIANTS.json'), read(cases_path)
    freeze = read(freeze_path)
    assert freeze['cases_sha256'] == digest(cases_path)
    assert freeze['frozen_before_from_start_results'] and len(cases) == 8
    assert len({case['seed'] for case in cases}) == 8
    assert not prerequisite_path.exists(), 'Freeze the prospective study before prerequisite results'
    recoveries = [*tasks['actual95_actual90'], tasks['delayed_wait']]
    files = [Path(__file__), cases_path, freeze_path, task_path, baseline_path, VARIANTS / 'VARIANTS.json']
    for row in variants['variants']:
        for bundle in [ROOT / row['source'], ROOT / row['output']]:
            manifest = read(bundle / 'manifest.json')
            files += [bundle / 'manifest.json'] + [bundle / name for name in manifest['files']]
    files += [ROOT / spec['trace'] for spec in recoveries]
    for folder in ['experiments', 'neural_navigation', 'navrl']:
        files += list((ROOT / 'training' / folder).rglob('*.py'))
    files += [ROOT / 'training/neural_navigation/static_tracking_kernel.cpp']
    scenes = {case['scene'] for case in tasks['normal18'] + tasks['sensor5'] + tasks['heading2'] + cases}
    scenes |= {spec['case']['scene'] for spec in recoveries}
    for scene in scenes:
        files += [path for path in (ROOT / 'training' / scene).iterdir() if path.is_file()]
    hashes = {str(path.relative_to(ROOT)): digest(path) for path in set(files)}
    write(output / 'PLAN.json', dict(at=datetime.now(timezone.utc).isoformat(), inputs=hashes,
        dependency=DEPENDENCY, expected_invocation=INVOCATION,
        maximum_physical_episodes=44, fixed_groups=dict(normal18=18, sensor5=5, source_recovery3=3, heading2=2),
        fresh_seed_pairs=8, cases_sha256=digest(cases_path),
        prerequisite='All5 from-start prerequisite cases and its bounded gate must complete',
        candidate_selection='Exactly the source actor selected by the preceding independent retention study',
        no_model_or_configuration_selection_using_these_new_seeds=True,
        same_known_maps_not_unseen_maps=True,
        no_new_training_or_runtime_handover_or_action_substitution=True,
        original625_observation_12actions_footprint_4cm_and_dynamic_prediction_retained=True,
        changed_behavior='Only the already-built finite3frame free-evidence memory configuration',
        physical_gap_4cm_measured_independently_after_recovery_initialization=True,
        original106s_failure_requirement_not_waived=True,
        no_parallel_physical_job=True, soft_cpu_cores=.35, memory_max_mib=3072,
        new_ppo_steps=0, new_optimizer_updates=0, goal_achieved=False, deployment_approved=False))
    while True:
        result = subprocess.run(['systemctl', '--user', 'show', DEPENDENCY,
            '--property=MainPID,ActiveState,InvocationID,ExecMainStatus'],
            capture_output=True, text=True, check=True)
        state = dict(line.split('=', 1) for line in result.stdout.splitlines() if '=' in line)
        if state['InvocationID'] not in ('', INVOCATION):
            raise RuntimeError('Exact predecessor was replaced or restarted')
        if state['MainPID'] == '0' and state['ActiveState'] in ('inactive', 'failed'):
            break
        write(output / 'STATUS.json', dict(status='waiting_for_exact_from_start_prerequisite',
            dependency=state, new_physical_episodes=0, new_ppo_steps=0))
        time.sleep(5.)
    report = dict(completed=False, groups={}, frozen_rows=[], gates={}, new_physical_episodes=0,
        new_ppo_steps=0, new_optimizer_updates=0, no_actor_weights_changed=True,
        original106s_requirement_remains=True, goal_achieved=False, deployment_approved=False)
    if state['ActiveState'] != 'inactive' or state['ExecMainStatus'] != '0':
        report.update(completed=True, reason='Exact predecessor failed; no additional physics', predecessor_state=state)
        write(output / 'RESULTS.json', report)
        write(output / 'STATUS.json', dict(status='skipped_predecessor_failed', new_physical_episodes=0))
        return
    prerequisite = read(prerequisite_path)
    assert prerequisite['completed']
    if not prerequisite.get('bounded_from_start_gate_passed') or len(prerequisite['rows']) != 5:
        report.update(completed=True, reason='From-start gain or retention prerequisite failed; no expansion',
            prerequisite_reason=prerequisite.get('reason'))
        write(output / 'RESULTS.json', report)
        write(output / 'STATUS.json', dict(status='skipped_prerequisite_gate_failed', new_physical_episodes=0))
        return
    choice = prerequisite['selected_frozen_actor']
    variant = next(row for row in variants['variants'] if row['name'] == choice)
    source, candidate = ROOT / variant['source'], ROOT / variant['output']
    baseline = read(baseline_path)
    source_results = read(ppo_path) if choice == 'ppo1665' else baseline
    assert source_results['completed']
    if choice == 'ppo1665':
        assert source_results['all_fixed_retention_passed'] and source_results['all_frozen_retention_passed']
    source_rows = {row['case_id']: row for group in source_results['groups'].values() for row in group['rows']}
    original_rows = {row['case_id']: row for group in baseline['groups'].values() for row in group['rows']}
    for name in ['actor.pt', 'parent_actor.pt', 'previous_risk_actor.pt', 'localizer.pt', 'router.pt',
                 'decision_feedback.json', 'release_observation.json', 'docking_contract.json', 'localization_contract.json']:
        assert digest(source / name) == digest(candidate / name), name
    source_hw, candidate_hw = [read(bundle / 'hardware.json') for bundle in [source, candidate]]
    assert candidate_hw['near_field_memory']['free_evidence_frames'] == 3
    assert candidate_hw['near_field_memory']['free_evidence_margin_m'] == .03
    assert candidate_hw['near_field_memory']['static_footprint_margin_m'] == .04
    assert candidate_hw['robot_footprint_m'] == [.393924904, .329460022]
    for field in ['free_evidence_frames', 'free_evidence_margin_m', 'free_evidence_max_gap_s']:
        candidate_hw['near_field_memory'].pop(field)
        source_hw['near_field_memory'].pop(field, None)
    assert source_hw == candidate_hw
    for path in [prerequisite_path, ppo_path]:
        hashes[str(path.relative_to(ROOT))] = digest(path)
    for name, expected in hashes.items():
        assert digest(ROOT / name) == expected, name
    write(output / 'DEPENDENCIES.json', dict(inputs=hashes, selected_actor=choice,
        source=str(source.relative_to(ROOT)), candidate=str(candidate.relative_to(ROOT)), predecessor_state=state))

    import numpy as np
    import torch
    from unittest.mock import patch
    import experiments.evaluate_docking_navigation as evaluator
    from experiments.release_recovery import evaluate_recovery
    from experiments.paired_navigation_gate import compare
    from neural_navigation.experiment_budget import ExperimentBudget
    from neural_navigation.goal_pose_runtime import check_files, resolve_runtime

    torch.set_num_threads(1)
    for bundle in [source, candidate]:
        check_files(bundle, read(bundle / 'manifest.json'))
    budget = ExperimentBudget(.35, memory_mib=2800, directory=output)
    physical = output / 'physical'
    physical.mkdir()
    report['selected_actor'] = choice

    def save():
        budget.tick()
        report.update(at=datetime.now(timezone.utc).isoformat(), resource=budget.statistics())
        write(output / 'RESULTS.json', report)

    def finish(reason):
        for name, expected in hashes.items():
            assert digest(ROOT / name) == expected, name
        report.update(completed=True, reason=reason, inputs_verified=len(hashes))
        save()
        write(output / 'STATUS.json', dict(status='complete', reason=reason,
            completed_episodes=report['new_physical_episodes']))

    def run(spec, group, recovery=False, bundle=candidate, variant_name='candidate'):
        environments = []
        env_type = evaluator.StressNavigationEnv
        release = spec['release_s'] if recovery else 0.
        runtime_type = resolve_runtime(read(bundle / 'manifest.json')['required_runtime'])

        class MeasuredEnvironment(env_type):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.gap_label_started = False
                environments.append(self)

            def step(self, action):
                # This affects evaluation labels only, never observations,
                # memory, masks, policy decisions or executed commands.
                if not self.gap_label_started and self.elapsed >= release-1e-8:
                    self.minimum_gap = self._minimum_gap()
                    self.gap_label_started = True
                return super().step(action)

        class VerifiedRuntime(runtime_type):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                if variant_name == 'candidate':
                    assert self.guard.memory.config['free_evidence_frames'] == 3
                    assert self.guard.memory.config['free_evidence_margin_m'] == .03

        write(output / 'STATUS.json', dict(status='physical_configuration_retention', group=group,
            case_id=spec['id'], variant=variant_name, completed_episodes=report['new_physical_episodes']))
        target = 'experiments.release_recovery.resolve_runtime' if recovery else 'neural_navigation.goal_pose_runtime.resolve_runtime'
        with patch.object(evaluator, 'StressNavigationEnv', MeasuredEnvironment), patch(target, return_value=VerifiedRuntime):
            row, trace = (evaluate_recovery if recovery else evaluator.evaluate)(spec, bundle, budget)
        assert len(environments) == 1
        minimum = float(environments[0].minimum_gap)
        row.update(variant=variant_name, minimum_evaluated_dynamic_gap_m=minimum,
            safe_success=bool(row['safe_success'] and minimum >= .04-1e-4),
            input_configuration_from_initialization=True, no_optimizer_updates=True)
        path = physical / (group + '_' + spec['id'] + '_' + variant_name + '.npz')
        np.savez_compressed(path, **trace)
        row.update(trace_path=str(path.relative_to(ROOT)), trace_sha256=digest(path))
        if group == 'fresh8':
            report['frozen_rows'].append(row)
        else:
            report['groups'].setdefault(group, dict(rows=[]))['rows'].append(row)
        report['new_physical_episodes'] += 1
        save()
        print(json.dumps(dict(group=group, **row)), flush=True)
        return row

    def retained(row, reference):
        key = 'post_release_seconds' if row.get('recovery_test') else 'seconds'
        return bool(row['safe_success'] and row['protocol_valid']
            and not any(row[k] for k in ['collisions', 'false_arrival', 'physical_guards'])
            and all(row[k] <= reference[k]*1.2+1e-8 for k in [key, 'travel_m']))

    save()
    for group, specs, recovery in [('normal18', tasks['normal18'], False), ('sensor5', tasks['sensor5'], False),
            ('source_recovery3', recoveries, True), ('heading2', tasks['heading2'], False)]:
        for spec in specs:
            try:
                row = run(spec, group, recovery)
            except ValueError as error:
                if 'frozen prefix command is no longer admissible' not in str(error):
                    raise
                report['prefix_incompatible'] = dict(case_id=spec['id'], error=str(error), no_invalid_action_forced=True)
                finish('Existing physical initialization became inadmissible; no expansion')
                return
            okay = retained(row, source_rows[row['case_id']]) and retained(row, original_rows[row['case_id']])
            report['gates'][row['case_id']] = dict(passed=okay)
            if not okay:
                finish('Fixed safety, outcome or efficiency retention failed; no fresh-pair expansion')
                return
    normal = report['groups']['normal18']['rows']
    report['normal18_against_source'] = compare([source_rows[r['case_id']] for r in normal], normal)
    report['normal18_against_bc4500'] = compare([original_rows[r['case_id']] for r in normal], normal)
    report['all_fixed_retention_passed'] = bool(report['normal18_against_source']['larger_training_gate_passed']
        and report['normal18_against_bc4500']['larger_training_gate_passed'])
    if not report['all_fixed_retention_passed']:
        finish('Aggregate fixed efficiency retention failed; no fresh-pair expansion')
        return
    for case in cases:
        for variant_name, bundle in [('source', source), ('candidate', candidate)]:
            row = run(case, 'fresh8', bundle=bundle, variant_name=variant_name)
            if variant_name == 'candidate' and (any(row[k] for k in ['collisions', 'false_arrival', 'physical_guards'])
                    or row['minimum_evaluated_dynamic_gap_m'] < .04-1e-4):
                finish('Candidate fresh-pair safety failure; stop without promotion')
                return
    before = [row for row in report['frozen_rows'] if row['variant'] == 'source']
    after = [row for row in report['frozen_rows'] if row['variant'] == 'candidate']
    report['fresh_pair_comparison'] = compare(before, after)
    report['all_fresh_retention_passed'] = report['fresh_pair_comparison']['larger_training_gate_passed']
    finish('Prospective configuration retention complete;106s recovery and overall acceptance still unresolved')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    try:
        main(args.output)
    except Exception as error:
        if args.output.is_dir():
            write(args.output / 'FAILURE.json', dict(error=repr(error), automatically_restarted=False,
                goal_achieved=False, deployment_approved=False))
        raise
