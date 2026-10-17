"""Bounded post-hoc diagnosis of the opposite PPO full-mission outcomes.

Two existing actors are each evaluated with the already built three-frame
memory variant, on both previously frozen tasks. Original-configuration rows
are immutable completed references. This is development diagnosis, not a fresh
test set, training, model selection, or deployment approval.
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
DEPENDENCY = 'rl-memory-joint-retention-20261017.service'
INVOCATION = '84a9c84271294b5c8f0d269de34b6cb2'
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
    capture = (BASE / 'native625_local_speedup_retention_20261017/RESULTS.json').read_bytes()
    references = [r for r in json.loads(capture)['frozen_rows']
                  if r['case_id'] in {'pre_ppo_holdout_dynamic_00', 'pre_ppo_holdout_dynamic_01'}]
    assert len(references) == 4
    reference_path = output / 'ORIGINAL_ROWS.json'
    write(reference_path, dict(rows=references, source_report_sha256_at_capture=hashlib.sha256(capture).hexdigest()))
    cases = read(HERE / 'CASES.json')
    assert {c['id'] for c in cases} == {r['case_id'] for r in references}
    variants = read(VARIANTS / 'VARIANTS.json')['variants']
    assert {v['name'] for v in variants} == {'bc4500', 'ppo1665'}
    for v in variants:
        source, candidate = ROOT / v['source'], ROOT / v['output']
        source_hw, candidate_hw = [read(p / 'hardware.json') for p in [source, candidate]]
        assert candidate_hw['near_field_memory']['free_evidence_frames'] == 3
        assert candidate_hw['near_field_memory']['free_evidence_margin_m'] == .03
        assert candidate_hw['near_field_memory']['free_evidence_max_gap_s'] == .4
        assert candidate_hw['near_field_memory']['static_footprint_margin_m'] == .04
        assert candidate_hw['robot_footprint_m'] == [.393924904, .329460022]
        for field in ['free_evidence_frames', 'free_evidence_margin_m', 'free_evidence_max_gap_s']:
            candidate_hw['near_field_memory'].pop(field)
            source_hw['near_field_memory'].pop(field, None)
        assert candidate_hw == source_hw
        for name in ['actor.pt', 'localizer.pt', 'router.pt', 'parent_actor.pt', 'previous_risk_actor.pt',
                     'decision_feedback.json', 'release_observation.json', 'docking_contract.json']:
            assert digest(source / name) == digest(candidate / name), (v['name'], name)
    files = [Path(__file__), HERE / 'CASES.json', reference_path, VARIANTS / 'VARIANTS.json',
             BASE / 'native625_learning_diagnostic_20261017/WHOLE_MISSION_FROZEN_REGRESSION_AUDIT.json',
             BASE / 'native625_learning_diagnostic_20261017/WHOLE_MISSION_FROZEN_GAIN_AUDIT.json']
    for r in references:
        path = ROOT / r['trace_path']
        assert digest(path) == r['trace_sha256']
        files.append(path)
    for v in variants:
        for bundle in [ROOT / v['source'], ROOT / v['output']]:
            manifest = read(bundle / 'manifest.json')
            assert not manifest['deployment_approved']
            files += [bundle / 'manifest.json'] + [bundle / name for name in manifest['files']]
    for folder in ['experiments', 'neural_navigation', 'navrl']:
        files += list((ROOT / 'training' / folder).rglob('*.py'))
    files.append(ROOT / 'training/neural_navigation/static_tracking_kernel.cpp')
    for scene in {c['scene'] for c in cases}:
        files += [p for p in (ROOT / 'training' / scene).iterdir() if p.is_file()]
    hashes = {str(p.relative_to(ROOT)): digest(p) for p in set(files)}
    write(output / 'PLAN.json', dict(at=datetime.now(timezone.utc).isoformat(), inputs=hashes,
        dependency=DEPENDENCY, expected_invocation=INVOCATION, maximum_new_physical_episodes=4,
        scope='Post-hoc development diagnosis; both tasks already evaluated',
        hypothesis='Finite multi-frame free evidence may reduce retained-point stop vetoes without losing either successful reference',
        case_selection='Both opposite outcomes selected, not just the newly gained success',
        no_new_optimizer_updates=True, new_ppo_steps=0, no_heldout_trace_training=True,
        original625_observation_12actions_footprint_and_4cm_margin_retained=True,
        no_old_command_prefix_or_timed_switch_or_forced_movement=True,
        no_model_promotion_or_original_gate_override=True, no_parallel_physical_job=True,
        soft_cpu_cores=.35, memory_hard_max_mib=3072, goal_achieved=False, deployment_approved=False))
    while True:
        state = dict(line.split('=', 1) for line in subprocess.run(
            ['systemctl', '--user', 'show', DEPENDENCY,
             '--property=MainPID,ActiveState,InvocationID,ExecMainStatus'],
            capture_output=True, text=True, check=True).stdout.splitlines() if '=' in line)
        if state['InvocationID'] not in ('', INVOCATION):
            raise RuntimeError('Pinned predecessor was restarted or replaced')
        if state['MainPID'] == '0' and state['ActiveState'] in ('inactive', 'failed'):
            break
        write(output / 'STATUS.json', dict(status='waiting_for_exact_joint_retention',
            dependency=state, new_physical_episodes=0, new_ppo_steps=0))
        time.sleep(5.)
    report = dict(completed=False, rows=[], new_physical_episodes=0, new_optimizer_updates=0,
        new_ppo_steps=0, same_known_map=True, post_hoc_development_cases=True,
        no_fresh_generalization_or_deployment_claim=True, goal_achieved=False, deployment_approved=False)

    def skipped(reason):
        report.update(completed=True, reason=reason)
        write(output / 'RESULTS.json', report)
        write(output / 'STATUS.json', dict(status='skipped', reason=reason, new_physical_episodes=0))

    if state['ActiveState'] != 'inactive' or state['ExecMainStatus'] != '0':
        skipped('Exact predecessor failed; no further physics')
        return
    prereq_path = BASE / 'memory_from_start_20261017/RESULTS.json'
    joint_path = BASE / 'memory_joint_retention_20261017/RESULTS.json'
    prerequisite, joint = read(prereq_path), read(joint_path)
    assert prerequisite['completed'] and joint['completed']
    for previous in [prerequisite, joint]:
        rows = list(previous.get('rows', [])) + list(previous.get('frozen_rows', []))
        rows += [r for group in previous.get('groups', {}).values() for r in group['rows']]
        if any(any(r.get(k, False) for k in ['collisions', 'false_arrival', 'physical_guards'])
               or min(r.get('minimum_post_release_dynamic_gap_m', 1.),
                      r.get('minimum_evaluated_dynamic_gap_m', 1.)) < .04 - 1e-4 for r in rows):
            skipped('Previous configuration experiment recorded a safety violation; diagnose it first')
            return
    hashes.update({str(p.relative_to(ROOT)): digest(p) for p in [prereq_path, joint_path]})
    for name, expected in hashes.items():
        assert digest(ROOT / name) == expected, name
    write(output / 'DEPENDENCIES.json', dict(inputs=hashes, predecessor_terminal=state))

    import numpy as np
    import torch
    from unittest.mock import patch
    import experiments.evaluate_docking_navigation as evaluator
    from neural_navigation.experiment_budget import ExperimentBudget
    from neural_navigation.goal_pose_runtime import check_files, resolve_runtime

    torch.set_num_threads(1)
    budget = ExperimentBudget(.35, memory_mib=2800, directory=output)
    (output / 'physical').mkdir()

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

    save()
    for case in cases:
        for variant in sorted(variants, key=lambda v: v['name']):
            bundle = ROOT / variant['output']
            check_files(bundle, read(bundle / 'manifest.json'))
            environments, runtimes = [], []
            env_type = evaluator.StressNavigationEnv
            runtime_type = resolve_runtime(read(bundle / 'manifest.json')['required_runtime'])

            class MeasuredEnvironment(env_type):
                def __init__(self, *args, **kwargs):
                    super().__init__(*args, **kwargs)
                    environments.append(self)

            class VerifiedRuntime(runtime_type):
                def __init__(self, *args, **kwargs):
                    super().__init__(*args, **kwargs)
                    assert self.guard.memory.config['free_evidence_frames'] == 3
                    assert self.guard.memory.config['free_evidence_margin_m'] == .03
                    assert self.guard.memory.config['static_footprint_margin_m'] == .04
                    runtimes.append(self)

            write(output / 'STATUS.json', dict(status='physical_post_hoc_pair_diagnosis',
                case_id=case['id'], variant=variant['name'], completed_episodes=report['new_physical_episodes']))
            with patch.object(evaluator, 'StressNavigationEnv', MeasuredEnvironment), \
                    patch('neural_navigation.goal_pose_runtime.resolve_runtime', return_value=VerifiedRuntime):
                row, trace = evaluator.evaluate(case, bundle, budget)
            assert len(environments) == len(runtimes) == 1
            minimum = float(environments[0].minimum_gap)
            row.update(variant=variant['name'], memory_verified_from_initialization=True,
                minimum_dynamic_gap_m=minimum, safe_success=bool(row['safe_success'] and minimum >= .04-1e-4))
            path = output / 'physical' / (case['id'] + '_' + variant['name'] + '.npz')
            np.savez_compressed(path, **trace)
            row.update(trace_path=str(path.relative_to(ROOT)), trace_sha256=digest(path),
                bundle=str(bundle.relative_to(ROOT)))
            source_variant = 'parent' if variant['name'] == 'bc4500' else 'candidate'
            reference = next(r for r in references if r['case_id'] == case['id'] and r['variant'] == source_variant)
            row['original_configuration_reference'] = {k: reference[k] for k in
                ['safe_success', 'seconds', 'travel_m', 'only_stop_frames', 'voluntary_wait_frames']}
            report['rows'].append(row)
            report['new_physical_episodes'] += 1
            save()
            print(json.dumps(row), flush=True)
            if any(row[k] for k in ['collisions', 'false_arrival', 'physical_guards']) or minimum < .04-1e-4:
                finish('Safety check failed; remaining cases stopped, no model promotion')
                return
    report['both_cases_safe_by_variant'] = {v['name']: all(r['safe_success'] for r in report['rows']
        if r['variant'] == v['name']) for v in variants}
    finish('Four diagnostic runs complete; full retention, late106s and real-car verification still required')


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
