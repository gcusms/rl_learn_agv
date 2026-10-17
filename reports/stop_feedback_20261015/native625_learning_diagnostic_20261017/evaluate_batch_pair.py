"""Evaluate the pre-PPO frozen new-seed cases after fixed retention passes."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import time


ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / 'reports/stop_feedback_20261015'
SOURCE = BASE / 'native625_frozen_holdout_20261016'
DEPENDENCY = 'rl-native625-batch-credit-check-20261017.service'
INVOCATION = '040e38b20f3e48848ebe4f4e36840fd3'


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')
    temporary.replace(path)


def main(args):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    cases_path, frozen_path = SOURCE / 'CASES.json', SOURCE / 'PLAN.json'
    frozen = read(frozen_path)
    assert frozen['frozen_before_ppo'] and digest(cases_path) == frozen['cases_sha256']
    cases = read(cases_path)
    assert len(cases) == len({c['id'] for c in cases}) == 8
    files = [Path(__file__), cases_path, frozen_path]
    for folder in ['experiments', 'neural_navigation', 'navrl']:
        files += list((ROOT / 'training' / folder).rglob('*.py'))
    files += [ROOT / 'training/neural_navigation/static_tracking_kernel.cpp']
    for scene in {c['scene'] for c in cases}:
        files += [p for p in (ROOT / 'training' / scene).iterdir() if p.is_file()]
    hashes = {str(p.relative_to(ROOT)):digest(p) for p in set(files)}
    write(output / 'PLAN.json', dict(inputs=hashes, exact_predecessor=DEPENDENCY,
        expected_invocation=INVOCATION, frozen_cases_sha256=digest(cases_path),
        maximum_physical_episodes=16, variants=['bc4500', 'batch_credit_ppo'],
        new_seeds_noise_and_timing_same_known_map=True,
        not_unseen_map_or_real_robot_acceptance=True,
        stop_on_candidate_collision_false_arrival_or_physical_guard=True,
        no_optimizer_updates=True, new_ppo_steps=0, deployment_approved=False))
    while True:
        result = subprocess.run(['systemctl', '--user', 'show', DEPENDENCY,
            '--property=ActiveState,MainPID,InvocationID,ExecMainStatus'],
            text=True, capture_output=True, check=True)
        observed = dict(line.split('=',1) for line in result.stdout.splitlines() if '=' in line)
        if observed['InvocationID'] not in ('', INVOCATION):
            raise RuntimeError('Pinned validation predecessor was restarted or replaced')
        if observed['ActiveState'] in ('inactive', 'failed') and observed['MainPID'] == '0':
            break
        write(output / 'STATUS.json', dict(status='waiting_for_exact_post_ppo_validation',
            dependency=observed, new_physical_episodes=0, new_ppo_steps=0))
        time.sleep(5.)
    previous_path = BASE / 'native625_batch_credit_validation_20261017/RESULTS.json'
    previous = read(previous_path) if previous_path.exists() else {}
    ready = bool(observed['ActiveState'] == 'inactive' and observed['ExecMainStatus'] == '0'
        and previous.get('completed') and previous.get('physical_validation_started')
        and previous.get('all_development_retention_passed'))
    if not ready:
        write(output / 'RESULTS.json', dict(completed=True, physical_validation_started=False,
            reason='Exact predecessor did not pass fixed post-PPO retention; frozen-pair expansion skipped',
            predecessor_result_sha256=digest(previous_path) if previous_path.exists() else None,
            new_physical_episodes=0, new_ppo_steps=0, goal_achieved=False, deployment_approved=False))
        write(output / 'STATUS.json', dict(status='not_started_predecessor_gate_failed'))
        return

    import numpy as np
    import torch
    from experiments.evaluate_docking_navigation import evaluate
    from experiments.paired_navigation_gate import compare
    from neural_navigation.experiment_budget import ExperimentBudget
    from neural_navigation.goal_pose_runtime import check_files

    parent_path = BASE / 'normal_suite_wait_validation_v2_20261016/RESULTS.json'
    parent_validation = read(parent_path)
    assert parent_validation['completed'] and parent_validation['normal18_pair']['all_regression_checks_passed']
    parent = ROOT / parent_validation['selected_model'] / 'portable'
    candidate = ROOT / 'models/stop_feedback_20261015/native625_batch_credit_s231017/portable'
    training_plan = BASE / 'native625_batch_credit_pilot_20261017/PLAN.json'
    assert str(cases_path.relative_to(ROOT)) not in read(training_plan)['inputs']
    assert int(read(candidate / 'manifest.json')['additional_ppo_steps']) == previous['validated_candidate_ppo_steps'] > 0
    for bundle in [parent, candidate]:
        manifest = read(bundle / 'manifest.json')
        check_files(bundle, manifest)
        assert not manifest['deployment_approved']
        files = [bundle / 'manifest.json'] + [bundle / name for name in manifest['files']]
        hashes.update({str(p.relative_to(ROOT)):digest(p) for p in files})
    for name in ['hardware.json', 'localizer.pt', 'router.pt', 'decision_feedback.json',
            'release_observation.json', 'docking_contract.json', 'localization_contract.json']:
        assert digest(parent / name) == digest(candidate / name), name
    hashes.update({str(p.relative_to(ROOT)):digest(p) for p in [previous_path, parent_path, training_plan]})
    for name, expected in hashes.items():
        assert digest(ROOT / name) == expected, name
    write(output / 'DEPENDENCIES.json', dict(inputs=hashes,
        validated_candidate_ppo_steps=previous['validated_candidate_ppo_steps'],
        identical_hardware_localizer_router_feedback_and_docking=True))
    torch.set_num_threads(1)
    budget = ExperimentBudget(.35, memory_mib=2800, directory=output)
    physical = output / 'physical'
    physical.mkdir()
    report = dict(completed=False, physical_validation_started=True, rows=[],
        new_physical_episodes=0, new_optimizer_updates=0, new_ppo_steps=0,
        validated_candidate_ppo_steps=previous['validated_candidate_ppo_steps'],
        goal_achieved=False, deployment_approved=False)

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
        for variant, bundle in [('bc4500', parent), ('batch_credit_ppo', candidate)]:
            write(output / 'STATUS.json', dict(status='paired_physical_validation',
                case_id=case['id'], variant=variant, completed_episodes=report['new_physical_episodes']))
            row, trace = evaluate(case, bundle, budget)
            path = physical / (case['id']+'_'+variant+'.npz')
            np.savez_compressed(path, **trace)
            row.update(variant=variant, bundle=str(bundle.relative_to(ROOT)),
                trace_path=str(path.relative_to(ROOT)), trace_sha256=digest(path),
                same_known_map_new_seed_and_perturbation=True)
            report['rows'].append(row)
            report['new_physical_episodes'] += 1
            save()
            print(json.dumps(row), flush=True)
            if variant == 'batch_credit_ppo' and any(row[k] for k in ['collisions', 'false_arrival', 'physical_guards']):
                finish('Candidate safety failure; frozen-pair expansion stopped')
                return
    before = [r for r in report['rows'] if r['variant']=='bc4500']
    after = [r for r in report['rows'] if r['variant']=='batch_credit_ppo']
    assert len(before) == len(after) == 8
    report['paired_comparison'] = compare(before, after)
    report['all_frozen_checks_passed'] = report['paired_comparison']['larger_training_gate_passed']
    report['wait_totals'] = {name:dict(voluntary_wait_frames=sum(r['voluntary_wait_frames'] for r in rows),
        only_stop_frames=sum(r['only_stop_frames'] for r in rows)) for name,rows in [('bc4500',before),('batch_credit_ppo',after)]}
    finish('Frozen new-seed paired evaluation complete; no automatic deployment or unseen-map claim')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    try:
        main(args)
    except Exception as error:
        if args.output.is_dir():
            write(args.output / 'FAILURE.json', dict(error=repr(error), automatically_restarted=False))
        raise
