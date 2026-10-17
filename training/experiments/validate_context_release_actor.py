"""Serial, gated physics regression of the sensor-context release candidate."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import torch

from experiments.context_release_actor import ContextReleaseActor, contract as support_contract
from experiments.evaluate_docking_navigation import evaluate
from experiments.long_risk_actor import digest
from experiments.paired_navigation_gate import compare
from experiments.release_recovery import evaluate_recovery
from neural_navigation.experiment_budget import ExperimentBudget
from neural_navigation.goal_pose_runtime import check_files
from neural_navigation.masked_actions import ACTIONS


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'reports/stop_feedback_20261015'


def write(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def main(args):
    output, protocol, model = [getattr(args, k).resolve() for k in ['output', 'protocol', 'model']]
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    budget = ExperimentBudget(.35, memory_mib=3072, directory=output)
    plan = json.loads((protocol / 'PLAN.json').read_text())
    fit = json.loads((model / 'RESULTS.json').read_text())
    assert fit['completed'] and fit['exported_for_physical_validation']
    assert datetime.fromisoformat(plan['at']) < datetime.fromisoformat(fit['at'])
    for name, expected in plan['inputs'].items():
        assert digest(ROOT / name) == expected, name
    assert digest(model / 'checkpoint.pt') == fit['checkpoint_sha256']
    state = torch.load(model / 'checkpoint.pt', map_location='cpu', weights_only=False)
    assert state['learning_support'] == support_contract() and state['new_ppo_steps'] == 0
    actor = ContextReleaseActor(torch.jit.load(str(model / 'parent_actor.pt'), map_location='cpu')).eval()
    actor.load_state_dict(state['actor'])
    bundle = model / 'portable'
    manifest = json.loads((bundle / 'manifest.json').read_text())
    check_files(bundle, manifest)
    portable = torch.jit.load(str(bundle / 'actor.pt'), map_location='cpu').eval()
    with np.load(model / 'demonstrations.npz', allow_pickle=False) as data:
        x = torch.from_numpy(data['observations'].copy())
        expected = ACTIONS[data['choices']]
    with torch.no_grad():
        np.testing.assert_array_equal(actor(x).numpy(), expected)
        np.testing.assert_array_equal(portable(x).numpy(), expected)
    del actor, portable, state, x
    tasks = json.loads((protocol / 'tasks.json').read_text())
    files = [Path(__file__), protocol / 'PLAN.json', protocol / 'tasks.json', model / 'checkpoint.pt',
        model / 'RESULTS.json', bundle / 'manifest.json']
    files += [bundle / name for name in manifest['files']]
    files += [ROOT / name for name in plan['inputs']]
    hashes = {str(p.relative_to(ROOT)): digest(p) for p in set(files)}
    write(output / 'PLAN.json', dict(inputs=hashes, protocol=plan,
        maximum_physical_episodes=43, new_ppo_steps=0, deployment_approved=False))
    (output / 'driver_snapshot.py').write_text(Path(__file__).read_text())
    physical = output / 'physical'; physical.mkdir()
    report = dict(completed=False, groups={}, gates={}, new_physical_episodes=0,
        new_ppo_steps=0, goal_achieved=False, deployment_approved=False,
        all_tests_development=True, no_unseen_map_or_real_car_claim=True)

    def save():
        budget.tick()
        report.update(at=datetime.now(timezone.utc).isoformat(), resource=budget.statistics())
        write(output / 'RESULTS.json', report)

    def finish(reason):
        for name, expected in hashes.items():
            assert digest(ROOT / name) == expected, name
        assert report['new_physical_episodes'] <= 43
        report.update(completed=True, reason=reason, frozen_inputs_verified=len(hashes))
        save()
        write(output / 'COMPLETION_CHECKS.json', dict(completed=True, inputs_verified=len(hashes),
            service_exit_is_not_navigation_acceptance=True))
        write(output / 'STATUS.json', dict(status='complete', reason=reason,
            completed_episodes=report['new_physical_episodes']))

    def run(specification, group, recovery=False):
        write(output / 'STATUS.json', dict(status='pure_model_physics', group=group,
            case_id=specification['id'], completed_episodes=report['new_physical_episodes']))
        print(json.dumps(dict(starting=specification['id'], group=group)), flush=True)
        row, trace = (evaluate_recovery if recovery else evaluate)(specification, bundle, budget)
        path = physical / (group + '_' + specification['id'] + '.npz')
        np.savez_compressed(path, **trace)
        row.update(trace_path=str(path.relative_to(ROOT)), trace_sha256=digest(path),
            variant=model.name, pure_learned_model=True, no_teacher_after_initialization=True)
        report['groups'].setdefault(group, dict(rows=[]))['rows'].append(row)
        report['new_physical_episodes'] += 1
        save()
        print(json.dumps(row), flush=True)
        return row

    def paired_gate(group, reference):
        rows = report['groups'][group]['rows']
        paired = compare(reference, rows)
        report['groups'][group]['paired'] = paired
        passed = all(r['safe_success'] for r in rows) and paired['all_regression_checks_passed']
        report['gates'][group] = dict(passed=bool(passed))
        save()
        return passed

    save()
    for spec in tasks['actual95_actual90']:
        run(spec, 'actual95_actual90', True)
    if not paired_gate('actual95_actual90', tasks['reference']['actual95_actual90']):
        finish('Original 95/90 recovery gate failed; no wider expansion')
        return
    run(tasks['delayed_wait'], 'delayed_wait', True)
    if not paired_gate('delayed_wait', tasks['reference']['delayed_wait']):
        finish('Delayed 2.4-second recovery gate failed; no wider expansion')
        return
    for case in tasks['heading2']:
        run(case, 'heading2')
    if not paired_gate('heading2', tasks['reference']['heading2']):
        finish('Directional docking gate failed; no wider expansion')
        return
    for case in tasks['sensor5']:
        row = run(case, 'sensor5')
        ref = next(r for r in tasks['reference']['sensor5'] if r['case_id'] == row['case_id'])
        passed = (row['safe_success'] and row['seconds'] <= ref['seconds'] * 1.2 + 1e-8
            and row['travel_m'] <= ref['travel_m'] * 1.2 + 1e-8)
        report['gates'][case['id']] = dict(passed=bool(passed))
        if not passed:
            finish('Sensor retention safety or efficiency gate failed; no wider expansion')
            return
    for case in tasks['normal18']:
        run(case, 'normal18')
    if not paired_gate('normal18', tasks['reference']['normal18']):
        finish('Normal18 safety or efficiency regression; no wider expansion')
        return
    for case in tasks['extra12']:
        run(case, 'extra12')
    for spec in tasks['prefixes2']:
        run(spec, 'prefixes2', True)
    extra = paired_gate('extra12', tasks['reference']['extra12'])
    prefixes = paired_gate('prefixes2', tasks['reference']['prefixes2'])
    if not (extra and prefixes):
        finish('Extra12 or prefix regression gate failed; actual106 not expanded')
        return
    run(tasks['actual106'], 'actual106_diagnostic', True)
    report['all_physical_tasks_safe'] = all(r['safe_success']
        for group in report['groups'].values() for r in group['rows'])
    finish('All declared development groups finished; independent validation still required')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['model', 'protocol', 'output']:
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    try:
        main(args)
    except Exception as error:
        if args.output.is_dir():
            write(args.output / 'FAILURE.json', dict(error=repr(error), automatically_restarted=False))
        raise
