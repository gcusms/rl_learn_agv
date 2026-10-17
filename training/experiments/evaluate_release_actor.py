"""Pure-model release, retention and untouched-seed gates, in that order."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import torch

from experiments.evaluate_docking_navigation import evaluate
from experiments.experiment_provenance import record_provenance
from experiments.long_risk_actor import digest
from experiments.release_actor import ReleaseActor
from neural_navigation.experiment_budget import ExperimentBudget
from neural_navigation.masked_actions import ACTIONS
from neural_navigation.release_observation import contract


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'reports/stop_feedback_20261015'


def write(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def main(args):
    args.model, args.output = args.model.resolve(), args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    budget = ExperimentBudget(.35, memory_mib=3072, directory=args.output)
    fit_path = args.model / 'RESULTS.json'
    fit = json.loads(fit_path.read_text())
    assert fit['completed'] and fit['exported_for_physical_validation']
    bundle = args.model / 'portable'
    manifest = json.loads((bundle / 'manifest.json').read_text())
    assert manifest['observation_contract'] == contract()
    from neural_navigation.goal_pose_runtime import check_files
    check_files(bundle, manifest)
    old_path = BASE / 'long_risk_validation_20261016/RESULTS.json'
    old = json.loads(old_path.read_text())
    assert old['completed'] and len(old['rows']) == 8
    original_path = BASE / 'proactive_augmented_protocol_20261016/fresh_cases.json'
    development_path = BASE / 'early_yield_remaining_protocol_20261016/heldout_cases.json'
    fresh_path = BASE / 'long_risk_protocol_20261016/new_sensor_cases.json'
    originals = json.loads(original_path.read_text())
    development = json.loads(development_path.read_text())
    fresh = json.loads(fresh_path.read_text())
    lookup = {case['id']: case for case in [*originals, *development]}
    repaired = [lookup['early_yield_sensor_holdout_05_0']]
    retained = [lookup[row['case_id']] for row in old['rows'] if row['safe_success']]
    assert len(retained) == 7 and len(fresh) == 4
    groups = [('repaired_wait_condition', repaired), ('retained_development', retained),
        ('frozen_new_noise', fresh)]
    paths = [Path(__file__), fit_path, args.model / 'checkpoint.pt', args.model / 'parent_actor.pt',
        args.model / 'demonstrations.npz', old_path, original_path, development_path, fresh_path,
        bundle / 'manifest.json']
    paths += [bundle / name for name in manifest['files']]
    for folder in ['neural_navigation', 'navrl', 'experiments']:
        paths += list((ROOT / 'training' / folder).rglob('*.py'))
    for _, cases in groups:
        for case in cases:
            paths += [p for p in (ROOT / 'training' / case['scene']).iterdir() if p.is_file()]
    inputs = {str(path.relative_to(ROOT)): digest(path) for path in set(paths)}
    write(args.output / 'PLAN.json', dict(inputs=inputs, ordered_groups=[name for name, _ in groups],
        maximum_new_physical_episodes=12, stop_after_any_group_failure=True,
        retained_time_and_travel_factor_vs_risk_head=1.2,
        fresh_four_seeds_frozen_before_both_head_fits=True,
        development_cases_are_not_heldout=True, no_teacher_or_motion_insertion=True,
        original_guards_masks_body_margin_blind_memory_and_deadlines=True,
        new_optimizer_updates=0, new_ppo_steps=0, full_regression_required_afterward=True))
    state = torch.load(args.model / 'checkpoint.pt', map_location='cpu', weights_only=False)
    assert state['contract'] == contract() and state['new_ppo_steps'] == 0
    assert state['parent_actor_sha256'] == digest(args.model / 'parent_actor.pt')
    actor = ReleaseActor(torch.jit.load(str(args.model / 'parent_actor.pt'), map_location='cpu')).eval()
    actor.load_state_dict(state['actor'])
    optimizer = torch.optim.Adam(actor.head.parameters(), lr=.001)
    optimizer.load_state_dict(state['optimizer'])
    assert optimizer.state_dict()['state'] and state['torch_rng'].dtype == torch.uint8
    portable = torch.jit.load(str(bundle / 'actor.pt'), map_location='cpu').eval()
    with np.load(args.model / 'demonstrations.npz', allow_pickle=False) as archive:
        data = {key: archive[key].copy() for key in archive.files}
    x = torch.from_numpy(data['observations'])
    with torch.no_grad():
        np.testing.assert_array_equal(actor(x).numpy(), ACTIONS[data['choices']])
        np.testing.assert_array_equal(portable(x).numpy(), actor(x).numpy())
    record_provenance(args.output, bundle, [Path(__file__)])
    budget.tick()
    write(args.output / 'MODEL_ADMISSION.json', dict(completed=True,
        reproduced_source_decisions=len(x), actual_supervised_updates=state['supervised_updates'],
        checkpoint_sha256=digest(args.model / 'checkpoint.pt'),
        optimizer_and_rng_loaded=True, no_navigation_acceptance_claim=True))
    physical = args.output / 'physical'
    physical.mkdir()
    report = dict(completed=False, rows=[], gates={}, new_ppo_steps=0,
        new_optimizer_updates=0, actual106_not_claimed_fixed=True,
        full_regression_still_required=True, goal_achieved=False, deployment_approved=False)

    def save():
        budget.tick()
        report['resource'] = budget.statistics()
        report['at'] = datetime.now(timezone.utc).isoformat()
        write(args.output / 'RESULTS.json', report)

    for group_name, cases in groups:
        group_rows = []
        for case in cases:
            write(args.output / 'STATUS.json', dict(status='pure_model_physical', group=group_name,
                case_id=case['id'], finished_episodes=len(report['rows'])))
            row, trace = evaluate(case, bundle, budget)
            file = physical / (group_name + '_' + case['id'] + '.npz')
            np.savez_compressed(file, **trace)
            row.update(group=group_name, variant='measured_release_context',
                trace_path=str(file.relative_to(ROOT)), trace_sha256=digest(file),
                learned_model_only=True, no_teacher=True)
            report['rows'].append(row)
            group_rows.append(row)
            save()
        passed = all(row['safe_success'] for row in group_rows)
        gate = dict(safe=sum(row['safe_success'] for row in group_rows), total=len(group_rows), all_safe=passed)
        if group_name == 'retained_development':
            reference = {row['case_id']: row for row in old['rows']}
            efficient = all(row['safe_success'] and row['seconds'] <= reference[row['case_id']]['seconds'] * 1.2
                and row['travel_m'] <= reference[row['case_id']]['travel_m'] * 1.2 for row in group_rows)
            gate['time_and_travel_within_1_2_factor'] = efficient
            passed = passed and efficient
        gate['passed'] = passed
        report['gates'][group_name] = gate
        save()
        if not passed:
            report.update(completed=True, reason='Declared release/safety/efficiency gate failed; expansion stopped')
            break
    else:
        report.update(completed=True, reason='Twelve bounded checks passed; full regression still required')
    for name, expected in inputs.items():
        assert digest(ROOT / name) == expected, name
    save()
    write(args.output / 'STATUS.json', dict(status='complete', finished_episodes=len(report['rows']), gates=report['gates']))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    main(parser.parse_args())
