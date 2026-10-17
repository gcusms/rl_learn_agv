"""Bounded pure-model validation; stop expansion after any declared group fails."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import torch

from experiments.evaluate_docking_navigation import evaluate
from experiments.experiment_provenance import record_provenance
from experiments.long_risk_actor import LongRiskActor, digest
from neural_navigation.experiment_budget import ExperimentBudget
from neural_navigation.long_risk_observation import contract
from neural_navigation.masked_actions import ACTIONS


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
    bundle = args.model / 'portable'
    fit = json.loads((args.model / 'RESULTS.json').read_text())
    assert fit['completed'] and fit['exported_for_physical_validation']
    manifest = json.loads((bundle / 'manifest.json').read_text())
    assert manifest['observation_contract'] == contract()
    from neural_navigation.goal_pose_runtime import check_files
    check_files(bundle, manifest)
    old_validation_path = BASE / 'early_yield_remaining_validation_20261016/RESULTS.json'
    old_validation = json.loads(old_validation_path.read_text())
    assert old_validation['completed']
    original_path = BASE / 'proactive_augmented_protocol_20261016/fresh_cases.json'
    old_noise_path = BASE / 'early_yield_remaining_protocol_20261016/heldout_cases.json'
    new_noise_path = BASE / 'long_risk_protocol_20261016/new_sensor_cases.json'
    original_all = json.loads(original_path.read_text())
    original = [next(case for case in original_all if case['id'] == 'proactive_fresh_condition_' + suffix)
        for suffix in ('01', '02', '04', '05')]
    groups = [('original_conditions', original),
        ('development_noise', json.loads(old_noise_path.read_text())),
        ('frozen_new_noise', json.loads(new_noise_path.read_text()))]
    paths = [Path(__file__), args.model / 'checkpoint.pt', args.model / 'RESULTS.json',
        args.model / 'demonstrations.npz', args.model / 'parent_actor.pt', bundle / 'manifest.json',
        old_validation_path, original_path, old_noise_path, new_noise_path]
    paths += [bundle / name for name in manifest['files']]
    for folder in ('neural_navigation', 'navrl', 'experiments'):
        paths += list((ROOT / 'training' / folder).rglob('*.py'))
    inputs = {str(path.relative_to(ROOT)): digest(path) for path in set(paths)}
    write(args.output / 'PLAN.json', dict(inputs=inputs, maximum_new_physical_episodes=12,
        ordered_groups=[name for name, _ in groups], all_four_in_each_group_must_be_safe=True,
        original_group_time_and_travel_factor_vs_latest_safe_candidate=1.2,
        stop_expansion_after_group_failure=True, no_teacher_or_motion_insertion=True,
        no_simulator_truth_or_future_frames_in_policy=True,
        unchanged_masks_body_margin_blind_memory_guard_and_deadlines=True,
        new_ppo_steps=0, new_optimizer_updates=0, broader_regression_still_required=True))
    state = torch.load(args.model / 'checkpoint.pt', map_location='cpu', weights_only=False)
    assert state['contract'] == contract() and state['new_ppo_steps'] == 0
    assert state['parent_actor_sha256'] == digest(args.model / 'parent_actor.pt')
    model = LongRiskActor(torch.jit.load(str(args.model / 'parent_actor.pt'), map_location='cpu')).eval()
    model.load_state_dict(state['actor'])
    optimizer = torch.optim.Adam(model.head.parameters(), lr=.002)
    optimizer.load_state_dict(state['optimizer'])
    assert optimizer.state_dict()['state'] and state['torch_rng'].dtype == torch.uint8
    actor = torch.jit.load(str(bundle / 'actor.pt'), map_location='cpu').eval()
    with np.load(args.model / 'demonstrations.npz', allow_pickle=False) as archive:
        data = {key: archive[key].copy() for key in archive.files}
    x = torch.from_numpy(data['observations'])
    with torch.no_grad():
        eager, portable = model(x).numpy(), actor(x).numpy()
    np.testing.assert_array_equal(eager, ACTIONS[data['choices']])
    np.testing.assert_array_equal(portable, eager)
    record_provenance(args.output, bundle, [Path(__file__)])
    budget.tick()
    write(args.output / 'MODEL_ADMISSION.json', dict(completed=True,
        source_decisions_reproduced=len(x), optimizer_and_rng_loaded=True,
        checkpoint_sha256=digest(args.model / 'checkpoint.pt'),
        actual_bc_updates=state['supervised_updates'], new_ppo_steps=0,
        no_claim_of_navigation_success=True))
    physical = args.output / 'physical'
    physical.mkdir()
    report = dict(completed=False, rows=[], gates={}, new_ppo_steps=0,
        new_optimizer_updates=0, actual106_not_claimed_fixed=True,
        full18_extra12_prefix_heading_wait_regressions_still_required=True,
        goal_achieved=False, deployment_approved=False)

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
            result, trace = evaluate(case, bundle, budget)
            file = physical / (group_name + '_' + case['id'] + '.npz')
            np.savez_compressed(file, **trace)
            with torch.no_grad():
                supported = actor.support(torch.from_numpy(trace['observations'])).numpy()
            result.update(group=group_name, variant='long_risk_bc',
                trace_path=str(file.relative_to(ROOT)), trace_sha256=digest(file),
                learned_model_only=True, head_supported_frames=int(supported.sum()))
            report['rows'].append(result)
            group_rows.append(result)
            save()
        safe = len(group_rows) == 4 and all(row['safe_success'] for row in group_rows)
        gate = dict(all_safe=safe, safe=sum(row['safe_success'] for row in group_rows), total=len(group_rows))
        if group_name == 'original_conditions':
            reference = {row['case_id']: row for row in old_validation['rows']
                if row['group'] == 'original_collision_conditions'}
            efficient = all(row['safe_success'] and row['seconds'] <= reference[row['case_id']]['seconds'] * 1.2
                and row['travel_m'] <= reference[row['case_id']]['travel_m'] * 1.2 for row in group_rows)
            gate['time_and_travel_within_1_2_factor'] = efficient
            safe = safe and efficient
        gate['passed'] = safe
        report['gates'][group_name] = gate
        save()
        if not safe:
            report.update(completed=True, reason='Declared safety/efficiency group failed; expansion stopped')
            break
    else:
        report.update(completed=True, reason='All twelve bounded checks passed; broader regression still required')
    for name, expected in inputs.items():
        assert digest(ROOT / name) == expected, name
    save()
    write(args.output / 'STATUS.json', dict(status='complete', finished_episodes=len(report['rows']),
        gates=report['gates']))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    try:
        main(args)
    except Exception as error:
        if args.output.exists():
            write(args.output / 'FAILURE.json', dict(error=repr(error), automatically_restarted=False))
        raise
