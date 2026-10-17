"""Bounded paired closed-loop check of the release-context supervised fits."""
import argparse
import copy
from datetime import datetime, timezone
import gc
import json
from pathlib import Path

import numpy as np
import torch

from experiments.evaluate_docking_navigation import evaluate
from experiments.experiment_provenance import record_provenance
from experiments.long_risk_actor import digest
from experiments.release_actor import ReleaseActor
from neural_navigation.experiment_budget import ExperimentBudget
from neural_navigation.goal_pose_runtime import check_files
from neural_navigation.masked_actions import ACTIONS
from neural_navigation.release_observation import contract


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'reports/stop_feedback_20261015'


def write(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def admit(model):
    fit = json.loads((model / 'RESULTS.json').read_text())
    assert fit['completed'] and fit['exported_for_physical_validation']
    assert digest(model / 'checkpoint.pt') == fit['checkpoint_sha256']
    bundle = model / 'portable'
    manifest = json.loads((bundle / 'manifest.json').read_text())
    assert manifest['observation_contract'] == contract()
    check_files(bundle, manifest)
    state = torch.load(model / 'checkpoint.pt', map_location='cpu', weights_only=False)
    assert state['contract'] == contract() and state['new_ppo_steps'] == 0
    assert state['parent_actor_sha256'] == digest(model / 'parent_actor.pt')
    actor = ReleaseActor(torch.jit.load(str(model / 'parent_actor.pt'), map_location='cpu')).eval()
    actor.load_state_dict(state['actor'])
    optimizer = torch.optim.Adam(actor.head.parameters(), lr=.001)
    optimizer.load_state_dict(state['optimizer'])
    assert optimizer.state_dict()['state'] and state['torch_rng'].dtype == torch.uint8
    portable = torch.jit.load(str(bundle / 'actor.pt'), map_location='cpu').eval()
    with np.load(model / 'demonstrations.npz', allow_pickle=False) as data:
        x = torch.from_numpy(data['observations'].copy())
        expected = ACTIONS[data['choices']]
    with torch.no_grad():
        np.testing.assert_array_equal(actor(x).numpy(), expected)
        np.testing.assert_array_equal(portable(x).numpy(), expected)
    result = dict(model=str(model.relative_to(ROOT)), reproduced_source_decisions=len(x),
        actual_supervised_updates=state['supervised_updates'],
        checkpoint_sha256=digest(model / 'checkpoint.pt'), optimizer_and_rng_loaded=True,
        no_navigation_acceptance_claim=True)
    del actor, portable, optimizer, state, x
    gc.collect()
    return result, manifest


def main(args):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    budget = ExperimentBudget(.35, memory_mib=3072, directory=output)
    fit_path = BASE / 'release_phase_fit_20261016/RESULTS.json'
    old_path = BASE / 'release_validation_20261016/RESULTS.json'
    fresh_path = BASE / 'long_risk_protocol_20261016/new_sensor_cases.json'
    development_path = BASE / 'early_yield_remaining_protocol_20261016/heldout_cases.json'
    fits = json.loads(fit_path.read_text())
    assert fits['completed'] and len(fits['rows']) == 2
    models = {row['mode']: ROOT / row['model'] for row in fits['rows']}
    assert list(models) == ['stop_move', 'recovery_phase']
    old = json.loads(old_path.read_text())
    assert old['completed']
    known_failure = next(c for c in json.loads(fresh_path.read_text())
        if c['id'] == 'long_risk_sensor_05_1')
    retention = next(c for c in json.loads(development_path.read_text())
        if c['id'] == 'early_yield_sensor_holdout_05_0')
    reference = next(r for r in old['rows'] if r['case_id'] == retention['id'])
    assert reference['safe_success']
    new_noise = copy.deepcopy(known_failure)
    new_noise.update(id='release_phase_frozen_noise_05_2', seed=261016983,
        pilot_category='new_noise_seed_same_development_map_frozen_after_fit')
    case_path = output / 'cases.json'
    write(case_path, dict(known_failure=known_failure, retention=retention, new_noise=new_noise))
    files = [Path(__file__), fit_path, old_path, fresh_path, development_path, case_path,
        BASE / 'release_recovery_sources_v2_20261016/RESULTS.json',
        BASE / 'release_recovery_sources_v2_20261016/demonstrations.npz']
    for model in models.values():
        manifest = json.loads((model / 'portable/manifest.json').read_text())
        files += [model / name for name in ['RESULTS.json', 'checkpoint.pt',
            'parent_actor.pt', 'demonstrations.npz', 'portable/manifest.json']]
        files += [model / 'portable' / name for name in manifest['files']]
    for folder in ['neural_navigation', 'navrl', 'experiments']:
        files += list((ROOT / 'training' / folder).rglob('*.py'))
    for case in [known_failure, retention, new_noise]:
        files += [p for p in (ROOT / 'training' / case['scene']).iterdir() if p.is_file()]
    hashes = {str(p.relative_to(ROOT)): digest(p) for p in set(files)}
    write(output / 'PLAN.json', dict(inputs=hashes, models={k:str(v.relative_to(ROOT)) for k,v in models.items()},
        ordered_checks=['known_failure_both_models', 'retention_best_safe_model', 'new_noise_best_safe_model'],
        maximum_physics_episodes=4, model_selection='safe success, then simulation time, then travel',
        retention_time_and_travel_factor=1.2, reference=reference,
        stop_expansion_when_no_safe_candidate=True, stop_expansion_on_retention_failure=True,
        original_fresh_four_now_development=True, known_failed_trace_not_a_positive_demonstration=True,
        new_noise_frozen_after_fit_before_physics=True, no_unseen_map_claim=True,
        no_teacher_or_inserted_actions=True, no_changed_masks_body_guards_or_margins=True,
        new_ppo_steps=0, new_optimizer_updates=0, full_regression_required_afterward=True))
    admissions = []
    for model in models.values():
        admission, _ = admit(model)
        admissions.append(admission)
        budget.tick()
    write(output / 'MODEL_ADMISSION.json', dict(completed=True, models=admissions))
    record_provenance(output, next(iter(models.values())) / 'portable', [Path(__file__)])
    (output / 'driver_snapshot.py').write_text(Path(__file__).read_text())
    physical = output / 'physical'
    physical.mkdir()
    report = dict(completed=False, rows=[], gates={}, new_ppo_steps=0, new_optimizer_updates=0,
        goal_achieved=False, deployment_approved=False, full_regression_still_required=True,
        actual106_not_claimed_fixed=True)

    def save():
        budget.tick()
        report.update(resource=budget.statistics(), at=datetime.now(timezone.utc).isoformat())
        write(output / 'RESULTS.json', report)

    def run(mode, case, group):
        write(output / 'STATUS.json', dict(status='pure_model_physical', model=mode, group=group,
            case_id=case['id'], finished_episodes=len(report['rows'])))
        print(json.dumps(dict(status='starting', model=mode, case_id=case['id'], group=group)), flush=True)
        row, trace = evaluate(case, models[mode] / 'portable', budget)
        path = physical / (mode + '_' + case['id'] + '.npz')
        np.savez_compressed(path, **trace)
        row.update(variant=mode, group=group, trace_path=str(path.relative_to(ROOT)),
            trace_sha256=digest(path), learned_model_only=True, no_teacher=True)
        report['rows'].append(row)
        save()
        print(json.dumps(row), flush=True)
        return row

    save()
    failed_rows = [run(mode, known_failure, 'known_failure') for mode in models]
    safe = [row for row in failed_rows if row['safe_success']]
    report['gates']['known_failure'] = dict(safe=len(safe), total=len(failed_rows), passed=bool(safe))
    if not safe:
        report['reason'] = 'Neither candidate safely recovered; stopped before retention and new noise'
    else:
        best = min(safe, key=lambda r: (r['seconds'], r['travel_m']))['variant']
        report['selected_pilot_candidate'] = best
        row = run(best, retention, 'retention')
        passed = bool(row['safe_success'] and row['seconds'] <= 1.2 * reference['seconds']
            and row['travel_m'] <= 1.2 * reference['travel_m'])
        report['gates']['retention'] = dict(passed=passed)
        if not passed:
            report['reason'] = 'Retention safety or efficiency failed; stopped before new noise'
        else:
            row = run(best, new_noise, 'new_noise')
            report['gates']['new_noise'] = dict(passed=bool(row['safe_success']))
            report['reason'] = ('Pilot gates passed; full regression still required' if row['safe_success']
                else 'New-noise gate failed; no deployment claim')
    assert len(report['rows']) <= 4
    for name, expected in hashes.items():
        assert digest(ROOT / name) == expected, name
    report['completed'] = True
    save()
    write(output / 'STATUS.json', dict(status='complete', finished_episodes=len(report['rows']),
        gates=report['gates'], reason=report['reason']))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    main(parser.parse_args())
