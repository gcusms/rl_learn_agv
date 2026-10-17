"""Frozen earlier-release, paired-noise and directional-docking checks."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import torch

from experiments.evaluate_docking_navigation import evaluate
from experiments.evaluate_release_phase_pilot import admit
from experiments.experiment_provenance import record_provenance
from experiments.long_risk_actor import digest
from neural_navigation.experiment_budget import ExperimentBudget


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'reports/stop_feedback_20261015'


def write(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def main(args):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    budget = ExperimentBudget(.35, memory_mib=3072, directory=output)
    protocol = BASE / 'sensor_release_protocol_20261016'
    protocol_path = protocol / 'PLAN.json'
    plan = json.loads(protocol_path.read_text())
    cases = json.loads((protocol / 'cases.json').read_text())
    assert plan['new_noise_frozen_before_fit'] and plan['maximum_validation_physics_episodes'] == 9
    for name, expected in plan['inputs'].items():
        assert digest(ROOT / name) == expected, name
    models = {key:ROOT / plan[key + '_model'] for key in ['baseline', 'candidate']}
    fit = json.loads((models['candidate'] / 'RESULTS.json').read_text())
    assert datetime.fromisoformat(plan['at']) < datetime.fromisoformat(fit['at'])
    assert fit['new_ppo_steps'] == 0 and fit['exported_for_physical_validation']
    files = [Path(__file__), protocol_path, protocol / 'cases.json']
    files += [ROOT / name for name in plan['inputs']]
    for model in models.values():
        manifest = json.loads((model / 'portable/manifest.json').read_text())
        files += [model / name for name in ['RESULTS.json', 'checkpoint.pt', 'parent_actor.pt',
            'demonstrations.npz', 'portable/manifest.json']]
        files += [model / 'portable' / name for name in manifest['files']]
    for folder in ['neural_navigation', 'navrl', 'experiments']:
        files += list((ROOT / 'training' / folder).rglob('*.py'))
    flat_cases = [cases['known'], cases['retention'], cases['previous_noise'], *cases['new_noise'], *cases['heading']]
    for case in flat_cases:
        files += [p for p in (ROOT / 'training' / case['scene']).iterdir() if p.is_file()]
    hashes = {str(p.relative_to(ROOT)):digest(p) for p in set(files)}
    write(output / 'PLAN.json', dict(protocol=plan, inputs=hashes, maximum_new_physics_episodes=9,
        pure_model_no_teacher=True, original_body_guards_masks_and_deadlines=True,
        full_regression_still_required=True, deployment_approved=False))
    admission = []
    for model in models.values():
        record, _ = admit(model)
        admission.append(record)
        budget.tick()
    write(output / 'MODEL_ADMISSION.json', dict(completed=True, models=admission))
    record_provenance(output, models['candidate'] / 'portable', [Path(__file__)])
    physical = output / 'physical'
    physical.mkdir()
    report = dict(completed=False, rows=[], gates={}, new_ppo_steps=0, new_optimizer_updates=0,
        actual106_not_claimed_fixed=True, full_regression_still_required=True,
        goal_achieved=False, deployment_approved=False)

    def save():
        budget.tick()
        report.update(at=datetime.now(timezone.utc).isoformat(), resource=budget.statistics())
        write(output / 'RESULTS.json', report)

    def finish(reason):
        for name, expected in hashes.items():
            assert digest(ROOT / name) == expected, name
        assert len(report['rows']) <= 9
        report.update(completed=True, reason=reason)
        save()
        write(output / 'STATUS.json', dict(status='complete', reason=reason,
            completed_episodes=len(report['rows']), gates=report['gates']))
        (output / 'driver_snapshot.py').write_text(Path(__file__).read_text())

    def run(variant, case, group):
        write(output / 'STATUS.json', dict(status='pure_model_physical', group=group,
            case_id=case['id'], variant=variant, finished_episodes=len(report['rows'])))
        print(json.dumps(dict(status='starting', group=group, case_id=case['id'], variant=variant)), flush=True)
        row, trace = evaluate(case, models[variant] / 'portable', budget)
        path = physical / (variant + '_' + case['id'] + '.npz')
        np.savez_compressed(path, **trace)
        row.update(variant=variant, group=group, trace_path=str(path.relative_to(ROOT)),
            trace_sha256=digest(path), learned_model_only=True, no_teacher=True)
        report['rows'].append(row)
        save()
        print(json.dumps(row), flush=True)
        return row

    def efficient(row, reference, factor):
        return bool(row['safe_success'] and row['seconds'] <= reference['seconds']*factor+1e-8
            and row['travel_m'] <= reference['travel_m']*factor+1e-8)

    save()
    row = run('candidate', cases['known'], 'known_earlier_release')
    passed = bool(row['safe_success'] and row['seconds'] <= plan['known_seconds_max']
        and row['voluntary_wait_frames'] <= plan['known_wait_frames_max'])
    report['gates']['known_earlier_release'] = dict(passed=passed,
        maximum_seconds=plan['known_seconds_max'], maximum_wait_frames=plan['known_wait_frames_max'])
    if not passed:
        finish('Known earlier-release safety or improvement gate failed; no expansion')
        return
    for case in [cases['retention'], cases['previous_noise']]:
        row = run('candidate', case, 'retention')
        passed = efficient(row, plan['reference'][case['id']], plan['retention_time_and_travel_factor'])
        report['gates']['retention_' + case['id']] = dict(passed=passed)
        if not passed:
            finish('Retained safety or efficiency gate failed; no expansion')
            return
    for case in cases['new_noise']:
        baseline = run('baseline', case, 'paired_new_noise')
        candidate = run('candidate', case, 'paired_new_noise')
        passed = bool(candidate['safe_success'] and (not baseline['safe_success']
            or efficient(candidate, baseline, plan['paired_new_noise_time_and_travel_factor'])))
        report['gates']['paired_' + case['id']] = dict(passed=passed,
            baseline_safe=baseline['safe_success'], candidate_safe=candidate['safe_success'])
        if not passed:
            finish('Paired new-noise safety or efficiency gate failed; no expansion')
            return
    heading_reference = {row['case_id']:row for row in plan['heading_reference']}
    for case in cases['heading']:
        row = run('candidate', case, 'heading')
        passed = efficient(row, heading_reference[case['id']], plan['heading_time_and_travel_factor'])
        report['gates']['heading_' + case['id']] = dict(passed=passed)
        if not passed:
            finish('Directional docking retention failed; no expansion')
            return
    finish('All nine bounded checks passed; full normal/dynamic/prefix regression still required')


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
