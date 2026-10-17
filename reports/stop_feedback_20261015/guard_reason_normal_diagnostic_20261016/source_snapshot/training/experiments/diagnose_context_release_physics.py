"""Isolated physics diagnosis of a rejected demonstration-fit candidate.

The original all-label gate remains failed. This does not approve training,
deployment, or erase a reference-action mismatch. Three complete physical
continuations distinguish a behavior failure from a nonunique imitation label.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import torch

from experiments.context_release_actor import ContextReleaseActor, export
from experiments.long_risk_actor import digest
from experiments.paired_navigation_gate import compare
from experiments.release_recovery import evaluate_recovery
from neural_navigation.experiment_budget import ExperimentBudget
from neural_navigation.masked_actions import ACTIONS, masks_from_observation


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'reports/stop_feedback_20261015'


def write(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def main(args):
    output, model, protocol = [getattr(args, k).resolve() for k in ['output', 'model', 'protocol']]
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    budget = ExperimentBudget(.35, memory_mib=3072, directory=output)
    fit = json.loads((model / 'RESULTS.json').read_text())
    assert fit['completed'] and not fit['exported_for_physical_validation']
    assert fit['total_source_labels'] == 3252 and fit['correct_source_labels'] == 3251
    assert digest(model / 'checkpoint.pt') == fit['checkpoint_sha256']
    source = json.loads((protocol / 'tasks.json').read_text())
    specs = [*source['actual95_actual90'], source['delayed_wait']]
    assert len(specs) == 3
    risk = ROOT / 'models/stop_feedback_20261015/long_risk_bc1000_s231016/portable'
    files = [Path(__file__), protocol / 'tasks.json', model / 'RESULTS.json', model / 'checkpoint.pt',
        model / 'parent_actor.pt', model / 'demonstrations.npz',
        BASE / 'context_longfit_audit_20261016/RESULTS.json',
        BASE / 'recovery_input_separation_20261016/RESULTS.json']
    manifest = json.loads((risk / 'manifest.json').read_text())
    files += [risk / 'manifest.json', *[risk / k for k in manifest['files']]]
    files += [ROOT / s['trace'] for s in specs]
    for folder in ['neural_navigation', 'navrl', 'experiments']:
        files += list((ROOT / 'training' / folder).rglob('*.py'))
    files += [ROOT / 'training/neural_navigation/static_tracking_kernel.cpp']
    for spec in specs:
        files += [p for p in (ROOT / 'training' / spec['case']['scene']).iterdir() if p.is_file()]
    hashes = {str(p.relative_to(ROOT)): digest(p) for p in set(files)}
    write(output / 'PLAN.json', dict(at=datetime.now(timezone.utc).isoformat(), inputs=hashes,
        model=str(model.relative_to(ROOT)), checkpoint_sha256=fit['checkpoint_sha256'],
        specifications=specs, maximum_physical_episodes=3, diagnostic_only=True,
        previous_fit_gate_remains_failed=True, no_training_or_deployment_admission=True,
        all_source_labels_retained=True, no_failed_trace_labels=True,
        unchanged_actions_masks_body_margin_blind_memory_and_deadlines=True,
        prefix_initialization_only_no_teacher_after_release=True,
        new_ppo_steps=0, new_optimizer_updates=0, deployment_approved=False))
    (output / 'driver_snapshot.py').write_text(Path(__file__).read_text())
    actor = ContextReleaseActor(torch.jit.load(str(model / 'parent_actor.pt'), map_location='cpu')).eval()
    state = torch.load(model / 'checkpoint.pt', map_location='cpu', weights_only=False)
    actor.load_state_dict(state['actor'])
    with np.load(model / 'demonstrations.npz', allow_pickle=False) as archive:
        data = {k: archive[k].copy() for k in archive.files}
    x = torch.from_numpy(data['observations'])
    with torch.no_grad():
        commands = actor(x).numpy()
    choice = np.abs(commands[:, None] - ACTIONS[None]).max(2).argmin(1)
    wrong = np.flatnonzero(choice != data['choices'])
    np.testing.assert_array_equal(wrong, [3182])
    np.testing.assert_array_equal(choice[:2848], data['choices'][:2848])
    np.testing.assert_array_equal(masks_from_observation(data['observations'][:, :282]), data['masks'])
    assert data['masks'][np.arange(len(choice)), choice].all()
    assert (choice[data['choices'] == 0] == 0).all()
    bundle = output / 'diagnostic_bundle'
    export(actor, risk, bundle, data['observations'])
    manifest = json.loads((bundle / 'manifest.json').read_text())
    manifest.update(diagnostic_only=True, rejected_source_fit_gate=True,
        source_checkpoint_sha256=fit['checkpoint_sha256'], deployment_approved=False)
    write(bundle / 'manifest.json', manifest)
    portable = torch.jit.load(str(bundle / 'actor.pt'), map_location='cpu').eval()
    with torch.no_grad():
        np.testing.assert_array_equal(portable(x).numpy(), commands)
    write(output / 'MODEL_CHECKS.json', dict(completed=True, labels=3252, correct=3251,
        original2848_exact=True, all_stop_labels_exact=True, all_proposed_actions_legal=True,
        remaining_reference_mismatch=wrong.tolist(), eager_portable_exact=True,
        rejected_fit_gate_unchanged=True, diagnostic_only=True))
    del actor, portable, state, x, data
    physical = output / 'physical'; physical.mkdir()
    report = dict(completed=False, rows=[], comparisons={}, diagnostic_only=True,
        previous_fit_gate_remains_failed=True, no_automatic_promotion=True,
        new_ppo_steps=0, new_optimizer_updates=0, deployment_approved=False, goal_achieved=False)

    def save():
        budget.tick()
        report.update(at=datetime.now(timezone.utc).isoformat(), resource=budget.statistics())
        write(output / 'RESULTS.json', report)

    save()
    for spec in specs:
        write(output / 'STATUS.json', dict(status='isolated_physics_diagnosis',
            case_id=spec['id'], completed_episodes=len(report['rows'])))
        print(json.dumps(dict(starting=spec['id'])), flush=True)
        row, trace = evaluate_recovery(spec, bundle, budget)
        path = physical / (spec['id'] + '.npz')
        np.savez_compressed(path, **trace)
        row.update(trace_path=str(path.relative_to(ROOT)), trace_sha256=digest(path),
            variant=model.name, pure_learned_model=True, diagnostic_only=True)
        report['rows'].append(row)
        save()
        print(json.dumps(row), flush=True)
    report['comparisons']['actual95_actual90'] = compare(
        source['reference']['actual95_actual90'], report['rows'][:2])
    report['comparisons']['delayed_wait'] = compare(
        source['reference']['delayed_wait'], report['rows'][2:])
    report['safe_physical_diagnostics'] = sum(r['safe_success'] for r in report['rows'])
    for name, expected in hashes.items():
        assert digest(ROOT / name) == expected, name
    report.update(completed=True, frozen_inputs_verified=len(hashes))
    save()
    write(output / 'COMPLETION_CHECKS.json', dict(completed=True, inputs_verified=len(hashes),
        previous_fit_gate_remains_failed=True, diagnostic_only=True))
    write(output / 'STATUS.json', dict(status='complete', completed_episodes=3,
        safe_physical_diagnostics=report['safe_physical_diagnostics']))


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
