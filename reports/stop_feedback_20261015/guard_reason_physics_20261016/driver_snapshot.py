"""One isolated recovery: test the observed causal effect of current reasons.

Source-fit rejection remains in force. Offline170-frame changes warrant one
physics diagnosis, never deployment, failed-label admission or forced motion.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess

import numpy as np
import torch

from experiments.diagnose_context_release_physics import write
from experiments.guard_reason_release_actor import GuardReasonReleaseActor, export
from experiments.long_risk_actor import digest
from experiments.release_recovery import evaluate_recovery
from neural_navigation.experiment_budget import ExperimentBudget
from neural_navigation.masked_actions import ACTIONS, masks_from_observation


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'reports/stop_feedback_20261015'


def main(args):
    output, model = args.output.resolve(), args.model.resolve()
    previous_service = subprocess.check_output(['systemctl', '--user', 'show',
        'rl-context-broad-diagnostic-20261016.service', '-p', 'MainPID', '-p', 'ActiveState'], text=True)
    assert 'MainPID=0' in previous_service and 'ActiveState=inactive' in previous_service
    prior_report = BASE / 'context_broad_diagnostic_20261016/RESULTS.json'
    prior = json.loads(prior_report.read_text())
    assert prior['completed']
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    budget = ExperimentBudget(.35, memory_mib=3072, directory=output)
    fit = json.loads((model / 'RESULTS.json').read_text())
    assert fit['completed'] and not fit['exported_for_physical_validation']
    assert fit['correct_source_labels'] == 3250 and fit['total_source_labels'] == 3252
    assert fit['original2848_all_retained'] and fit['safe_stop_labels_all_retained']
    assert digest(model / 'checkpoint.pt') == fit['checkpoint_sha256']
    audit_path = BASE / 'guard_reason_input_audit_20261016/RESULTS.json'
    audit = json.loads(audit_path.read_text())
    assert audit['completed'] and audit['rows'][0]['frames'] == 170
    assert audit['rows'][0]['candidate_legal_wait_frames'] == 5
    assert audit['rows'][0]['head_only_reason_ablation_changed_decisions'] == 163
    tasks_path = BASE / 'context_feedback_longfit_protocol_20261016/tasks.json'
    specification = json.loads(tasks_path.read_text())['actual106']
    dataset = BASE / 'context_feedback_sources_20261016/demonstrations.npz'
    assert digest(dataset) == fit['source_sha256']
    risk = ROOT / 'models/stop_feedback_20261015/long_risk_bc1000_s231016/portable'
    risk_manifest = json.loads((risk / 'manifest.json').read_text())
    files = [Path(__file__), prior_report, tasks_path, audit_path, dataset,
        model / 'RESULTS.json', model / 'checkpoint.pt', model / 'parent_actor.pt',
        ROOT / specification['trace'], risk / 'manifest.json']
    files += [risk / k for k in risk_manifest['files']]
    for folder in ['neural_navigation', 'navrl', 'experiments']:
        files += list((ROOT / 'training' / folder).rglob('*.py'))
    files += [ROOT / 'training/neural_navigation/static_tracking_kernel.cpp']
    files += [p for p in (ROOT / 'training' / specification['case']['scene']).iterdir() if p.is_file()]
    hashes = {str(p.relative_to(ROOT)): digest(p) for p in set(files)}
    write(output / 'PLAN.json', dict(at=datetime.now(timezone.utc).isoformat(), inputs=hashes,
        checkpoint_sha256=fit['checkpoint_sha256'], specification=specification,
        maximum_new_physical_episodes=1, diagnostic_only=True,
        reason='163offline_decisions_depend_on_new_current_reasons; complete_recovery_unproven',
        source_fit_rejection_unchanged=True, not_a_deployment_or_training_admission=True,
        no_failed_labels_added=True, no_forced_motion_or_runtime_teacher=True,
        actions_masks_body4cm_margin_blind_memory_and_deadlines_unchanged=True,
        common_prefix_initialization_only=True, no_second_physics_process=True,
        new_ppo_steps=0, new_optimizer_updates=0, deployment_approved=False))
    (output / 'driver_snapshot.py').write_text(Path(__file__).read_text())
    actor = GuardReasonReleaseActor(torch.jit.load(str(model / 'parent_actor.pt'), map_location='cpu')).eval()
    actor.load_state_dict(torch.load(model / 'checkpoint.pt', map_location='cpu', weights_only=False)['actor'])
    with np.load(dataset, allow_pickle=False) as z:
        observations, masks, labels = [z[k].copy() for k in ['observations', 'masks', 'choices']]
    with torch.no_grad():
        commands = actor(torch.from_numpy(observations)).numpy()
    choice = np.abs(commands[:, None] - ACTIONS[None]).max(2).argmin(1)
    np.testing.assert_array_equal(np.flatnonzero(choice != labels), [3182, 3245])
    np.testing.assert_array_equal(choice[:2848], labels[:2848])
    assert (choice[labels == 0] == 0).all() and masks[np.arange(len(choice)), choice].all()
    np.testing.assert_array_equal(masks_from_observation(observations[:, :282]), masks)
    invalid, near, stopped = (observations[:16].copy() for _ in range(3))
    invalid[:, 435] = 0.; near[:, 218:220] = 0.; near[:, 302] = 1.
    stopped[:, 235:244] = stopped[:, 267:269] = 0.
    checks = np.r_[observations, invalid, near, stopped]
    bundle = output / 'diagnostic_bundle'
    export(actor, risk, bundle, checks)
    manifest = json.loads((bundle / 'manifest.json').read_text())
    manifest.update(diagnostic_only=True, rejected_source_fit_gate=True,
        source_checkpoint_sha256=fit['checkpoint_sha256'], deployment_approved=False)
    write(bundle / 'manifest.json', manifest)
    portable = torch.jit.load(str(bundle / 'actor.pt'), map_location='cpu').eval()
    with torch.no_grad():
        np.testing.assert_array_equal(portable(torch.from_numpy(checks)).numpy(), actor(torch.from_numpy(checks)).numpy())
    write(output / 'MODEL_CHECKS.json', dict(completed=True, full_source_labels=3252,
        correct_source_labels=3250, rejected_fit_gate_unchanged=True, original2848_exact=True,
        all_stop_labels_exact=True, all_actions_currently_legal=True, eager_portable_exact=True,
        invalid_stop_only_and_terminal_export_checked=True, diagnostic_only=True))
    del actor, portable, observations, masks, labels, commands, choice, checks
    (output / 'physical').mkdir()
    write(output / 'STATUS.json', dict(status='single_isolated_physics_diagnosis',
        case_id=specification['id'], completed_episodes=0))
    print(json.dumps(dict(starting=specification['id'])), flush=True)
    row, trace = evaluate_recovery(specification, bundle, budget)
    path = output / 'physical' / (specification['id'] + '.npz')
    np.savez_compressed(path, **trace)
    row.update(trace_path=str(path.relative_to(ROOT)), trace_sha256=digest(path),
        variant=model.name, pure_learned_model=True, diagnostic_only=True)
    for name, expected in hashes.items():
        assert digest(ROOT / name) == expected, name
    budget.tick()
    write(output / 'RESULTS.json', dict(completed=True, at=datetime.now(timezone.utc).isoformat(),
        rows=[row], baseline=prior['groups']['hard3']['rows'][0], new_physical_episodes=1,
        source_fit_rejection_unchanged=True, diagnostic_only=True, no_automatic_promotion=True,
        new_ppo_steps=0, new_optimizer_updates=0, goal_achieved=False,
        deployment_approved=False, inputs_verified=len(hashes), resource=budget.statistics()))
    write(output / 'STATUS.json', dict(status='complete', completed_episodes=1,
        safe_physical_diagnostics=int(row['safe_success'])))
    print(json.dumps(row), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ['model', 'output']:
        parser.add_argument('--' + key, type=Path, required=True)
    args = parser.parse_args()
    try:
        main(args)
    except Exception as error:
        if args.output.is_dir():
            write(args.output / 'FAILURE.json', dict(error=repr(error), automatically_restarted=False))
        raise
