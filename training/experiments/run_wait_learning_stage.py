"""Run a bounded validation or PPO stage after its exact predecessor finishes.

Waiting loads no model. A failed fit, regression or safety gate records that
the next stage did not run. It never restarts a predecessor or promotes models.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import time
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[2]


def write(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')
    temporary.replace(path)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def unit_state(unit):
    result = subprocess.run(['systemctl', '--user', 'show', unit,
        '--property=ActiveState,MainPID,InvocationID,ExecMainStatus'],
        text=True, capture_output=True, check=True)
    return dict(row.split('=', 1) for row in result.stdout.splitlines() if '=' in row)


def main(args):
    control = args.control.resolve()
    control.mkdir(parents=True, exist_ok=False)
    parent_path = args.parent_result.resolve()
    files = [Path(__file__), ROOT / 'training/experiments/validate_normal_retention_candidate.py',
        ROOT / 'training/experiments/train_native625_recovery.py',
        ROOT / 'training/experiments/native625_onpolicy.py']
    hashes = {str(path.relative_to(ROOT)): digest(path) for path in files}
    write(control / 'PLAN.json', dict(stage=args.stage, parent_result=str(parent_path),
        dependency_unit=args.dependency_unit, expected_invocation=args.dependency_invocation,
        inputs=hashes, no_restart_or_automatic_deployment=True,
        normal18_retention_required_before_PPO=True, full_source_fit_required_before_BC_validation=True,
        ppo_release_frames=[450, 480, 530, 572], ppo_maximum_decisions=6144,
        ppo_maximum_initialization_steps=2032, ppo_maximum_episodes=32))
    while True:
        state = unit_state(args.dependency_unit)
        if state['InvocationID'] not in ('', args.dependency_invocation):
            raise RuntimeError('Predecessor was replaced or restarted')
        if state['ActiveState'] in ('failed', 'inactive') and state['MainPID'] == '0':
            break
        write(control / 'STATUS.json', dict(status='waiting_for_exact_predecessor',
            dependency=state, physical_simulation_started=False, optimizer_started=False,
            new_ppo_steps=0))
        time.sleep(5.)
    for name, expected in hashes.items():
        assert digest(ROOT / name) == expected, name
    parent = json.loads(parent_path.read_text()) if parent_path.exists() else {}
    ready = bool(state['ActiveState'] == 'inactive' and state['ExecMainStatus'] == '0'
        and parent.get('completed', False))
    if args.stage == 'validate':
        ready = ready and bool(parent.get('selected_for_bounded_physical_validation'))
    else:
        ready = ready and bool(parent.get('normal18_pair', {}).get('all_regression_checks_passed'))
        rows = [row for group in parent.get('groups', {}).values() for row in group.get('rows', [])]
        ready = ready and bool(rows) and not any(row.get('collisions', 0) or row.get('physical_guards', 0)
            or row.get('false_arrival', False) for row in rows)
    if not ready:
        write(control / 'RESULTS.json', dict(completed=True, requested_stage=args.stage,
            stage_started=False, predecessor_gate_passed=False, new_ppo_steps=0,
            reason='Pinned predecessor failed or did not meet the required source/retention/safety gate',
            parent_result_sha256=digest(parent_path) if parent_path.exists() else None,
            goal_achieved=False, deployment_approved=False))
        write(control / 'STATUS.json', dict(status='stage_not_started_predecessor_gate_failed'))
        return
    write(control / 'DEPENDENCIES.json', dict(parent_result_sha256=digest(parent_path),
        predecessor_gate_passed=True))
    write(control / 'STATUS.json', dict(status='executing_'+args.stage))
    if args.stage == 'validate':
        from experiments.validate_normal_retention_candidate import main as validate
        validate(SimpleNamespace(fits=args.fits, output=args.output,
            priority_normal_id=['independent_dynamic_000', 'train_failure_w11_e7'],
            priority_sensor_id=['sensor_release_frozen_noise_05_0', 'sensor_release_frozen_noise_05_1']))
    else:
        from experiments.train_native625_recovery import main as train
        train(SimpleNamespace(report=args.output, models=args.models,
            model=ROOT / parent['selected_model'], anchor_dataset=args.anchor_dataset,
            parent_validation=parent_path, dependency_unit=args.dependency_unit,
            dependency_invocation=args.dependency_invocation,
            release_frames=[450, 480, 530, 572], episodes_per_stage=8))
    stage_path = args.output.resolve() / 'RESULTS.json'
    stage = json.loads(stage_path.read_text())
    write(control / 'RESULTS.json', dict(completed=True, requested_stage=args.stage,
        stage_started=True, stage_completed=bool(stage['completed']), stage_result=str(stage_path),
        stage_result_sha256=digest(stage_path), new_ppo_steps=stage.get('new_ppo_steps', 0),
        at=datetime.now(timezone.utc).isoformat(), goal_achieved=False, deployment_approved=False))
    write(control / 'STATUS.json', dict(status='stage_finished'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['validate', 'ppo'], required=True)
    parser.add_argument('--dependency-unit', required=True)
    parser.add_argument('--dependency-invocation', required=True)
    parser.add_argument('--parent-result', type=Path, required=True)
    parser.add_argument('--control', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--fits', type=Path)
    parser.add_argument('--models', type=Path)
    parser.add_argument('--anchor-dataset', type=Path)
    args = parser.parse_args()
    if args.stage == 'validate' and args.fits is None:
        parser.error('Validation requires --fits')
    if args.stage == 'ppo' and (args.models is None or args.anchor_dataset is None):
        parser.error('PPO requires --models and --anchor-dataset')
    try:
        main(args)
    except Exception as error:
        if args.control.is_dir():
            write(args.control / 'FAILURE.json', dict(error=repr(error), automatically_restarted=False))
        raise
