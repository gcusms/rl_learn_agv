"""Bounded sensor-gated turn probes from one exactly restored physical stall.

This is a teacher feasibility experiment, never a learned-policy result. It
waits for the specific live validation to finish, verifies physical branching,
then tests at most three bounded turns using only original legal actions.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import time

import numpy as np
import torch

from experiments.long_risk_actor import digest
from experiments.recovery_state_snapshot import RecoverySnapshot
from experiments.refit_phase_tail_pair import write
from experiments.verify_recovery_snapshot_replay import RecordingState, RecordedActions
from navrl.geometry import wrap
from neural_navigation.experiment_budget import ExperimentBudget
from neural_navigation.masked_actions import ACTIONS, COMMANDS, masks_from_observation
from neural_navigation.release_runtime import ReleaseRuntime
from neural_navigation.stress_env import StressNavigationEnv


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'reports/stop_feedback_20261015'
DEPENDENCY = 'rl-sensor-retention-validate-20261016.service'
EXPECTED_INVOCATION = 'dfdfe9f2beb1424c9aa5017b86869c10'


def unit_state():
    result = subprocess.run(['systemctl', '--user', 'show', DEPENDENCY,
        '--property=MainPID,ActiveState,InvocationID,ExecMainStatus'],
        text=True, capture_output=True, check=True)
    return dict(line.split('=', 1) for line in result.stdout.splitlines() if '=' in line)


class TurnWhenLegal:
    def __init__(self, delegate, core, target):
        self.delegate, self.core, self.target = delegate, core, target
        self.previous_yaw = float(core.odom[2])
        self.clockwise_turn = 0.
        self.phase = 'wait_for_legal_turn'
        self.records = []

    def predict(self, observation):
        yaw = float(self.core.odom[2])
        self.clockwise_turn -= float(wrap(yaw - self.previous_yaw))
        self.previous_yaw = yaw
        mask = masks_from_observation(np.asarray(observation)[:282])
        nominal = self.delegate.predict(observation)
        baseline = int(np.abs(ACTIONS - nominal).max(1).argmin())
        if self.phase == 'wait_for_legal_turn' and mask[5]:
            self.phase = 'turn'
        if self.phase == 'turn' and self.clockwise_turn >= self.target:
            self.phase = 'learned'
        choice = baseline if self.phase == 'learned' else 5 if self.phase == 'turn' and mask[5] else 0
        assert mask[choice]
        self.records.append(dict(sensor_time_s=float(self.core.elapsed), phase=self.phase,
            measured_clockwise_turn_rad=self.clockwise_turn, legal_turn=bool(mask[5]),
            choice=choice, frozen_model_choice=baseline, changed_from_model=choice != baseline))
        return ACTIONS[choice].copy()


def physical_step(env, runtime, budget, expected=None, index=None):
    if expected is not None:
        for key, actual in [('scans', env.scan), ('odometry', env.observed_odometry),
                ('measured_velocity', env.measured_velocity)]:
            np.testing.assert_allclose(actual, expected[key][index], rtol=0, atol=2e-8, equal_nan=True)
        np.testing.assert_allclose(env.truth, expected['labels'][index, 1:4], rtol=0, atol=2e-8)
    result = runtime.step(env.scan, env.observed_odometry, env.measured_velocity,
        env.elapsed, capture_observation=True)
    observation = np.asarray(result['policy_observation'], np.float32)
    mask = masks_from_observation(observation[:282])
    choice = int(np.abs(ACTIONS - result['proposed_action']).max(1).argmin())
    assert mask[choice]
    packet = dict(labels=np.r_[env.elapsed, env.truth, runtime.core.estimate,
        np.linalg.norm(env.truth[:2] - env.goal), 0., env._info('running')['actual_velocity']],
        observations=observation, masks=mask, choices=choice,
        commands=np.asarray(result['command']), scans=env.scan.copy(),
        odometry=env.observed_odometry.copy(), measured_velocity=env.measured_velocity.copy(),
        human_positions=np.asarray(env._humans()))
    if expected is not None:
        np.testing.assert_array_equal(observation, expected['observations'][index])
        np.testing.assert_array_equal(mask, expected['masks'][index])
        assert choice == expected['choices'][index]
        np.testing.assert_allclose(result['command'], expected['commands'][index], rtol=0, atol=2e-8)
    if result['stop_reason'] != 'estimated_goal_reached':
        np.testing.assert_allclose(result['command'], COMMANDS[choice], rtol=0, atol=2e-8)
        env.step(np.asarray(result['command']) / [.2, .5])
    budget.tick()
    return result, packet


def main(args):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    tasks = BASE / 'context_feedback_longfit_protocol_20261016/tasks.json'
    spec = json.loads(tasks.read_text())['actual106']
    source = BASE / 'context_broad_diagnostic_20261016/physical/hard3_w11_training_stall_t106.npz'
    bundle = ROOT / 'models/stop_feedback_20261015/sensor_retention_pair_s231016/retained500/portable'
    manifest = json.loads((bundle / 'manifest.json').read_text())
    files = [Path(__file__), tasks, source, bundle / 'manifest.json',
        ROOT / 'training/experiments/recovery_state_snapshot.py',
        ROOT / 'training/experiments/verify_recovery_snapshot_replay.py',
        BASE / 'recovery_snapshot_replay_20261016/PLAN.json']
    files += [bundle / name for name in manifest['files']]
    for folder in ['neural_navigation', 'navrl']:
        files += list((ROOT / 'training' / folder).rglob('*.py'))
    files += [p for p in (ROOT / 'training' / spec['case']['scene']).iterdir() if p.is_file()]
    hashes = {str(p.relative_to(ROOT)): digest(p) for p in set(files)}
    write(output / 'PLAN.json', dict(inputs=hashes, dependency_unit=DEPENDENCY,
        expected_invocation_id=EXPECTED_INVOCATION, no_parallel_physical_job=True,
        prefix_frames=530, maximum_parity_steps=30, maximum_teacher_trials=3,
        clockwise_turn_targets_rad=[.1, .3, .6], maximum_new_physical_steps=1070,
        first_legal_turn_is_sensor_mask_triggered_not_episode_time=True,
        original_actions_4cm_body_and_guard_unchanged=True,
        source_is_reused_training_failure_not_new_full_mission=True,
        teacher_probes_not_model_success=True, no_positive_labels_automatically_admitted=True,
        no_optimizer_updates=True, new_ppo_steps=0, goal_achieved=False, deployment_approved=False))
    while True:
        state = unit_state()
        if state['InvocationID'] not in ('', EXPECTED_INVOCATION):
            raise RuntimeError('Validation dependency was replaced or restarted')
        if state['ActiveState'] in ('inactive', 'failed') and state['MainPID'] == '0':
            validation = json.loads((BASE / 'sensor_retention_validation_20261016/RESULTS.json').read_text())
            if state['ActiveState'] != 'inactive' or state['ExecMainStatus'] != '0' or not validation['completed']:
                raise RuntimeError('Validation did not finish normally; no physical probe')
            break
        write(output / 'STATUS.json', dict(status='waiting_for_exact_validation',
            physical_simulation_started=False, dependency=state))
        time.sleep(5.)
    verified = BASE / 'recovery_snapshot_replay_20261016/RESULTS.json'
    verification = json.loads(verified.read_text())
    assert verification['completed'] and verification['full_sensor_runtime_restoration_verified']
    for name, expected in hashes.items():
        assert digest(ROOT / name) == expected, name
    write(output / 'DEPENDENCIES.json', dict(validation_completed=True,
        validation_sha256=digest(BASE / 'sensor_retention_validation_20261016/RESULTS.json'),
        snapshot_replay_sha256=digest(verified)))
    budget = ExperimentBudget(.35, memory_mib=2800, directory=output)
    with np.load(source, allow_pickle=False) as archive:
        reference = {key: archive[key].copy() for key in archive.files}
    assert reference['observations'].shape == (700, 625)
    case = spec['case']
    env = StressNavigationEnv(case=case, components=bundle, scene_dir=ROOT / 'training' / case['scene'],
        route_mode='shortcut', max_steps=case['max_steps'])
    env.reset(seed=case['seed'], options=dict(start=case['start'], goal=case['goal'][:2]))
    runtime = ReleaseRuntime(bundle, ROOT / 'training' / case['scene'] / 'decision_map.yaml',
        case['start'], case['goal'], 'shortcut')
    runtime._branch_delegate = runtime.policy.policy
    runtime.policy.policy = RecordedActions(RecordingState(reference['choices']))
    report = dict(completed=False, rows=[], new_physical_steps=0,
        goal_achieved=False, deployment_approved=False, new_ppo_steps=0)
    try:
        for index in range(530):
            physical_step(env, runtime, budget, reference, index)
            report['new_physical_steps'] += 1
            if index % 50 == 0:
                write(output / 'STATUS.json', dict(status='exact_physical_prefix', frames=index+1))
        snapshot = RecoverySnapshot(env, runtime)
        original = []
        for index in range(530, 540):
            _, packet = physical_step(env, runtime, budget, reference, index)
            original.append(packet); report['new_physical_steps'] += 1
        restored_rows = []
        for repeat in range(2):
            restored_env, restored_runtime = snapshot.restore()
            try:
                for offset, index in enumerate(range(530, 540)):
                    _, packet = physical_step(restored_env, restored_runtime, budget, reference, index)
                    for key, value in packet.items():
                        np.testing.assert_array_equal(value, original[offset][key])
                    report['new_physical_steps'] += 1
                restored_rows.append(dict(repeat=repeat, exact_complete_physical_sensor_and_runtime_frames=10))
            finally:
                restored_env.close()
        report['physical_snapshot_parity'] = restored_rows
        write(output / 'RESULTS.json', report)
        for target in [.1, .3, .6]:
            trial_env, trial_runtime = snapshot.restore()
            name = 'turn_when_legal_' + str(target).replace('.', '_')
            teacher = TurnWhenLegal(trial_runtime._branch_delegate, trial_runtime.core, target)
            trial_runtime.policy.policy = teacher
            trial_env.minimum_gap = trial_env._minimum_gap()
            packets = []
            arrived = false_arrival = False
            try:
                for _ in range(case['max_steps'] - 530):
                    write(output / 'STATUS.json', dict(status='teacher_probe', trial=name,
                        physical_simulation_started=True, simulation_seconds=float(trial_env.elapsed),
                        phase=teacher.phase, clockwise_turn_rad=teacher.clockwise_turn))
                    result, packet = physical_step(trial_env, trial_runtime, budget)
                    packets.append(packet)
                    if result['stop_reason'] == 'estimated_goal_reached':
                        arrived = bool(packet['labels'][7] < .2)
                        false_arrival = not arrived
                        break
                    report['new_physical_steps'] += 1
                    if trial_env.collision_events or trial_env.hard_guard_activations:
                        break
                row = dict(trial=name, safe_success=bool(arrived and not false_arrival
                    and not trial_env.collision_events and not trial_env.hard_guard_activations
                    and trial_env.minimum_gap >= .04 - 1e-4),
                    arrived=arrived, false_arrival=false_arrival,
                    collisions=int(trial_env.collision_events), physical_guards=int(trial_env.hard_guard_activations),
                    post_release_seconds=float(trial_env.elapsed - 106.),
                    final_xy_error_m=float(np.linalg.norm(trial_env.truth[:2] - case['goal'])),
                    minimum_post_release_dynamic_gap_m=float(trial_env.minimum_gap),
                    target_clockwise_turn_rad=target, measured_clockwise_turn_rad=teacher.clockwise_turn,
                    teacher_phase=teacher.phase, scripted_frames=sum(r['phase'] != 'learned' for r in teacher.records),
                    changed_from_frozen_model_frames=sum(r['changed_from_model'] for r in teacher.records),
                    original_permissions_unchanged=True, teacher_probe_not_model_success=True,
                    no_automatic_source_admission=True)
                path = output / (name + '.npz')
                np.savez_compressed(path, **{key: np.asarray([p[key] for p in packets]) for key in packets[0]})
                row.update(trace_path=str(path.relative_to(ROOT)), trace_sha256=digest(path))
                write(output / (name + '_decisions.json'), teacher.records)
                report['rows'].append(row)
                report['resource'] = budget.statistics()
                write(output / 'RESULTS.json', report)
                if row['safe_success'] or row['collisions'] or row['false_arrival'] or row['physical_guards']:
                    break
            finally:
                trial_env.close()
        snapshot.verify_shared_models()
        for name, expected in hashes.items():
            assert digest(ROOT / name) == expected, name
        report.update(completed=True, at=datetime.now(timezone.utc).isoformat(),
            complete_safe_teacher_trials=sum(r['safe_success'] for r in report['rows']),
            input_hashes_verified=len(hashes), resource=budget.statistics())
        assert report['new_physical_steps'] <= 1070
        write(output / 'RESULTS.json', report)
        write(output / 'STATUS.json', dict(status='complete', goal_achieved=False))
    finally:
        env.close()


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
