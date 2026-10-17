"""Bounded legal, measured-yaw multi-action probes of the actual106s state.

Earlier probes used short fixed turns or fixed action windows. These tests
interleave the original pure clockwise turn and tiny clockwise creep under
the current sensor mask, then return to the frozen model. Expert trajectories
remain diagnostics; neither success nor failure is called a learned result.
"""
from datetime import datetime
import gc
import hashlib
import json
from pathlib import Path
import subprocess
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / 'reports/stop_feedback_20261015'
HERE = Path(__file__).resolve().parent
OUT = BASE / 'late106_adaptive_legal_20261017'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(name, value):
    path = OUT/name
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')
    temp.replace(path)


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    for unit in ['rl-memory-from-start-20261017.service', 'rl-memory-joint-retention-20261017.service',
                 'rl-paired-memory-tradeoff-20261017.service']:
        result = subprocess.run(['systemctl', '--user', 'show', unit,
            '--property=MainPID,ActiveState,ExecMainStatus'], capture_output=True, text=True, check=True)
        state = dict(line.split('=', 1) for line in result.stdout.splitlines() if '=' in line)
        assert state['MainPID'] == '0' and state['ActiveState'] == 'inactive' and state['ExecMainStatus'] == '0', (unit, state)
    prerequisite_path = BASE/'memory_from_start_20261017/RESULTS.json'
    prerequisite = json.loads(prerequisite_path.read_text())
    assert prerequisite['completed'] and prerequisite['selected_frozen_actor'] == 'bc4500'
    assert not prerequisite.get('bounded_from_start_gate_passed')
    tradeoff_path = BASE/'paired_memory_tradeoff_20261017/RESULTS.json'
    assert json.loads(tradeoff_path.read_text())['completed']
    bundle = ROOT/'models/stop_feedback_20261015/memory_evidence_s231017/bc4500/portable'
    tasks_path = BASE/'context_feedback_longfit_protocol_20261016/tasks.json'
    spec = json.loads(tasks_path.read_text())['actual106']
    source = ROOT/spec['trace']
    manifest = json.loads((bundle/'manifest.json').read_text())
    paths = [Path(__file__), HERE/'legal_turn_policy.py', HERE/'test_legal_turn_policy.py',
        prerequisite_path, tradeoff_path, tasks_path, source, bundle/'manifest.json']
    paths += [bundle/name for name in manifest['files']]
    for folder in ['experiments', 'neural_navigation', 'navrl']:
        paths += list((ROOT/'training'/folder).rglob('*.py'))
    paths += [p for p in (ROOT/'training'/spec['case']['scene']).iterdir() if p.is_file()]
    hashes = {str(p.relative_to(ROOT)): digest(p) for p in set(paths)}
    write('PLAN.json', dict(at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(), inputs=hashes,
        prefix_frames=530, snapshot_parity_restores=2, snapshot_parity_suffix=4,
        maximum_teacher_trials=3, targets=[4.5, 6., 'path_heading'], maximum_new_physical_steps=1052,
        new_factor='Continuous current-mask-gated pure turn or original tiny creep until measured yaw target',
        earlier_short_fixed_turns_and_fixed_windows_already_failed=True,
        same625_v1_model_unchanged_weights=True, finite3frame_memory_from_initialization=True,
        original12actions_4cm_body_masks_and_deadline_unchanged=True,
        source_is_reused_development_failure=True, no_full_mission_or_rl_success_claim=True,
        no_automatic_training_labels=True, no_new_optimizer_updates=True, new_ppo_steps=0,
        no_parallel_physical_job=True, soft_cpu_cores=.35, memory_hard_max_mib=3072,
        goal_achieved=False, deployment_approved=False))

    import numpy as np
    import torch
    from experiments.probe_observed_turn_release import physical_step
    from experiments.recovery_state_snapshot import RecoverySnapshot
    from neural_navigation.experiment_budget import ExperimentBudget
    from neural_navigation.masked_actions import ACTIONS, masks_from_observation
    from neural_navigation.release_runtime import ReleaseRuntime
    from neural_navigation.stress_env import StressNavigationEnv
    from legal_turn_policy import LegalClockwiseManeuver

    torch.set_num_threads(1)
    budget = ExperimentBudget(.35, memory_mib=2800, directory=OUT)
    with np.load(source, allow_pickle=False) as saved:
        reference = {key: saved[key].copy() for key in saved.files}
    case = spec['case']
    env = StressNavigationEnv(case=case, components=bundle, scene_dir=ROOT/'training'/case['scene'],
        route_mode='shortcut', max_steps=case['max_steps'])
    env.reset(seed=case['seed'], options=dict(start=case['start'], goal=case['goal'][:2]))
    runtime = ReleaseRuntime(bundle, ROOT/'training'/case['scene']/'decision_map.yaml',
        case['start'], case['goal'], 'shortcut')
    runtime._branch_delegate = runtime.policy.policy

    class StrictPrefix:
        def __init__(self):
            self.cursor = 0
        def predict(self, observation):
            choice = int(reference['choices'][self.cursor])
            if not masks_from_observation(np.asarray(observation)[:282])[choice]:
                raise ValueError('Original initialization command is no longer admissible')
            self.cursor += 1
            return ACTIONS[choice].copy()

    runtime.policy.policy = StrictPrefix()
    report = dict(completed=False, rows=[], new_physical_steps=0, new_ppo_steps=0,
        teacher_diagnostics_not_trained_model=True, goal_achieved=False, deployment_approved=False)

    def finish(reason):
        for name, expected in hashes.items():
            assert digest(ROOT/name) == expected, name
        budget.tick()
        report.update(completed=True, reason=reason, inputs_verified=len(hashes), resource=budget.statistics())
        write('RESULTS.json', report)
        write('STATUS.json', dict(status='complete', reason=reason, completed_trials=len(report['rows'])))

    try:
        for index in range(530):
            for key, actual in [('scans', env.scan), ('odometry', env.observed_odometry), ('measured_velocity', env.measured_velocity)]:
                np.testing.assert_allclose(actual, reference[key][index], rtol=0, atol=2e-8, equal_nan=True)
            np.testing.assert_allclose(env.truth, reference['labels'][index, 1:4], rtol=0, atol=2e-8)
            _, packet = physical_step(env, runtime, budget)
            np.testing.assert_allclose(packet['commands'], reference['commands'][index], rtol=0, atol=2e-8)
            report['new_physical_steps'] += 1
            if index % 40 == 0:
                write('STATUS.json', dict(status='strict_original_physical_initialization', frame=index+1, target=530))
        np.testing.assert_allclose(env.truth, reference['labels'][530, 1:4], rtol=0, atol=2e-8)
        snapshot = RecoverySnapshot(env, runtime)
        original = []
        for _ in range(4):
            _, packet = physical_step(env, runtime, budget)
            original.append(packet)
            report['new_physical_steps'] += 1
        for repeat in range(2):
            restored_env, restored_runtime = snapshot.restore()
            try:
                for offset in range(4):
                    _, packet = physical_step(restored_env, restored_runtime, budget)
                    for key in packet:
                        np.testing.assert_array_equal(packet[key], original[offset][key])
                    report['new_physical_steps'] += 1
            finally:
                restored_env.close()
        report['two_physical_snapshot_restores_exact'] = True
        for target in [4.5, 6., 'path_heading']:
            trial_env, trial_runtime = snapshot.restore()
            teacher = LegalClockwiseManeuver(trial_runtime._branch_delegate, trial_runtime.core, target)
            trial_runtime.policy.policy = teacher
            trial_env.minimum_gap = trial_env._minimum_gap()
            packets, arrived, false_arrival = [], False, False
            try:
                for index in range(170):
                    result, packet = physical_step(trial_env, trial_runtime, budget)
                    packets.append(packet)
                    if result['stop_reason'] == 'estimated_goal_reached':
                        arrived = bool(packet['labels'][7] < .2)
                        false_arrival = not arrived
                        break
                    report['new_physical_steps'] += 1
                    if (trial_env.collision_events or trial_env.hard_guard_activations
                            or trial_env.minimum_gap < .04-1e-4):
                        break
                    if index % 25 == 0:
                        write('STATUS.json', dict(status='adaptive_legal_teacher_diagnostic', target=target,
                            frame=index+1, simulation_seconds=float(trial_env.elapsed), phase=teacher.phase,
                            measured_clockwise_turn_rad=teacher.clockwise_turn, previous_completed_trials=len(report['rows'])))
                path = OUT/('trial_'+str(target).replace('.', '_')+'.npz')
                np.savez_compressed(path, **{key: np.asarray([p[key] for p in packets]) for key in packets[0]})
                row = dict(target=target, safe_success=bool(arrived and not false_arrival
                    and not trial_env.collision_events and not trial_env.hard_guard_activations
                    and trial_env.minimum_gap >= .04-1e-4), physical_arrival=arrived,
                    false_arrival=false_arrival, collisions=trial_env.collision_events,
                    physical_guards=trial_env.hard_guard_activations, minimum_dynamic_gap_m=float(trial_env.minimum_gap),
                    post_release_seconds=float(trial_env.elapsed-106.), final_xy_error_m=float(np.linalg.norm(trial_env.truth[:2]-case['goal'][:2])),
                    measured_clockwise_turn_rad=teacher.clockwise_turn, final_phase=teacher.phase,
                    changed_from_model_frames=sum(r['changed_from_model'] for r in teacher.records),
                    teacher_records=teacher.records, trace_path=str(path.relative_to(ROOT)), trace_sha256=digest(path),
                    not_a_learned_model_success=True, automatic_training_admission=False)
                report['rows'].append(row)
                write('RESULTS.json', report)
                print(json.dumps({k:v for k,v in row.items() if k != 'teacher_records'}), flush=True)
                if row['safe_success']:
                    finish('One complete legal teacher recovery found; requires separate data and training review')
                    return
                if row['collisions'] or row['physical_guards'] or row['false_arrival'] or row['minimum_dynamic_gap_m'] < .04-1e-4:
                    finish('Safety check failed; remaining teacher trials stopped')
                    return
            finally:
                trial_env.close()
                del trial_runtime, trial_env
                gc.collect()
        finish('All three bounded legal maneuvers finished; no infeasibility or learned-success inference')
    finally:
        env.close()


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        if OUT.exists():
            write('FAILURE.json', dict(error=repr(error), automatically_restarted=False))
        raise
