"""Bounded all-direction sensor teacher after the exact v2 preservation job.

No trained model is promoted by a teacher trial. The two hardware variants
retain identical network weights; only the declared finite-free-memory config
differs, and every initialization command must remain currently admissible.
"""
from datetime import datetime
import gc
import hashlib
import json
from pathlib import Path
import subprocess
import time
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT/'reports/stop_feedback_20261015'
HERE = Path(__file__).resolve().parent
OUT = BASE/'late106_all_action_20261017'
DEPENDENCY = 'rl-guard-forecast-migration-validate-20261017.service'
INVOCATION = '76ac8e974503455a802f5be2998a7cac'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(name, value):
    path = OUT/name
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')
    temporary.replace(path)


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    task_path = BASE/'context_feedback_longfit_protocol_20261016/tasks.json'
    spec = json.loads(task_path.read_text())['actual106']
    source = ROOT/spec['trace']
    bundles = [('original_memory', ROOT/'models/stop_feedback_20261015/normal_suite_wait_pair_v2_s231016/wait_continuation/portable'),
        ('three_frame_memory', ROOT/'models/stop_feedback_20261015/memory_evidence_s231017/bc4500/portable')]
    paths = [Path(__file__), HERE/'legal_goal_teacher.py', HERE/'test_legal_goal_teacher.py',
        HERE/'CLOCKWISE_TEACHER_AUDIT.json', task_path, source,
        BASE/'guard_forecast_migration_validation_20261017/PLAN.json']
    for _, bundle in bundles:
        manifest = json.loads((bundle/'manifest.json').read_text())
        paths += [bundle/'manifest.json'] + [bundle/name for name in manifest['files']]
    for folder in ['experiments', 'neural_navigation', 'navrl']:
        paths += list((ROOT/'training'/folder).rglob('*.py'))
    paths += [p for p in (ROOT/'training'/spec['case']['scene']).iterdir() if p.is_file()]
    hashes = {str(p.relative_to(ROOT)): digest(p) for p in set(paths)}
    write('PLAN.json', dict(at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(), inputs=hashes,
        dependency=DEPENDENCY, expected_invocation=INVOCATION, target_modes=['waypoint', 'goal'],
        hardware_variants=[name for name, _ in bundles], identical_frozen_networks=True,
        maximum_teacher_trials=4, maximum_new_physical_steps=1764,
        original12actions_and_current_masks=True, finite_response_ranking_from_sensor_velocity=True,
        no_simulator_truth_teacher_inputs=True, collision_margin_m=.04,
        no_deadline_extension=True, stop_remaining_on_safety_failure=True,
        all_direction_coverage_not_only_clockwise=True,
        shared_prefix_initialization_not_autonomous_navigation=True,
        prospective_diagnostic_not_new_generalization=True,
        no_automatic_training_admission=True, no_automatic_promotion=True,
        no_parallel_physical_job=True, soft_cpu_cores=.35, memory_hard_max_mib=3072,
        new_ppo_steps=0, goal_achieved=False, deployment_approved=False))
    while True:
        checked = subprocess.run(['systemctl', '--user', 'show', DEPENDENCY,
            '--property=MainPID,ActiveState,InvocationID,ExecMainStatus'], capture_output=True, text=True, check=True)
        state = dict(line.split('=', 1) for line in checked.stdout.splitlines() if '=' in line)
        if state['InvocationID'] not in ('', INVOCATION):
            raise RuntimeError('The exact preservation dependency changed invocation')
        if state['MainPID'] == '0':
            result_path = BASE/'guard_forecast_migration_validation_20261017/RESULTS.json'
            if (state['ActiveState'] != 'inactive' or state['ExecMainStatus'] != '0'
                    or not result_path.exists() or not json.loads(result_path.read_text())['completed']):
                raise RuntimeError('Preservation process did not finish normally')
            validation = json.loads(result_path.read_text())
            hashes[str(result_path.relative_to(ROOT))] = digest(result_path)
            break
        write('STATUS.json', dict(status='waiting_for_exact_v2_preservation', dependency=state,
            neural_and_physics_libraries_not_loaded=True, new_physical_steps=0, new_ppo_steps=0))
        time.sleep(5.)
    for name, expected in hashes.items():
        assert digest(ROOT/name) == expected, name
    write('DEPENDENCIES.json', dict(inputs=hashes,
        migration_preservation_passed=validation.get('all_fixed_retention_passed', False),
        teacher_is_diagnosis_not_a_waiver_of_training_or_deployment_gates=True))

    import numpy as np
    import torch
    from experiments.probe_observed_turn_release import physical_step
    from experiments.recovery_state_snapshot import RecoverySnapshot
    from neural_navigation.experiment_budget import ExperimentBudget
    from neural_navigation.goal_pose_runtime import check_files
    from neural_navigation.masked_actions import ACTIONS, masks_from_observation
    from neural_navigation.release_runtime import ReleaseRuntime
    from neural_navigation.stress_env import StressNavigationEnv
    from legal_goal_teacher import LegalGoalTeacher

    torch.set_num_threads(1)
    budget = ExperimentBudget(.35, memory_mib=2800, directory=OUT)
    with np.load(source, allow_pickle=False) as stored:
        reference = {key: stored[key].copy() for key in stored.files}
    for name in ['actor.pt', 'localizer.pt', 'router.pt', 'parent_actor.pt', 'docking_contract.json']:
        assert digest(bundles[0][1]/name) == digest(bundles[1][1]/name), name
    report = dict(completed=False, rows=[], new_physical_steps=0, new_ppo_steps=0,
        teacher_diagnostics_not_learned_results=True, goal_achieved=False, deployment_approved=False)

    def finish(reason):
        for name, expected in hashes.items():
            assert digest(ROOT/name) == expected, name
        assert report['new_physical_steps'] <= 1764
        report.update(completed=True, reason=reason, inputs_verified=len(hashes), resource=budget.statistics())
        write('RESULTS.json', report)
        write('STATUS.json', dict(status='complete', completed_trials=len(report['rows']), reason=reason))

    def advance(env, runtime):
        result, packet = physical_step(env, runtime, budget)
        contacts = result['predictive_constraint']
        packet['constraint_diagnostics'] = np.asarray([contacts['supported_tracks'], contacts['rejected_nonzero_choices'], 0.,
            *[np.nan if x is None else x for x in contacts['predicted_contact_seconds']]], float)
        report['new_physical_steps'] += int(result['stop_reason'] != 'estimated_goal_reached')
        return result, packet

    class StrictPrefix:
        def __init__(self):
            self.cursor = 0

        def predict(self, observation):
            choice = int(reference['choices'][self.cursor])
            if not masks_from_observation(np.asarray(observation)[:282])[choice]:
                raise ValueError('Prefix choice no longer sensor-admissible')
            self.cursor += 1
            return ACTIONS[choice].copy()

    case = spec['case']
    for hardware, bundle in bundles:
        check_files(bundle, json.loads((bundle/'manifest.json').read_text()))
        env = StressNavigationEnv(case=case, components=bundle, scene_dir=ROOT/'training'/case['scene'],
            route_mode='shortcut', max_steps=case['max_steps'])
        env.reset(seed=case['seed'], options=dict(start=case['start'], goal=case['goal']))
        runtime = ReleaseRuntime(bundle, ROOT/'training'/case['scene']/'decision_map.yaml',
            case['start'], case['goal'], 'shortcut')
        runtime._branch_delegate = runtime.policy.policy
        runtime.policy.policy = StrictPrefix()
        prefix_packets = []
        try:
            for index in range(530):
                for key, actual in [('scans', env.scan), ('odometry', env.observed_odometry), ('measured_velocity', env.measured_velocity)]:
                    np.testing.assert_allclose(actual, reference[key][index], rtol=0, atol=2e-8, equal_nan=True)
                np.testing.assert_allclose(env.truth, reference['labels'][index, 1:4], rtol=0, atol=2e-8)
                _, packet = advance(env, runtime)
                np.testing.assert_allclose(packet['commands'], reference['commands'][index], rtol=0, atol=2e-8)
                prefix_packets.append(packet)
                if index % 50 == 0:
                    write('STATUS.json', dict(status='strict_physical_prefix', hardware=hardware, frame=index+1))
            np.testing.assert_allclose(env.truth, reference['labels'][530, 1:4], rtol=0, atol=2e-8)
            snapshot = RecoverySnapshot(env, runtime)
            original = [advance(env, runtime)[1] for _ in range(4)]
            for _ in range(2):
                restored_env, restored_runtime = snapshot.restore()
                try:
                    for index in range(4):
                        _, packet = advance(restored_env, restored_runtime)
                        for key in packet:
                            np.testing.assert_array_equal(packet[key], original[index][key])
                finally:
                    restored_env.close()
                    del restored_env, restored_runtime
                    gc.collect()
            for mode in ['waypoint', 'goal']:
                trial_env, trial_runtime = snapshot.restore()
                teacher = LegalGoalTeacher(trial_runtime._branch_delegate, trial_runtime.core, mode)
                trial_runtime.policy.policy = teacher
                trial_env.minimum_gap = trial_env._minimum_gap()
                packets, arrived, false_arrival = [], False, False
                try:
                    for index in range(170):
                        result, packet = advance(trial_env, trial_runtime)
                        packets.append(packet)
                        if result['stop_reason'] == 'estimated_goal_reached':
                            arrived = bool(packet['labels'][7] < .2)
                            false_arrival = not arrived
                            break
                        if (trial_env.collision_events or trial_env.hard_guard_activations
                                or trial_env.minimum_gap < .04-1e-4):
                            break
                        if index % 25 == 0:
                            write('STATUS.json', dict(status='all_legal_sensor_teacher', hardware=hardware,
                                mode=mode, frame=index+1, simulation_seconds=float(trial_env.elapsed)))
                    all_packets = prefix_packets+packets
                    trace = {key: np.asarray([packet[key] for packet in all_packets]) for key in all_packets[0]}
                    trace['teacher_controlled'] = np.arange(len(all_packets)) >= 530
                    path = OUT/(hardware+'_'+mode+'.npz')
                    np.savez_compressed(path, **trace)
                    row = dict(hardware=hardware, mode=mode,
                        safe_success=bool(arrived and not false_arrival and not trial_env.collision_events
                            and not trial_env.hard_guard_activations and trial_env.minimum_gap >= .04-1e-4),
                        physical_arrival=arrived, false_arrival=false_arrival, collisions=trial_env.collision_events,
                        physical_guards=trial_env.hard_guard_activations, minimum_dynamic_gap_m=float(trial_env.minimum_gap),
                        post_release_seconds=float(trial_env.elapsed-106.),
                        final_xy_error_m=float(np.linalg.norm(trial_env.truth[:2]-case['goal'])),
                        teacher_records=teacher.records, trace_path=str(path.relative_to(ROOT)), trace_sha256=digest(path),
                        original_prefix_sensor_and_physics_exact=True, snapshot_two_restores_exact=True,
                        not_a_learned_model_success=True, automatic_training_admission=False)
                    report['rows'].append(row)
                    write('RESULTS.json', report)
                    print(json.dumps({key:value for key, value in row.items() if key != 'teacher_records'}), flush=True)
                    if row['safe_success']:
                        finish('One complete all-direction teacher recovery; separate data/memory/training admission required')
                        return
                    if row['collisions'] or row['physical_guards'] or row['false_arrival'] or row['minimum_dynamic_gap_m'] < .04-1e-4:
                        finish('Teacher safety failed; remaining diagnoses stopped')
                        return
                finally:
                    trial_env.close()
                    del teacher, trial_env, trial_runtime
                    gc.collect()
            snapshot.verify_shared_models()
            del snapshot
        finally:
            env.close()
            del env, runtime
            gc.collect()
    finish('All declared teacher trials finished; no infeasibility claim')


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        if OUT.exists():
            write('FAILURE.json', dict(error=repr(error), automatically_restarted=False))
        raise
