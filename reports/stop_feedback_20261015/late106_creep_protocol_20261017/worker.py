"""One fresh, sensor-gated teacher trial or exact fresh-process replay."""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parents[3]


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def main(job_path):
    job = read(job_path)
    out = ROOT / job['directory']
    out.mkdir()
    for name, expected in job['inputs'].items():
        assert digest(ROOT / name) == expected, name
    if job['mode'] == 'replay':
        assert digest(ROOT / job['expected_trace']) == job['expected_sha256']

    import numpy as np
    import torch
    from experiments.probe_observed_turn_release import physical_step
    from navrl.geometry import wrap
    from neural_navigation.aligned_release_runtime import AlignedReleaseRuntime
    from neural_navigation.experiment_budget import ExperimentBudget
    from neural_navigation.masked_actions import ACTIONS, masks_from_observation
    from neural_navigation.stress_env import StressNavigationEnv

    torch.set_num_threads(1)
    budget = ExperimentBudget(.35, 2800, out)
    reference_path = ROOT / job['source_trace']
    with np.load(reference_path, allow_pickle=False) as data:
        source = {key: data[key].copy() for key in data.files}
    expected_trace = None
    if job['mode'] == 'replay':
        with np.load(ROOT / job['expected_trace'], allow_pickle=False) as data:
            expected_trace = {key: data[key].copy() for key in data.files}
    case = job['case']
    bundle = ROOT / job['bundle']
    env = StressNavigationEnv(case=case, components=bundle,
        scene_dir=ROOT / 'training' / case['scene'], route_mode='shortcut', max_steps=700)
    env.reset(seed=case['seed'], options=dict(start=case['start'], goal=case['goal'][:2]))
    runtime = AlignedReleaseRuntime(bundle, ROOT / 'training' / case['scene'] / 'decision_map.yaml',
        case['start'], case['goal'], 'shortcut')
    delegate = runtime.policy.policy

    class Recorded:
        def __init__(self, choices):
            self.choices, self.cursor = choices, 0
        def predict(self, observation):
            choice = int(self.choices[self.cursor]); self.cursor += 1
            assert masks_from_observation(np.asarray(observation)[:282])[choice]
            return ACTIONS[choice].copy()

    class CreepTurn:
        def __init__(self, target):
            self.target = target
            self.previous_yaw = float(runtime.core.odom[2])
            self.clockwise = 0.
            self.records = []
        def predict(self, observation):
            yaw = float(runtime.core.odom[2])
            self.clockwise -= float(wrap(yaw - self.previous_yaw))
            self.previous_yaw = yaw
            mask = masks_from_observation(np.asarray(observation)[:282])
            nominal = delegate.predict(observation)
            baseline = int(np.abs(ACTIONS - nominal).max(1).argmin())
            assert mask[baseline]
            turning = self.clockwise < self.target and runtime.core.elapsed < 112. - 1e-8
            choice = next((i for i in (9, 5, 7) if mask[i]), baseline) if turning else baseline
            self.records.append(dict(sensor_time_s=float(runtime.core.elapsed),
                clockwise_odometry_rad=self.clockwise, target_rad=self.target,
                phase='bounded_creep_turn' if turning else 'frozen_actor',
                original_choice=baseline, choice=choice, changed=choice != baseline))
            return ACTIONS[choice].copy()

    runtime.policy.policy = Recorded(source['choices'][:530])
    packets = []
    steps = 0
    def step(initialization=False):
        nonlocal steps
        assert steps < 700
        result, packet = physical_step(env, runtime, budget)
        packet['current_action_reason_strings'] = np.asarray(
            [reason or 'none' for reason in result['current_action_rejection_reasons']], dtype='U64')
        packet['executed_stop_reason'] = np.asarray(result['decision_feedback_executed_reason'], dtype='U64')
        packet['initialization_prefix'] = np.asarray(initialization)
        packet['expert_continuation'] = np.asarray(not initialization)
        packet['model_controlled'] = np.asarray(False)
        packets.append(packet)
        steps += 1
        return result, packet

    def unsafe():
        return bool(env.collision_events or env.hard_guard_activations or env.minimum_gap < .04 - 1e-4)

    success = False
    reason = 'Original deadline reached'
    teacher = None
    try:
        for index in range(530):
            _, packet = step(True)
            for name in ('observations', 'masks', 'choices', 'commands', 'scans', 'odometry', 'measured_velocity'):
                np.testing.assert_array_equal(packet[name], source[name][index])
            np.testing.assert_allclose(packet['labels'][:7], source['labels'][index, :7], rtol=0, atol=2e-8)
            if index % 80 == 0:
                write(out / 'STATUS.json', dict(stage='exact_current625_prefix', frame=index + 1, target=530))
        assert not unsafe(), 'Original prefix does not admit the unchanged physical safety gate'
        assert abs(env.elapsed - 106.) < 2e-8
        if job['mode'] == 'probe':
            teacher = CreepTurn(job['target_rad'])
            runtime.policy.policy = teacher
            remaining = 170
        else:
            runtime.policy.policy = Recorded(expected_trace['choices'][530:])
            remaining = len(expected_trace['choices']) - 530
            assert 0 < remaining <= 170
        for offset in range(remaining):
            result, packet = step()
            if result['stop_reason'] == 'estimated_goal_reached':
                success = bool(packet['labels'][7] < .2 and not unsafe())
                reason = 'Physical arrival' if success else 'False or unsafe arrival rejected'
                break
            if unsafe():
                reason = 'Physical safety failure rejected'
                break
            if offset % 20 == 0:
                write(out / 'STATUS.json', dict(stage='sensor_gated_expert_continuation',
                    elapsed_s=float(env.elapsed), frames=offset + 1,
                    target_rad=job.get('target_rad'),
                    legal_motion_actions=int(packet['masks'][1:].sum())))
        assert steps <= 700 and env.elapsed <= 140. + 1e-8
        arrays = {name: np.asarray([p[name] for p in packets]) for name in packets[0]}
        exact_replay = False
        if job['mode'] == 'replay':
            for name, expected in expected_trace.items():
                np.testing.assert_array_equal(arrays[name], expected)
            assert success, 'A winner must reproduce complete safe arrival'
            exact_replay = True
        trace = out / 'trace.npz'
        np.savez_compressed(trace, **arrays)
        if teacher is not None:
            write(out / 'TEACHER_DECISIONS.json', dict(records=teacher.records,
                sensor_odometry_only=True, never_policy_input_or_runtime_deployment=True))
        for name, expected in job['inputs'].items():
            assert digest(ROOT / name) == expected, name
        if job['mode'] == 'replay':
            assert digest(ROOT / job['expected_trace']) == job['expected_sha256']
        budget.tick()
        row = dict(completed=True, at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
            mode=job['mode'], target_rad=job.get('target_rad'), safe_teacher_success=success,
            reason=reason, frames=len(packets), new_physical_steps=steps,
            original_prefix_frames=530, original_prefix625_sensor_physics_and_commands_exact=True,
            release_sensor_timestamp_s=106., seconds=float(env.elapsed),
            post_release_seconds=float(env.elapsed - 106.), original_deadline_s=140.,
            final_xy_error_m=float(packets[-1]['labels'][7]),
            collisions=env.collision_events, physical_guards=env.hard_guard_activations,
            minimum_dynamic_gap_m=float(env.minimum_gap),
            complete_fresh_process_teacher_replay_exact=exact_replay,
            trace_path=str(trace.relative_to(ROOT)), trace_sha256=digest(trace),
            offline_teacher_not_learned_policy_success=True,
            no_automatic_training_admission=True, new_ppo_steps=0, new_training_updates=0,
            resource=budget.statistics(), goal_achieved=False, deployment_approved=False)
        write(out / 'RESULTS.json', row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
    finally:
        env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('job', type=Path)
    args = parser.parse_args()
    try:
        main(args.job)
    except Exception as error:
        job = read(args.job)
        directory = ROOT / job['directory']
        if directory.exists():
            write(directory / 'FAILURE.json', dict(error=repr(error), no_auto_restart=True))
        raise
