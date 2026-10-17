"""Fresh-process exact replay of a complete positive-free teacher candidate.

This verifies a recorded diagnostic, not autonomous learned policy performance.
No model, optimizer, production memory rule, or deployment is modified.
"""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
BASE = HERE.parent


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    value.update(at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
        new_ppo_steps=0, new_training_updates=0, goal_achieved=False,
        deployment_approved=False)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def main(job_path):
    job = read(job_path)
    output = ROOT / job['directory']
    assert not output.exists()
    for name, expected in job['inputs'].items():
        assert digest(ROOT / name) == expected, name
    running = subprocess.run(['systemctl', '--user', 'list-units', '--type=service',
        '--state=running', '--no-legend', '--no-pager', 'rl-*'],
        capture_output=True, text=True, check=True)
    allowed = {'rl-positive-free-fresh-replay-20261017.service',
               'rl-positive-free-replay-admission-20261017.service'}
    assert all(line.split()[0] in allowed for line in running.stdout.splitlines() if line.strip())
    output.mkdir()
    write(output / 'PLAN.json', dict(job=job, fresh_process=True,
        memory_installed_before_first_scan=True, compare_every_recorded_field=True,
        teacher_not_model_controlled=True, no_automatic_training_admission=True,
        maximum_new_physical_steps=700, original_body_4cm_actions_dynamics_deadline=True))

    import numpy as np
    import torch
    sys.path.insert(0, str(BASE / 'positive_free_shadow_v2_20261017'))
    from evidence_memory import EvidenceMemory
    from experiments.probe_observed_turn_release import physical_step
    from neural_navigation.aligned_release_runtime import AlignedReleaseRuntime
    from neural_navigation.experiment_budget import ExperimentBudget
    from neural_navigation.masked_actions import ACTIONS, masks_from_observation
    from neural_navigation.stress_env import StressNavigationEnv

    torch.set_num_threads(1)
    budget = ExperimentBudget(.35, 2800, output)
    with np.load(ROOT / job['expected_trace'], allow_pickle=False) as archive:
        expected = {name: archive[name].copy() for name in archive.files}
    total = len(expected['choices'])
    assert 530 < total <= 700
    assert not expected['model_controlled'].any()
    assert expected['initialization_prefix'][:530].all()
    assert not expected['initialization_prefix'][530:].any()
    assert (expected['experimental_memory_rule'] == 'strong_OR_stable_finite_free_v1').all()
    case, bundle = job['case'], ROOT / job['bundle']
    env = StressNavigationEnv(case=case, components=bundle,
        scene_dir=ROOT / 'training' / case['scene'], route_mode='shortcut', max_steps=700)
    env.reset(seed=case['seed'], options=dict(start=case['start'], goal=case['goal'][:2]))
    runtime = AlignedReleaseRuntime(bundle, ROOT / 'training' / case['scene'] / 'decision_map.yaml',
        case['start'], case['goal'], 'shortcut')
    assert not runtime.guard.memory.entries and runtime.guard.memory.last_time is None
    runtime.guard.memory = EvidenceMemory(runtime.guard.memory.config)

    class Recorded:
        def __init__(self):
            self.cursor = 0
        def predict(self, observation):
            choice = int(expected['choices'][self.cursor])
            self.cursor += 1
            assert masks_from_observation(np.asarray(observation)[:282])[choice]
            return ACTIONS[choice].copy()

    runtime.policy.policy = Recorded()
    packets = []
    success = False
    def unsafe():
        return bool(env.collision_events or env.hard_guard_activations
            or env.minimum_gap < .04 - 1e-4)
    try:
        for index in range(total):
            result, packet = physical_step(env, runtime, budget)
            packet['current_action_reason_strings'] = np.asarray(
                [reason or 'none' for reason in result['current_action_rejection_reasons']], dtype='U64')
            packet['executed_stop_reason'] = np.asarray(result['decision_feedback_executed_reason'], dtype='U64')
            packet['initialization_prefix'] = np.asarray(index < 530)
            packet['expert_continuation'] = np.asarray(index >= 530)
            packet['model_controlled'] = np.asarray(False)
            packet['experimental_memory_rule'] = np.asarray('strong_OR_stable_finite_free_v1', dtype='U64')
            assert set(packet) == set(expected), 'Every source field must be checked'
            for name, value in packet.items():
                np.testing.assert_array_equal(value, expected[name][index],
                    err_msg=f'Fresh replay frame {index}, field {name}')
            packets.append(packet)
            assert not unsafe(), 'Replay physical safety failure'
            if result['stop_reason'] == 'estimated_goal_reached':
                assert index == total - 1, 'Arrival must occur at the same final frame'
                success = bool(packet['labels'][7] < .2)
                break
            assert index != total - 1, 'Recorded candidate did not reproduce arrival'
            if index % 80 == 0:
                write(output / 'STATUS.json', dict(stage='fresh_from_init_exact_replay',
                    exact_frames=index + 1, total_frames=total))
        assert success and len(packets) == total
        assert env.elapsed <= 140. + 1e-8
        for name, expected_hash in job['inputs'].items():
            assert digest(ROOT / name) == expected_hash, name
        arrays = {name: np.asarray([packet[name] for packet in packets]) for name in packets[0]}
        trace = output / 'trace.npz'
        np.savez_compressed(trace, **arrays)
        budget.tick()
        write(output / 'RESULTS.json', dict(completed=True,
            complete_safe_fresh_process_teacher_replay_exact=True,
            exact_fields=sorted(arrays), exact_frames=total,
            new_physical_steps=total, original_initialization_prefix_frames=530,
            seconds=float(env.elapsed), post_release_seconds=float(env.elapsed - 106.),
            final_xy_error_m=float(packets[-1]['labels'][7]),
            collisions=env.collision_events, physical_guards=env.hard_guard_activations,
            minimum_dynamic_gap_m=float(env.minimum_gap),
            trace_path=str(trace.relative_to(ROOT)), trace_sha256=digest(trace),
            teacher_not_autonomous_learned_success=True, changed_memory_not_adopted=True,
            full_original28_known6_hard97_6_106_96_and_fresh8_required=True,
            no_automatic_training_admission=True, resource=budget.statistics()))
        write(output / 'STATUS.json', dict(stage='complete', exact_safe_teacher_replay=True))
    finally:
        env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('job', type=Path)
    args = parser.parse_args()
    try:
        main(args.job)
    except Exception as error:
        output = ROOT / read(args.job)['directory']
        if output.exists():
            write(output / 'FAILURE.json', dict(error=repr(error), no_auto_restart=True))
        raise
