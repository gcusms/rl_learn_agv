"""From-initialization experimental memory and full safe teacher diagnostic.

Frozen model files and original body/actions/dynamics/deadline. No memory is
adopted for deployment or training by this test. Changed observations/masks are
explicitly recorded, never relabelled as the original contract's observations.
"""
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
OUT = BASE / 'positive_free_physics_20261017'
OWN_UNIT = 'rl-positive-free-physics-20261017.service'


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(name, value):
    value.update(at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
        new_ppo_steps=0, new_training_updates=0, goal_achieved=False, deployment_approved=False)
    path = OUT / name
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def main():
    assert not OUT.exists()
    for unit, invocation in [
        ('rl-late106-compact-beam-20261017.service', 'aaf1528f6aaf4017a374919569122bd8'),
        ('rl-positive-free-shadow-v2-20261017.service', 'fb9b9fe346d5493ca418e6f419db671f')]:
        result = subprocess.run(['systemctl', '--user', 'show', unit, '-p', 'MainPID',
            '-p', 'ActiveState', '-p', 'InvocationID', '-p', 'ExecMainStatus'],
            capture_output=True, text=True, check=True)
        state = dict(line.split('=', 1) for line in result.stdout.splitlines() if '=' in line)
        assert state['InvocationID'] in ('', invocation), state
        assert state['MainPID'] == '0' and state['ActiveState'] == 'inactive' and state['ExecMainStatus'] == '0', state
    running = subprocess.run(['systemctl', '--user', 'list-units', '--type=service',
        '--state=running', '--no-legend', '--no-pager', 'rl-*'], capture_output=True, text=True, check=True)
    allowed = {OWN_UNIT, 'rl-positive-free-physics-admission-v2-20261017.service'}
    assert all(line.split()[0] in allowed for line in running.stdout.splitlines() if line.strip())
    shadow_path = BASE / 'positive_free_shadow_v2_20261017/RESULTS.json'
    shadow = read(shadow_path)
    assert shadow['completed'] and shadow['all_recorded_retained_point_sets_are_original_subsets']
    assert any(row['trace'] in ('current106', 'creep04_failed')
        and row['sampled_extra_geometry_allowances'] > 0 for row in shadow['summaries'])
    assert read(BASE / 'positive_free_shadow_v2_20261017/EVIDENCE_CHECKS.json')['completed']
    assert not read(BASE / 'late106_compact_beam_20261017/RESULTS.json')['safe_teacher_candidate_found']
    validation_path = BASE / 'full_navigation_joint_linear_validation_20261017/RESULTS.json'
    validation = read(validation_path)
    failure = validation['groups']['hard3']['rows'][0]
    assert failure['case_id'] == 'w11_training_stall_t106' and failure['timeout']
    source_path = ROOT / failure['trace_path']
    assert digest(source_path) == failure['trace_sha256']
    tasks_path = BASE / 'context_feedback_longfit_protocol_20261016/tasks.json'
    case = read(tasks_path)['actual106']['case']
    bundle = ROOT / validation['model'] / 'portable'
    extra = [BASE / 'positive_free_shadow_v2_20261017', BASE / 'late106_compact_beam_protocol_20261017',
             BASE / 'late106_all_action_protocol_20261017']
    for directory in extra:
        sys.path.insert(0, str(directory))
    paths = [Path(__file__), shadow_path, validation_path, source_path, tasks_path,
        extra[0] / 'evidence_memory.py', extra[0] / 'EVIDENCE_CHECKS.json',
        extra[1] / 'compact_state.py', extra[2] / 'legal_goal_teacher.py', bundle / 'manifest.json']
    paths += [bundle / name for name in read(bundle / 'manifest.json')['files']]
    for package in ('experiments', 'neural_navigation', 'navrl'):
        paths += list((ROOT / 'training' / package).rglob('*.py'))
    paths += [p for p in (ROOT / 'training' / case['scene']).iterdir() if p.is_file()]
    hashes = {str(path.relative_to(ROOT)): digest(path) for path in set(paths)}
    OUT.mkdir()
    write('PLAN.json', dict(inputs=hashes, case=case, frozen_source_model=validation['model'],
        memory_experiment='original_fast_free_OR_three_consecutive_finite_3cm_rays',
        memory_installed_before_first_scan_at_initialization=True,
        original_12actions_body4cm_uncertainty_unknown_blind_dynamics_deadline_retained=True,
        changed_memory_not_adopted=True, observations_are_explicit_new_shadow_semantics=True,
        original_physical_and_sensor_prefix_frames=530,
        full_original625_observation_equality_not_claimed_for_changed_memory=True,
        mandatory_two_exact_new_memory_continuation_replays=True,
        maximum_teacher_trials=2, target_modes=['waypoint', 'goal'],
        original_deadline_s=140., maximum_new_physical_steps=878,
        full_fresh_process_winner_replay_and_complete_retention_still_required=True,
        no_automatic_training_admission=True, soft_cpu_cores=.35, hard_memory_mib=3072))

    import numpy as np
    import torch
    from compact_state import CompactSnapshot, model_digest
    from evidence_memory import EvidenceMemory
    from legal_goal_teacher import LegalGoalTeacher
    from experiments.probe_observed_turn_release import physical_step
    from neural_navigation.aligned_release_runtime import AlignedReleaseRuntime
    from neural_navigation.experiment_budget import ExperimentBudget
    from neural_navigation.masked_actions import ACTIONS, masks_from_observation
    from neural_navigation.stress_env import StressNavigationEnv

    torch.set_num_threads(1)
    budget = ExperimentBudget(.35, 2800, OUT)
    with np.load(source_path, allow_pickle=False) as saved:
        source = {key: saved[key].copy() for key in saved.files}
    env = StressNavigationEnv(case=case, components=bundle,
        scene_dir=ROOT / 'training' / case['scene'], route_mode='shortcut', max_steps=700)
    env.reset(seed=case['seed'], options=dict(start=case['start'], goal=case['goal'][:2]))
    runtime = AlignedReleaseRuntime(bundle, ROOT / 'training' / case['scene'] / 'decision_map.yaml',
        case['start'], case['goal'], 'shortcut')
    assert not runtime.guard.memory.entries and runtime.guard.memory.last_time is None
    runtime.guard.memory = EvidenceMemory(runtime.guard.memory.config)
    runtime._branch_delegate = runtime.policy.policy
    native_data = env.data
    original_native_hash = model_digest(env.model)

    class Recorded:
        def __init__(self, sequence=None):
            self.sequence, self.cursor = sequence, 0
        def predict(self, observation):
            choice = int(self.sequence[self.cursor]) if self.sequence is not None else 0
            self.cursor += 1
            assert masks_from_observation(np.asarray(observation)[:282])[choice], (self.cursor, choice)
            return ACTIONS[choice].copy()

    runtime.policy.policy = Recorded(source['choices'][:530])
    count = 0
    prefix = []
    rows = []
    def step(e, r, initialization=False):
        nonlocal count
        assert count < 878
        result, packet = physical_step(e, r, budget)
        count += 1
        packet['current_action_reason_strings'] = np.asarray(
            [reason or 'none' for reason in result['current_action_rejection_reasons']], dtype='U64')
        packet['executed_stop_reason'] = np.asarray(result['decision_feedback_executed_reason'], dtype='U64')
        packet['initialization_prefix'] = np.asarray(initialization)
        packet['expert_continuation'] = np.asarray(not initialization)
        packet['model_controlled'] = np.asarray(False)
        packet['experimental_memory_rule'] = np.asarray('strong_OR_stable_finite_free_v1', dtype='U64')
        return result, packet
    def unsafe(e):
        return bool(e.collision_events or e.hard_guard_activations or e.minimum_gap < .04 - 1e-4)
    winner = None
    try:
        changed_prefix_masks = 0
        for index in range(530):
            _, packet = step(env, runtime, True)
            prefix.append(packet)
            for name in ('choices', 'commands', 'scans', 'odometry', 'measured_velocity'):
                np.testing.assert_array_equal(packet[name], source[name][index])
            np.testing.assert_array_equal(packet['observations'][:235], source['observations'][index, :235])
            np.testing.assert_allclose(packet['labels'][:7], source['labels'][index, :7], rtol=0, atol=2e-8)
            changed_prefix_masks += int(not np.array_equal(packet['masks'], source['masks'][index]))
            if index % 80 == 0:
                write('STATUS.json', dict(stage='new_memory_from_init_admissible_physical_prefix', frame=index + 1))
        assert not unsafe(env)
        runtime.policy.policy = Recorded()
        root = CompactSnapshot(env, runtime)
        first = []
        for repeat in range(2):
            e, r = root.restore(native_data)
            current = []
            for _ in range(4):
                _, packet = step(e, r)
                current.append(packet)
            if repeat:
                for a, b in zip(first, current):
                    for name in a:np.testing.assert_array_equal(a[name], b[name])
            else:
                first = current
            del e, r
        write('FROM_INIT_ADMISSION.json', dict(completed=True,
            source_physical_sensor_commands_and_base235_exact=True,
            original_prefix_frames=530, changed_prefix_mask_frames=changed_prefix_masks,
            all_prefix_commands_currently_admissible=True, two4frame_new_memory_continuations_exact=True,
            changed_memory_not_full_retention_or_adoption=True, resource=budget.statistics()))
        for mode in ['waypoint', 'goal']:
            e, r = root.restore(native_data)
            teacher = LegalGoalTeacher(r._branch_delegate, r.core, mode)
            r.policy.policy = teacher
            suffix = []
            success = False
            reason = 'Original140second deadline reached'
            for offset in range(170):
                result, packet = step(e, r)
                suffix.append(packet)
                if result['stop_reason'] == 'estimated_goal_reached':
                    success = bool(packet['labels'][7] < .2 and not unsafe(e))
                    reason = 'Complete physical arrival' if success else 'False or unsafe arrival rejected'
                    break
                if unsafe(e):
                    reason = 'Safety failure; abort remaining trials'
                    break
                if offset % 20 == 0:
                    write('STATUS.json', dict(stage='sensor_only_original_action_teacher', mode=mode,
                        elapsed_s=float(e.elapsed), suffix_frames=offset + 1,
                        legal_motion_actions=int(packet['masks'][1:].sum())))
            packets = prefix + suffix
            path = OUT / ('teacher_' + mode + '.npz')
            np.savez_compressed(path, **{name: np.asarray([p[name] for p in packets]) for name in packets[0]})
            write('TEACHER_' + mode.upper() + '_DECISIONS.json', dict(records=teacher.records))
            row = dict(mode=mode, safe_success=success, reason=reason,
                seconds=float(e.elapsed), post_release_seconds=float(e.elapsed - 106.),
                final_xy_error_m=float(suffix[-1]['labels'][7]),
                collisions=e.collision_events, physical_guards=e.hard_guard_activations,
                minimum_dynamic_gap_m=float(e.minimum_gap),
                trace_path=str(path.relative_to(ROOT)), trace_sha256=digest(path),
                frames=len(packets), original_prefix_frames=530,
                experimental_memory_from_initialization=True,
                offline_teacher_not_learned_success=True, independent_fresh_replay_passed=False)
            rows.append(row)
            write('INTERIM.json', dict(rows=rows, new_physical_steps=count))
            if success:
                winner = row
                break
            if unsafe(e):
                break
            del e, r, teacher
        assert model_digest(env.model) == original_native_hash
        for path, sha in hashes.items():assert digest(ROOT / path) == sha, path
        write('RESULTS.json', dict(completed=True, rows=rows, safe_teacher_candidate_found=winner is not None,
            new_physical_steps=count, original_model_files_and_native_physics_unchanged=True,
            changed_memory_not_adopted=True, independent_fresh_replay_required=winner is not None,
            full_original28_known6_hard97_6_106_96_and_fresh8_requirements_retained=True,
            no_automatic_training_admission=True, resource=budget.statistics()))
        write('STATUS.json', dict(stage='complete', safe_teacher_candidate_found=winner is not None))
    finally:
        env.close()


if __name__ == '__main__':
    try:main()
    except Exception as error:
        if OUT.exists():write('FAILURE.json', dict(error=repr(error), no_auto_restart=True))
        raise
