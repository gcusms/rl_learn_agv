"""Bounded offline search for a completely safe, originally legal recovery.

This is a teacher feasibility diagnostic, not runtime planning or RL success.
It may run only after the exact current validation and its neural audit finish.
Search truth is used for offline scoring only; every action still goes through
the original current-sensor mask, execution guard and physical dynamics.
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
OUT = BASE / 'late106_beam_20261017'
OWN_UNIT = 'rl-late106-beam-20261017.service'
MAX_STEPS = 4300


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(name, value):
    value = dict(value, at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat())
    path = OUT / name
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def main():
    assert not OUT.exists()
    OUT.mkdir(parents=True)
    for unit, invocation in [
        ('rl-full-navigation-margin-resume-cycle-20261017.service', 'ef95cccf487943edb1a993b9732a2520'),
        ('rl-full-navigation-mechanism-post-20261017.service', 'a4e3d1b2f3db4520ac7289f9bc3b5938')]:
        result = subprocess.run(['systemctl', '--user', 'show', unit,
            '-p', 'MainPID', '-p', 'ActiveState', '-p', 'InvocationID', '-p', 'ExecMainStatus'],
            capture_output=True, text=True, check=True)
        state = dict(line.split('=', 1) for line in result.stdout.splitlines() if '=' in line)
        assert state['InvocationID'] in ('', invocation), state
        assert state['MainPID'] == '0' and state['ActiveState'] == 'inactive' and state['ExecMainStatus'] == '0', state
    running = subprocess.run(['systemctl', '--user', 'list-units', '--type=service',
        '--state=running', '--no-legend', '--no-pager', 'rl-*'], capture_output=True, text=True, check=True)
    assert all(line.split()[0] == OWN_UNIT for line in running.stdout.splitlines() if line.strip()), running.stdout
    validation_path = BASE / 'full_navigation_margin_resume_validation_20261017/RESULTS.json'
    validation = read(validation_path)
    assert validation['completed'] and validation['all_fixed28_passed']
    failures = [row for row in validation['groups'].get('hard3', {}).get('rows', [])
        if row['case_id'] == 'w11_training_stall_t106' and not row['safe_success']]
    assert len(failures) == 1, 'Only an actually failed original106 gate admits this diagnostic'
    failure = failures[0]
    assert failure['protocol_valid'] and not failure['collisions'] and not failure['false_arrival']
    neural_path = BASE / 'full_navigation_residual_20261017/NEURAL_MECHANISM.json'
    neural = read(neural_path)
    assert neural['completed'], 'The serialized input-use audit must finish first'
    model = ROOT / validation['model']
    bundle = model / 'portable'
    tasks_path = BASE / 'context_feedback_longfit_protocol_20261016/tasks.json'
    spec = read(tasks_path)['actual106']
    case = spec['case']
    source = ROOT / failure['trace_path']
    assert digest(source) == failure['trace_sha256']
    assert failure['frozen_prefix_frames'] == 530 and case['max_steps'] == 700
    paths = [Path(__file__), validation_path, neural_path, tasks_path, source, bundle / 'manifest.json']
    paths += [bundle / name for name in read(bundle / 'manifest.json')['files']]
    for folder in ('experiments', 'neural_navigation', 'navrl'):
        paths += list((ROOT / 'training' / folder).rglob('*.py'))
    paths += [path for path in (ROOT / 'training' / case['scene']).iterdir() if path.is_file()]
    hashes = {str(path.relative_to(ROOT)): digest(path) for path in set(paths)}
    write('PLAN.json', dict(inputs=hashes, source_case=spec, source_is_training_development=True,
        maximum_new_physical_steps=MAX_STEPS, maximum_beam_width=3, macro_steps=3,
        original_prefix_frames=530, original_episode_deadline_s=140.,
        snapshot_restores=2, parity_steps_each=4, complete_winner_replay_required=True,
        original12actions_4cm_body_unknown_blind_memory_dynamics_unchanged=True,
        oracle_scoring_only_in_offline_search_never_policy_input=True,
        no_automatic_training_admission=True, no_new_optimizer_updates=True, new_ppo_steps=0,
        soft_cpu_cores=.35, memory_hard_max_mib=3072, no_parallel_model_or_physics=True,
        failed_search_is_not_infeasibility_evidence=True, goal_achieved=False, deployment_approved=False))

    import ctypes
    import numpy as np
    import torch
    from experiments.probe_observed_turn_release import physical_step
    from experiments.recovery_state_snapshot import RecoverySnapshot
    from neural_navigation.aligned_release_runtime import AlignedReleaseRuntime
    from neural_navigation.experiment_budget import ExperimentBudget
    from neural_navigation.masked_actions import ACTIONS, COMMANDS, masks_from_observation
    from neural_navigation.stress_env import StressNavigationEnv
    from neural_navigation.training_distance import TrainingDistanceGraph
    torch.set_num_threads(1)
    budget = ExperimentBudget(.35, 2800, OUT)
    with np.load(source, allow_pickle=False) as archive:
        reference = {key: archive[key].copy() for key in archive.files}
    assert reference['observations'].shape[1] == 625
    env = StressNavigationEnv(case=case, components=bundle,
        scene_dir=ROOT / 'training' / case['scene'], route_mode='shortcut', max_steps=700)
    env.reset(seed=case['seed'], options=dict(start=case['start'], goal=case['goal'][:2]))
    runtime = AlignedReleaseRuntime(bundle, ROOT / 'training' / case['scene'] / 'decision_map.yaml',
        case['start'], case['goal'], 'shortcut')
    runtime._search_delegate = runtime.policy.policy

    class Selector:
        def __init__(self, choices=None):
            self.choices = choices
            self.cursor = 0
            self.requested = 0
            self.fallbacks = 0
        def predict(self, observation):
            mask = masks_from_observation(np.asarray(observation)[:282])
            choice = self.requested if self.choices is None else int(self.choices[self.cursor])
            self.cursor += 1
            if not mask[choice]:
                assert self.choices is None, 'A replayed action became inadmissible'
                choice = 0
                self.fallbacks += 1
            return ACTIONS[choice].copy()

    runtime.policy.policy = Selector(reference['choices'][:530])
    report = dict(completed=False, rows=[], new_physical_steps=0, expanded_nodes=0,
        rejected_unsafe_branches=0, new_ppo_steps=0, new_optimizer_updates=0,
        teacher_search_not_a_learned_model=True, goal_achieved=False, deployment_approved=False)
    libc = ctypes.CDLL(None)

    def cleanup():
        gc.collect()
        libc.malloc_trim(0)
        budget.tick()

    def step(e, r):
        assert report['new_physical_steps'] < MAX_STEPS, 'Declared search budget exhausted'
        result, packet = physical_step(e, r, budget)
        report['new_physical_steps'] += 1
        return result, packet

    def unsafe(e):
        return bool(e.collision_events or e.hard_guard_activations or e.minimum_gap < .04 - 1e-4)

    def finish(reason, success=False):
        for name, expected in hashes.items():
            assert digest(ROOT / name) == expected, name
        assert report['new_physical_steps'] <= MAX_STEPS
        report.update(completed=True, safe_teacher_found=success, reason=reason,
            resource=budget.statistics(), verified_inputs=len(hashes),
            no_infeasibility_or_learned_success_claim=True)
        write('RESULTS.json', report)
        write('STATUS.json', dict(stage='complete', reason=reason, safe_teacher_found=success))

    # Keep a waiting lineage and two physically distinct progress lineages.
    def retain(nodes):
        ordered = sorted(nodes, key=lambda node: node['score'], reverse=True)
        waiting = next((node for node in ordered if node['waiting']), None)
        selected, signatures = [], set()
        for node in ordered:
            if node is waiting:
                continue
            if node['signature'] not in signatures:
                selected.append(node)
                signatures.add(node['signature'])
            if len(selected) == (2 if waiting is not None else 3):
                break
        if waiting is not None:
            selected.append(waiting)
        return selected

    try:
        for index in range(530):
            _, packet = step(env, runtime)
            for name in ('observations', 'masks', 'choices', 'commands', 'scans', 'odometry', 'measured_velocity'):
                np.testing.assert_array_equal(packet[name], reference[name][index])
            np.testing.assert_allclose(packet['labels'][:7], reference['labels'][index, :7], rtol=0, atol=2e-8)
            if index % 50 == 0:
                write('STATUS.json', dict(stage='exact_current625_initialization', frame=index + 1, target=530))
        assert not unsafe(env)
        runtime.policy.policy = Selector()
        root = RecoverySnapshot(env, runtime)
        samples = []
        for repeat in range(2):
            e, r = root.restore()
            current = []
            try:
                for _ in range(4):
                    _, packet = step(e, r)
                    current.append(packet)
                if repeat:
                    for a, b in zip(samples, current):
                        for name in a:
                            np.testing.assert_array_equal(a[name], b[name])
                else:
                    samples = current
            finally:
                e.close()
                del e, r, current
                cleanup()
        report['two_original625_sensor_physics_snapshot_restores_exact'] = True
        field = TrainingDistanceGraph(env.static_map).field(case['goal'][:2])
        beam = [dict(snapshot=root, choices=[], score=0., signature=(), waiting=True, packets=[])]
        winner = None
        for depth in range(57):
            children = []
            for parent in beam:
                if len(parent['choices']) >= 170:
                    continue
                # A zero-command probe computes the unchanged current input and
                # mask. It is also a genuine waiting branch, never a mask bypass.
                e, r = parent['snapshot'].restore()
                try:
                    _, first = step(e, r)
                finally:
                    e.close()
                    del e, r
                    cleanup()
                mask = first['masks']
                motions = np.flatnonzero(mask[1:]) + 1
                if len(motions) <= 5:
                    candidates = [0, *motions.tolist()]
                else:
                    # Preserve both rotation signs and the most useful existing
                    # forward choices; no new velocity or reverse action exists.
                    waypoint = first['observations'][216:218] * 2.
                    target_yaw = float(np.arctan2(waypoint[1], waypoint[0]))
                    translating = [int(i) for i in motions if COMMANDS[i, 0] > 0.]
                    translating.sort(key=lambda i: abs(target_yaw - COMMANDS[i, 1] * .8) - COMMANDS[i, 0])
                    candidates = [0, *[i for i in (4, 5) if mask[i]], *translating[:3]]
                for requested in candidates:
                    if report['new_physical_steps'] + 175 >= MAX_STEPS:
                        finish('Bounded search budget reached; unresolved, not proved infeasible')
                        return
                    e, r = parent['snapshot'].restore()
                    r.policy.policy.requested = requested
                    packets = []
                    arrived = bad = False
                    try:
                        for offset in range(min(3, 170 - len(parent['choices']))):
                            result, packet = step(e, r)
                            packets.append(packet)
                            if offset == 0:
                                for name in ('observations', 'masks', 'scans', 'odometry', 'measured_velocity'):
                                    np.testing.assert_array_equal(packet[name], first[name])
                            if result['stop_reason'] == 'estimated_goal_reached':
                                arrived = bool(packet['labels'][7] < .2)
                                bad = not arrived
                                break
                            if unsafe(e):
                                bad = True
                                break
                        report['expanded_nodes'] += 1
                        if bad or unsafe(e):
                            report['rejected_unsafe_branches'] += 1
                            continue
                        choices = parent['choices'] + [int(packet['choices']) for packet in packets]
                        history = parent['packets'] + packets
                        distance = field.value(e.truth[:2])
                        yaw = float(e.truth[2])
                        lookahead = e.truth[:2] + .15 * np.array([np.cos(yaw), np.sin(yaw)])
                        direction_gain = distance - field.value(lookahead)
                        legal = int(packets[-1]['masks'][1:].sum())
                        score = -distance + .3 * direction_gain + .002 * legal - .003 * len(choices) - .002 * e.travel
                        signature = (*np.round(e.truth[:2] / .03).astype(int).tolist(), int(round(yaw / .12)))
                        child = dict(snapshot=RecoverySnapshot(e, r), choices=choices, score=float(score),
                            signature=signature, waiting=parent['waiting'] and all(i == 0 for i in choices),
                            packets=history)
                        if arrived:
                            winner = child
                            break
                        children = retain(children + [child])
                        del child
                    finally:
                        e.close()
                        del e, r
                        cleanup()
                if winner is not None:
                    break
            if winner is not None:
                break
            if not children:
                finish('No surviving bounded branch; unresolved, not proved infeasible')
                return
            beam = retain(children)
            write('STATUS.json', dict(stage='bounded_original_mask_search', depth=depth + 1,
                decision_frames=len(beam[0]['choices']), new_physical_steps=report['new_physical_steps'],
                beam=[dict(score=n['score'], signature=n['signature'], tail=n['choices'][-12:],
                    waiting_lineage=n['waiting']) for n in beam], resource=budget.statistics()))
            del children, parent
            cleanup()
        if winner is None:
            finish('All bounded legal branches timed out; unresolved, not proved infeasible')
            return
        e, r = root.restore()
        r.policy.policy = Selector(winner['choices'])
        replay = []
        try:
            for expected in winner['packets']:
                result, packet = step(e, r)
                for name in expected:
                    np.testing.assert_array_equal(packet[name], expected[name])
                replay.append(packet)
                assert not unsafe(e)
            assert result['stop_reason'] == 'estimated_goal_reached' and replay[-1]['labels'][7] < .2
            path = OUT / 'safe_teacher_suffix.npz'
            np.savez_compressed(path, **{name: np.asarray([p[name] for p in replay]) for name in replay[0]})
            report['rows'].append(dict(safe_success=True, complete_suffix_replay_exact=True,
                release_sensor_timestamp_s=106., post_release_seconds=float(e.elapsed - 106.),
                original_deadline_s=140., collisions=e.collision_events, physical_guards=e.hard_guard_activations,
                minimum_dynamic_gap_m=float(e.minimum_gap), final_xy_error_m=float(replay[-1]['labels'][7]),
                suffix_frames=len(replay), trace_path=str(path.relative_to(ROOT)), trace_sha256=digest(path),
                choices=winner['choices'], offline_teacher_not_model_success=True, automatic_training_admission=False))
        finally:
            e.close()
            del e, r
            cleanup()
        finish('Complete originally legal teacher and exact replay found; separate admission required', True)
    finally:
        env.close()


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        if OUT.exists():
            write('FAILURE.json', dict(error=repr(error), no_auto_restart=True,
                goal_achieved=False, deployment_approved=False))
        raise
