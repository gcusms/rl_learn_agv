"""Search original legal recovery at single-frame control resolution.

No runtime, model, observation, action, memory, margin or deadline changes.
Offline truth can rank branches and assess arrival, never enter actor input.
Only a complete safe winner independently replayed can become a data candidate.
"""
from datetime import datetime
import gc
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
BASE = HERE.parent
COMPACT_HELPER = BASE / 'late106_compact_beam_protocol_20261017'
sys.path.insert(0, str(COMPACT_HELPER))
OUT = BASE / 'late106_framewise_20261017'
OWN_UNIT = 'rl-late106-framewise-20261017.service'
MAX_STEPS = 6500


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


def preconditions():
    assert not OUT.exists()
    for unit, invocation in [
        ('rl-late106-creep-teacher-20261017.service', '8330023aad7447e39dda12707067ce2a'),
        ('rl-late106-creep-geometry-audit-20261017.service', 'bcf545e28f594f1eb712709f67a2bc1e')]:
        result = subprocess.run(['systemctl', '--user', 'show', unit, '-p', 'MainPID',
            '-p', 'ActiveState', '-p', 'InvocationID', '-p', 'ExecMainStatus'],
            capture_output=True, text=True, check=True)
        state = dict(line.split('=', 1) for line in result.stdout.splitlines() if '=' in line)
        assert state['InvocationID'] in ('', invocation), state
        assert state['MainPID'] == '0' and state['ActiveState'] == 'inactive'
        assert state['ExecMainStatus'] == '0'
    previous = read(BASE / 'late106_creep_teacher_20261017/RESULTS.json')
    assert previous['completed'] and not previous['safe_teacher_found']
    audit = read(BASE / 'positive_free_failure_audit_20261017/RESULTS.json')
    assert audit['completed'] and audit['geometry_checks_matched'] == audit['geometry_checks_tested']
    fixture = read(COMPACT_HELPER / 'COMPACT_FIXTURE_CHECK.json')
    assert fixture['completed'] and fixture['two_native_contact_continuations_exact']
    running = subprocess.run(['systemctl', '--user', 'list-units', '--type=service',
        '--state=running', '--no-legend', '--no-pager', 'rl-*'], capture_output=True, text=True, check=True)
    assert all(line.split()[0] == OWN_UNIT for line in running.stdout.splitlines() if line.strip())


def main():
    preconditions()
    original_search_source = BASE / 'late106_compact_beam_protocol_20261017/run.py'
    validation_path = BASE / 'full_navigation_joint_linear_validation_20261017/RESULTS.json'
    validation = read(validation_path)
    assert validation['completed'] and validation['all_fixed28_passed']
    failure = next(row for row in validation['groups']['hard3']['rows'] if row['case_id'] == 'w11_training_stall_t106')
    assert failure['timeout'] and not failure['collisions'] and not failure['physical_guards']
    source_path = ROOT / failure['trace_path']
    assert digest(source_path) == failure['trace_sha256']
    tasks_path = BASE / 'context_feedback_longfit_protocol_20261016/tasks.json'
    spec = read(tasks_path)['actual106']
    bundle = ROOT / validation['model'] / 'portable'
    paths = [Path(__file__), original_search_source,
        BASE / 'positive_free_failure_audit_20261017/RESULTS.json',
        HERE / 'OPPORTUNITY_WINDOWS.json', HERE / 'replay_when_ready.py', COMPACT_HELPER / 'compact_state.py', COMPACT_HELPER / 'COMPACT_FIXTURE_CHECK.json',
        validation_path, source_path, tasks_path,
        BASE / 'late106_creep_teacher_20261017/RESULTS.json',
        BASE / 'late106_creep_analysis_20261017/RESULTS.json',
        BASE / 'late106_creep_protocol_20261017/worker.py', bundle / 'manifest.json']
    paths += [bundle / name for name in read(bundle / 'manifest.json')['files']]
    for package in ('experiments', 'neural_navigation', 'navrl'):
        paths += list((ROOT / 'training' / package).rglob('*.py'))
    paths += [p for p in (ROOT / 'training' / spec['case']['scene']).iterdir() if p.is_file()]
    hashes = {str(path.relative_to(ROOT)): digest(path) for path in set(paths)}
    OUT.mkdir()
    write('PLAN.json', dict(inputs=hashes, case=spec['case'], original_prefix_frames=530,
        original_deadline_s=140., frontier_width=5, maximum_macro_frames=1,
        maximum_search_physical_steps=MAX_STEPS, maximum_fresh_winner_replay_steps=700,
        maximum_total_physical_steps=7200, one_native_model_and_data=True,
        mandatory_two_exact_native625_sensor_physics_continuation_checks=True,
        original_12actions_footprint4cm_unknown_blind_memory_dynamics_unchanged=True,
        offline_truth_ranking_never_actor_input=True, no_automatic_training=True,
        retain_clockwise_counterclockwise_wait_goal_and_mobility_lineages=True,
        command_histories_not_coarse_pose_buckets_for_frontier_deduplication=True,
        independent_infeasibility_not_claimed_by_bounded_search=True,
        soft_cpu_cores=.35, hard_memory_mib=3072))

    import numpy as np
    import torch
    from compact_state import CompactSnapshot, model_digest
    from experiments.probe_observed_turn_release import physical_step
    from neural_navigation.aligned_release_runtime import AlignedReleaseRuntime
    from neural_navigation.experiment_budget import ExperimentBudget
    from neural_navigation.masked_actions import ACTIONS, COMMANDS, masks_from_observation
    from neural_navigation.stress_env import StressNavigationEnv
    from neural_navigation.training_distance import TrainingDistanceGraph
    from navrl.geometry import wrap

    torch.set_num_threads(1)
    budget = ExperimentBudget(.35, 2800, OUT)
    case = spec['case']
    env = StressNavigationEnv(case=case, components=bundle,
        scene_dir=ROOT / 'training' / case['scene'], route_mode='shortcut', max_steps=700)
    env.reset(seed=case['seed'], options=dict(start=case['start'], goal=case['goal'][:2]))
    runtime = AlignedReleaseRuntime(bundle, ROOT / 'training' / case['scene'] / 'decision_map.yaml',
        case['start'], case['goal'], 'shortcut')
    native_data = env.data
    original_native_hash = model_digest(env.model)
    with np.load(source_path, allow_pickle=False) as saved:
        reference = {key: saved[key].copy() for key in saved.files}

    class Selector:
        def __init__(self, sequence=None):
            self.sequence = sequence
            self.cursor = 0
            self.requested = 0
        def predict(self, observation):
            mask = masks_from_observation(np.asarray(observation)[:282])
            requested = int(self.sequence[self.cursor]) if self.sequence is not None else self.requested
            self.cursor += 1
            choice = requested if mask[requested] else 0
            if self.sequence is not None:
                assert choice == requested
            return ACTIONS[choice].copy()

    runtime.policy.policy = Selector(reference['choices'][:530])
    report = dict(completed=False, model=validation['model'], new_physical_steps=0,
        expanded_nodes=0, rejected_unsafe_branches=0, rows=[])
    prefix = []

    def step(e, r, initialization=False):
        assert report['new_physical_steps'] < MAX_STEPS
        result, packet = physical_step(e, r, budget)
        report['new_physical_steps'] += 1
        packet['current_action_reason_strings'] = np.asarray(
            [reason or 'none' for reason in result['current_action_rejection_reasons']], dtype='U64')
        packet['executed_stop_reason'] = np.asarray(result['decision_feedback_executed_reason'], dtype='U64')
        packet['initialization_prefix'] = np.asarray(initialization)
        packet['expert_continuation'] = np.asarray(not initialization)
        packet['model_controlled'] = np.asarray(False)
        return result, packet

    def unsafe(e):
        return bool(e.collision_events or e.hard_guard_activations or e.minimum_gap < .04 - 1e-4)

    def retain(nodes):
        # Direction coverage survives temporary increases in goal distance.
        # Do not collapse sub-centimeter states into the old 3cm pose buckets.
        if not nodes:
            return []
        ordered = sorted(nodes, key=lambda node: node['score'], reverse=True)
        selected, histories = [], set()
        def add(node):
            if node is not None and tuple(node['choices']) not in histories:
                selected.append(node)
                histories.add(tuple(node['choices']))
        add(next((node for node in ordered if node['waiting']), None))
        add(max(nodes, key=lambda node: (-node['turn'], node['score'])))
        add(max(nodes, key=lambda node: (node['turn'], node['score'])))
        add(ordered[0])
        add(max(nodes, key=lambda node: (node['legal'], node['score'])))
        for node in ordered:
            if len(selected) >= 5:
                break
            add(node)
        assert len(selected) <= 5
        return selected

    def finish(reason, winner=None):
        assert model_digest(env.model) == original_native_hash
        for path, sha in hashes.items():
            assert digest(ROOT / path) == sha, path
        report.update(completed=True, safe_teacher_found=winner is not None, reason=reason,
            native_model_unchanged=True, resource=budget.statistics(),
            no_learned_success_or_infeasibility_claim=True, no_automatic_training_admission=True)
        write('SEARCH_RESULTS.json', report)
        return winner

    winner = None
    try:
        for index in range(530):
            _, packet = step(env, runtime, True)
            prefix.append(packet)
            for name in ('observations', 'masks', 'choices', 'commands', 'scans', 'odometry', 'measured_velocity'):
                np.testing.assert_array_equal(packet[name], reference[name][index])
            np.testing.assert_allclose(packet['labels'][:7], reference['labels'][index, :7], rtol=0, atol=2e-8)
            if index % 80 == 0:
                write('STATUS.json', dict(stage='exact_original625_prefix', frame=index + 1))
        assert not unsafe(env)
        runtime.policy.policy = Selector()
        root = CompactSnapshot(env, runtime)
        parity_rows = []
        for repeat in range(2):
            e, r = root.restore(native_data)
            packets = []
            for index in range(530, 534):
                _, packet = step(e, r)
                for name in ('observations', 'masks', 'choices', 'commands', 'scans', 'odometry', 'measured_velocity'):
                    np.testing.assert_array_equal(packet[name], reference[name][index])
                np.testing.assert_allclose(packet['labels'][:7], reference['labels'][index, :7], rtol=0, atol=2e-8)
                packets.append(packet)
            if repeat:
                for a, b in zip(parity_rows[0], packets):
                    for name in a:
                        np.testing.assert_array_equal(a[name], b[name])
            parity_rows.append(packets)
            del e, r
            gc.collect()
        report['compact_state_two_complete_original_continuations_exact'] = True
        write('COMPACT_STATE_ADMISSION.json', dict(completed=True, original_prefix_frames=530,
            exact_original625_sensor_physics_frames_per_restore=4, exact_repeats=2,
            native_model_nbuffer_bytes=int(env.model.nbuffer), native_data_nbuffer_bytes=int(native_data.nbuffer),
            native_integration_state_values=len(root.state), native_model_unchanged=True,
            resource=budget.statistics(), no_runtime_or_safety_changes=True))
        del parity_rows, packets
        field = TrainingDistanceGraph(env.static_map).field(case['goal'][:2])
        restored_root_env, restored_root_runtime = root.restore(native_data)
        root_yaw = float(restored_root_env.truth[2])
        del restored_root_env, restored_root_runtime
        beam = [dict(snapshot=root, choices=[], packets=[], waiting=True, score=0., signature=(),
                     turn=0., yaw=root_yaw, legal=0)]
        exhausted = False
        for depth in range(170):
            children = []
            for parent in beam:
                if len(parent['choices']) >= 170:
                    continue
                e, r = parent['snapshot'].restore(native_data)
                _, first = step(e, r)
                motions = (np.flatnonzero(first['masks'][1:]) + 1).tolist()
                candidates = [0, *motions]
                del e, r
                for requested in candidates:
                    if report['new_physical_steps'] + 1 >= MAX_STEPS:
                        exhausted = True
                        break
                    e, r = parent['snapshot'].restore(native_data)
                    r.policy.policy.requested = requested
                    packets = []
                    arrived = bad = False
                    for offset in range(min(1, 170 - len(parent['choices']))):
                        result, packet = step(e, r)
                        packets.append(packet)
                        if offset == 0:
                            for name in ('observations', 'masks', 'scans', 'odometry', 'measured_velocity'):
                                np.testing.assert_array_equal(packet[name], first[name])
                        if result['stop_reason'] == 'estimated_goal_reached':
                            arrived = bool(packet['labels'][7] < .2 and not unsafe(e))
                            bad = not arrived
                            break
                        if unsafe(e):
                            bad = True
                            break
                    report['expanded_nodes'] += 1
                    if bad or unsafe(e):
                        report['rejected_unsafe_branches'] += 1
                        del e, r
                        continue
                    choices = parent['choices'] + [int(p['choices']) for p in packets]
                    distance = field.value(e.truth[:2])
                    yaw = float(e.truth[2])
                    turn = parent['turn'] + float(wrap(yaw - parent['yaw']))
                    lookahead = e.truth[:2] + .15 * np.array([np.cos(yaw), np.sin(yaw)])
                    direction_gain = distance - field.value(lookahead)
                    legal = int(packets[-1]['masks'][1:].sum())
                    score = -distance + .3 * direction_gain + .002 * legal - .003 * len(choices) - .002 * e.travel
                    signature = tuple(choices)
                    child = dict(snapshot=CompactSnapshot(e, r), choices=choices,
                        packets=parent['packets'] + packets, score=float(score), signature=signature,
                        turn=turn, yaw=yaw, legal=legal,
                        waiting=parent['waiting'] and all(i == 0 for i in choices))
                    del e, r
                    if arrived:
                        winner = child
                        break
                    children = retain(children + [child])
                    del child
                    gc.collect()
                    budget.tick()
                if winner is not None or exhausted:
                    break
            if winner is not None or exhausted or not children:
                break
            beam = retain(children)
            write('STATUS.json', dict(stage='bounded_original_mask_search', depth=depth + 1,
                decision_frames=len(beam[0]['choices']), new_physical_steps=report['new_physical_steps'],
                frontier=[dict(score=node['score'], history_frames=len(node['choices']), tail=node['choices'][-12:],
                    waiting=node['waiting'], cumulative_turn_rad=node['turn'],
                    pre_execution_legal_motion_actions=node['legal']) for node in beam], resource=budget.statistics()))
            del children, parent
            gc.collect()
            budget.tick()
        if winner is None:
            finish('Bounded originally legal search unresolved; not infeasibility')
        else:
            packets = prefix + winner['packets']
            trace = OUT / 'safe_teacher_full_trace.npz'
            np.savez_compressed(trace, **{name: np.asarray([p[name] for p in packets]) for name in packets[0]})
            report['rows'].append(dict(safe_success=True, trace_path=str(trace.relative_to(ROOT)),
                trace_sha256=digest(trace), frames=len(packets), choices=winner['choices'],
                original_deadline_s=140., offline_teacher_not_model_success=True))
            finish('Complete safe candidate found; independent fresh replay required', winner)
    finally:
        env.close()
    if winner is not None:
        # Release all native/neural objects before the independent fresh process.
        del env, runtime, native_data, root, beam, winner
        gc.collect()
        job = dict(mode='replay', target_rad=None, case=case, bundle=str(bundle.relative_to(ROOT)),
            source_trace=str(source_path.relative_to(ROOT)), inputs=hashes,
            expected_trace=report['rows'][0]['trace_path'], expected_sha256=report['rows'][0]['trace_sha256'],
            directory=str((OUT / 'winner_replay').relative_to(ROOT)))
        path = OUT / 'winner_replay.json'
        path.write_text(json.dumps(job, ensure_ascii=False, indent=2) + '\n')
        write('REPLAY_JOB_READY.json', dict(job=str(path.relative_to(ROOT)),
            command=[sys.executable, '-u', str(BASE / 'late106_creep_protocol_20261017/worker.py'), str(path)],
            launch_only_after_this_heavy_process_terminal=True, maximum_new_physical_steps=700))
    write('RESULTS.json', dict(completed=True, safe_teacher_candidate_found=bool(report['rows']),
        independent_fresh_replay_required=bool(report['rows']), independent_fresh_replay_passed=False,
        search_result='SEARCH_RESULTS.json', new_physical_steps=report['new_physical_steps'],
        no_automatic_training_admission=True, no_learned_success_or_infeasibility_claim=True))
    write('STATUS.json', dict(stage='complete', safe_teacher_candidate_found=bool(report['rows'])))


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        if OUT.exists():
            write('FAILURE.json', dict(error=repr(error), no_auto_restart=True))
        raise
