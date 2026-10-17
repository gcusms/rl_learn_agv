"""Resume the second complete PPO batch in a fresh bounded process.

All learned experience is on-policy. The offline teacher is evidence about
feasibility, never an automatically admitted action label or runtime controller.
"""
from datetime import datetime
import ctypes
import gc
import hashlib
import json
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT/'reports/stop_feedback_20261015'
HERE = BASE/'guard_forecast_alignment_protocol_20261017'
OUT = BASE/'guard_aligned_mixed_ppo_resume_20261017'
MODELS = ROOT/'models/stop_feedback_20261015/guard_aligned_mixed_ppo_resume_s231017'
PARENT = ROOT/'models/stop_feedback_20261015/guard_aligned_migration_s231017'
DEPENDENCY = 'rl-late106-all-action-20261017.service'
INVOCATION = 'c0703109bceb409d9a3e616a2f0611c1'
BATCHES, RELEASES = 2, (450, 480, 488, 530)


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')
    temporary.replace(path)


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    previous_out = BASE/'guard_aligned_mixed_ppo_20261017'
    previous = read(previous_out/'RESULTS.json')
    failure = read(previous_out/'FAILURE.json')
    resume_source = ROOT/'models/stop_feedback_20261015/guard_aligned_mixed_ppo_s231017/batch_01.pt'
    assert not previous['completed'] and len(previous['episodes']) == 5 and len(previous['batches']) == 1
    assert previous['new_ppo_steps'] == 1065 and previous['initialization_physical_steps'] == 598
    assert 'conservative RSS budget' in failure['error']
    assert digest(resume_source) == '0c24b61038f58f4927a5883d76d1d3899fa29af583aaf95933104533c6b401ab'
    for unit in ['rl-guard-aligned-mixed-ppo-20261017.service', 'rl-guard-aligned-mixed-ppo-validate-20261017.service']:
        checked = subprocess.run(['systemctl', '--user', 'show', unit, '--property=MainPID'], capture_output=True, text=True, check=True)
        assert checked.stdout.strip() == 'MainPID=0', 'Previous heavy process is still live'
    for name, expected in read(previous_out/'DEPENDENCIES.json')['inputs'].items():
        assert digest(ROOT/name) == expected, name
    for row in previous['episodes']:
        assert digest(ROOT/row['rollout_path']) == row['rollout_sha256']
    task_path = BASE/'context_feedback_longfit_protocol_20261016/tasks.json'
    tasks = read(task_path)
    case = tasks['actual106']['case']
    source = ROOT/tasks['actual106']['trace']
    anchors_path = HERE/'aligned_demonstrations.npz'
    kernel_dir = BASE/'native625_learning_diagnostic_20261017'
    manifest = read(PARENT/'portable/manifest.json')
    paths = [Path(__file__), resume_source, previous_out/'RESULTS.json', previous_out/'FAILURE.json',
        previous_out/'PLAN.json', previous_out/'DEPENDENCIES.json', HERE/'train_ppo.py',
        *[ROOT/row['rollout_path'] for row in previous['episodes']],
        task_path, source, anchors_path, PARENT/'checkpoint.pt', PARENT/'parent_actor.pt',
        PARENT/'RESULTS.json', PARENT/'portable/manifest.json',
        BASE/'guard_forecast_migration_validation_20261017/PLAN.json',
        BASE/'late106_all_action_20261017/PLAN.json',
        kernel_dir/'batch_credit_ppo.py', kernel_dir/'retained_stop_ppo.py',
        kernel_dir/'CONSTRAINT_AND_CREDIT_TESTS.json']
    paths += [PARENT/'portable'/name for name in manifest['files']]
    for folder in ['experiments', 'neural_navigation', 'navrl']:
        paths += list((ROOT/'training'/folder).rglob('*.py'))
    paths += [p for p in (ROOT/'training'/case['scene']).iterdir() if p.is_file()]
    hashes = {str(p.relative_to(ROOT)): digest(p) for p in set(paths)}
    write(OUT/'PLAN.json', dict(at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(), inputs=hashes,
        dependency=DEPENDENCY, expected_invocation=INVOCATION, collection_batches=BATCHES,
        recovery_release_frames=list(RELEASES), full_navigation_cases=['motion_profile_015'],
        resume_exact_complete_batch01=True, source_complete_decisions=1065, batches_this_process=1,
        maximum_additional_complete_decisions=1552, maximum_additional_initialization_steps=598,
        source_incomplete_fragment_excluded_from_optimizer=True, unpersisted_fragment_at_least_one_decision=True,
        source_unpersisted_fragment_exact_length_unavailable=True, allocator_trim_between_episodes=True,
        maximum_ppo_decisions=3104, maximum_initialization_physical_steps=1196,
        maximum_complete_training_episodes=10, one_common450_prefix_snapshot=True,
        one_policy_and_critic_frozen_during_each_complete_five_episode_batch=True,
        gamma=.99, gae_lambda=1., sampling_temperature=4., learning_rate=.0003,
        epochs=4, minibatch_size=64, anchor_kl_strength=.05,
        all557_reference_stops_checked_after_each_proposed_actor_update=True,
        new_v2_inputs_and_matching_runtime=True, original_hardware_memory_config=True,
        no_teacher_trajectory_labels=True, no_failed_prefix_positive_labels=True,
        initialization_commands_not_ppo_experience=True, private_sampling_and_minibatch_rngs=True,
        original_rewards_actions_masks_4cm_body_dynamics_and_deadlines=True,
        simulator_truth_only_rewards_and_evaluation=True, no_minimum_forward_speed_or_forced_motion=True,
        soft_cpu_cores=.35, memory_hard_max_mib=3072, maximum_model_output_mib=40,
        no_parallel_physical_job=True, goal_achieved=False, deployment_approved=False))
    while True:
        checked = subprocess.run(['systemctl', '--user', 'show', DEPENDENCY,
            '--property=ActiveState,MainPID,InvocationID,ExecMainStatus'], capture_output=True, text=True, check=True)
        state = dict(line.split('=', 1) for line in checked.stdout.splitlines() if '=' in line)
        if state['InvocationID'] not in ('', INVOCATION):
            raise RuntimeError('The exact diagnostic dependency changed invocation')
        if state['MainPID'] == '0':
            teacher_path = BASE/'late106_all_action_20261017/RESULTS.json'
            validation_path = BASE/'guard_forecast_migration_validation_20261017/RESULTS.json'
            if (state['ActiveState'] != 'inactive' or state['ExecMainStatus'] != '0'
                    or not teacher_path.exists() or not read(teacher_path)['completed']):
                raise RuntimeError('Exact teacher dependency did not finish normally; no PPO started')
            validation, teacher = read(validation_path), read(teacher_path)
            hashes.update({str(p.relative_to(ROOT)): digest(p) for p in [teacher_path, validation_path]})
            reason = None
            if not validation.get('all_fixed_retention_passed'):
                reason = 'V2 parent full28 preservation failed; no PPO started'
            original = any(row['safe_success'] and row['hardware'] == 'original_memory' for row in teacher['rows'])
            alternative = any(row['safe_success'] and row['hardware'] == 'three_frame_memory' for row in teacher['rows'])
            if alternative and not original:
                reason = 'Only alternative memory has a verified teacher path; require matching memory/data replay before training'
            if reason:
                write(OUT/'RESULTS.json', dict(completed=True, training_started=False, reason=reason,
                    new_ppo_steps=0, goal_achieved=False, deployment_approved=False))
                write(OUT/'STATUS.json', dict(status='not_started_prerequisite', reason=reason))
                return
            break
        write(OUT/'STATUS.json', dict(status='waiting_for_exact_teacher_and_full28', dependency=state,
            neural_and_physics_libraries_not_loaded=True, new_ppo_steps=0))
        time.sleep(5.)
    for name, expected in hashes.items():
        assert digest(ROOT/name) == expected, name
    assert read(kernel_dir/'CONSTRAINT_AND_CREDIT_TESTS.json')['passed']
    write(OUT/'DEPENDENCIES.json', dict(inputs=hashes, parent_full28_passed=True,
        teachers_not_admitted_to_training=True, original_memory_kept=True))

    import numpy as np
    import torch
    sys.path.insert(0, str(kernel_dir))
    from batch_credit_ppo import BatchCreditOnPolicy, prepare_completed_batch
    from experiments.guard_reason_release_actor import GuardReasonReleaseActor, export
    from experiments.recovery_state_snapshot import RecoverySnapshot, tensor_digest
    from experiments.train_native625_recovery import collect_episode, physical_step
    from navrl.map import StaticMap
    from neural_navigation.aligned_release_runtime import AlignedReleaseRuntime, RUNTIME_NAME, contract
    from neural_navigation.aligned_risk_observation import contract as risk_contract
    from neural_navigation.experiment_budget import ExperimentBudget
    from neural_navigation.goal_pose_runtime import check_files
    from neural_navigation.masked_actions import ACTIONS, masks_from_observation
    from neural_navigation.stress_env import StressNavigationEnv
    from neural_navigation.training_distance import TrainingDistanceGraph

    MODELS.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.manual_seed(231017)
    np.random.seed(231017)
    random.seed(231017)
    budget = ExperimentBudget(.35, memory_mib=2800, directory=OUT)
    with np.load(anchors_path, allow_pickle=False) as stored:
        anchors, labels = stored['observations'].copy(), stored['actions'].copy()
    with np.load(source, allow_pickle=False) as stored:
        reference = {key: stored[key].copy() for key in stored.files}
    actor = GuardReasonReleaseActor(torch.jit.load(str(PARENT/'parent_actor.pt'), map_location='cpu')).eval()
    resumed = torch.load(resume_source, map_location='cpu', weights_only=False)
    assert resumed['ppo_decisions'] == 1065 and resumed['contract'] == contract()
    actor.load_state_dict(resumed['actor'])
    with torch.no_grad():
        initial_matches = actor(torch.from_numpy(anchors)).numpy() == ACTIONS[labels]
        assert initial_matches[labels == 0].all()
    learner = BatchCreditOnPolicy(actor, anchors[labels == 0], learning_rate=.0003, temperature=4.)
    generator = torch.Generator().manual_seed(231017)
    minibatches = torch.Generator().manual_seed(231018)
    learner.critic.load_state_dict(resumed['critic'])
    learner.actor_optimizer.load_state_dict(resumed['actor_optimizer'])
    learner.critic_optimizer.load_state_dict(resumed['critic_optimizer'])
    learner.reference_head.load_state_dict(resumed['frozen_reference_head'])
    generator.set_state(resumed['policy_rng'])
    minibatches.set_state(resumed['minibatch_rng'])
    torch.set_rng_state(resumed['torch_rng'])
    np.random.set_state(resumed['numpy_rng'])
    random.setstate(resumed['python_rng'])

    def assert_restored(actual, expected):
        if isinstance(expected, torch.Tensor):
            torch.testing.assert_close(actual, expected, rtol=0., atol=0.)
        elif isinstance(expected, np.ndarray):
            np.testing.assert_array_equal(actual, expected)
        elif isinstance(expected, dict):
            assert actual.keys() == expected.keys()
            for key in expected:
                assert_restored(actual[key], expected[key])
        elif isinstance(expected, (list, tuple)):
            assert len(actual) == len(expected)
            for value, reference_value in zip(actual, expected):
                assert_restored(value, reference_value)
        else:
            assert actual == expected

    for actual, expected in [(actor.state_dict(), resumed['actor']),
            (learner.critic.state_dict(), resumed['critic']),
            (learner.actor_optimizer.state_dict(), resumed['actor_optimizer']),
            (learner.critic_optimizer.state_dict(), resumed['critic_optimizer']),
            (learner.reference_head.state_dict(), resumed['frozen_reference_head']),
            (generator.get_state(), resumed['policy_rng']), (minibatches.get_state(), resumed['minibatch_rng']),
            (torch.get_rng_state(), resumed['torch_rng']), (np.random.get_state(), resumed['numpy_rng']),
            (random.getstate(), resumed['python_rng'])]:
        assert_restored(actual, expected)
    assert learner.stop_mismatches() == 0
    write(OUT/'RESUME_CHECKS.json', dict(completed=True, exact_actor_critic_optimizer_reference_and_rng_restore=True,
        all557_stop_labels_retained=True, checkpoint_sha256=digest(resume_source),
        source_complete_decisions=1065, incomplete_fragment_excluded=True, goal_achieved=False))
    trim = getattr(ctypes.CDLL(None), 'malloc_trim', None)

    def trim_allocator():
        if trim is not None:
            trim(0)
    parent_digest = tensor_digest(actor.parent)
    pools = [np.flatnonzero(labels == 0), np.flatnonzero(labels != 0)]
    graph = TrainingDistanceGraph(StaticMap(ROOT/'training'/case['scene']/'decision_map.yaml', radius=.3))
    field = graph.field(case['goal'])
    report = read(previous_out/'RESULTS.json')
    report.update(completed=False, training_started=True, source_complete_decisions=1065,
        resume_checkpoint=str(resume_source.relative_to(ROOT)), resume_checkpoint_sha256=digest(resume_source),
        source_failed_attempt=str(previous_out.relative_to(ROOT)),
        source_attempt_resource=previous['resource'],
        unpersisted_failed_fragment_excluded=True, unpersisted_fragment_at_least_one_decision=True,
        unpersisted_fragment_exact_length_unavailable=True, current_process_resume_not_physical_snapshot_resume=True,
        counters_cover_complete_admitted_episodes_only=True,
        goal_achieved=False, deployment_approved=False)


    def save():
        report.update(new_ppo_steps=report['ppo_decisions'],
            additional_complete_decisions_this_process=report['ppo_decisions']-1065,
            additional_actor_updates_this_process=report['new_actor_optimizer_updates']-previous['new_actor_optimizer_updates'],
            additional_critic_updates_this_process=report['new_critic_optimizer_updates']-previous['new_critic_optimizer_updates'],
            additional_initialization_steps_this_process=report['initialization_physical_steps']-598,
            resource=budget.statistics())
        write(OUT/'RESULTS.json', report)

    class StrictPrefix:
        def __init__(self):
            self.cursor = 0

        def predict(self, observation):
            choice = int(reference['choices'][self.cursor])
            if not masks_from_observation(np.asarray(observation)[:282])[choice]:
                raise ValueError('Recovery initialization command became inadmissible')
            self.cursor += 1
            return ACTIONS[choice].copy()

    def initialize(env, runtime, index):
        for key, actual in [('scans', env.scan), ('odometry', env.observed_odometry), ('measured_velocity', env.measured_velocity)]:
            np.testing.assert_allclose(actual, reference[key][index], rtol=0, atol=2e-8, equal_nan=True)
        np.testing.assert_allclose(env.truth, reference['labels'][index, 1:4], rtol=0, atol=2e-8)
        _, packet = physical_step(env, runtime, budget)
        np.testing.assert_allclose(packet['commands'], reference['commands'][index], rtol=0, atol=2e-8)
        report['initialization_physical_steps'] += 1

    def collect(env, runtime, task_field, maximum):
        sensors = {key: [] for key in ['raw_scans', 'input_odometry', 'input_measured_velocity',
            'per_action_guard_reasons', 'per_action_guard_executed', 'short_guard_contacts', 'supported_tracks']}
        original_step = runtime.step

        def record_step(scan, odom, velocity, timestamp, **kwargs):
            sensors['raw_scans'].append(np.asarray(scan).copy())
            sensors['input_odometry'].append(np.asarray(odom).copy())
            sensors['input_measured_velocity'].append(np.asarray(velocity).copy())
            result = original_step(scan, odom, velocity, timestamp, **kwargs)
            frame = runtime.context.guard_probe_frame
            sensors['per_action_guard_reasons'].append([str(reason) for reason in frame['reasons']])
            sensors['per_action_guard_executed'].append(np.asarray(frame['executed']).copy())
            predicted = result['predictive_constraint']
            sensors['short_guard_contacts'].append([np.nan if x is None else x for x in predicted['predicted_contact_seconds']])
            sensors['supported_tracks'].append(predicted['supported_tracks'])
            return result

        runtime.step = record_step
        try:
            rollout, row = collect_episode(env, runtime, learner, generator, task_field, budget, maximum, report)
        finally:
            runtime.step = original_step
        rollout.update({key: np.asarray(value) for key, value in sensors.items()})
        assert all(len(value) == len(rollout['actions']) for value in sensors.values())
        return rollout, row

    base_env = StressNavigationEnv(case=case, components=PARENT/'portable', scene_dir=ROOT/'training'/case['scene'],
        route_mode='shortcut', max_steps=case['max_steps'])
    base_env.reset(seed=case['seed'], options=dict(start=case['start'], goal=case['goal']))
    base_runtime = AlignedReleaseRuntime(PARENT/'portable', ROOT/'training'/case['scene']/'decision_map.yaml',
        case['start'], case['goal'], 'shortcut')
    base_runtime.policy.policy = StrictPrefix()
    try:
        for index in range(450):
            initialize(base_env, base_runtime, index)
            if index % 50 == 0:
                write(OUT/'STATUS.json', dict(status='exact_common450_initialization', frame=index+1,
                    new_ppo_steps=report['ppo_decisions'], additional_complete_decisions_this_process=0))
        snapshot = RecoverySnapshot(base_env, base_runtime)
        for batch_index in range(1, BATCHES):
            behavior_actor, behavior_critic = tensor_digest(actor), tensor_digest(learner.critic)
            episodes = []
            normal = next(c for c in tasks['normal18'] if c['id'] == ['train_failure_w11_e7', 'motion_profile_015'][batch_index])
            for release in [*RELEASES, None]:
                if release is not None:
                    env, runtime = snapshot.restore()
                    for index in range(450, release):
                        initialize(env, runtime, index)
                    task_field, maximum = field, case['max_steps']-release
                else:
                    env = StressNavigationEnv(case=normal, components=PARENT/'portable', scene_dir=ROOT/'training'/normal['scene'],
                        route_mode='shortcut', max_steps=normal['max_steps'])
                    env.reset(seed=normal['seed'], options=dict(start=normal['start'], goal=normal['goal']))
                    runtime = AlignedReleaseRuntime(PARENT/'portable', ROOT/'training'/normal['scene']/'decision_map.yaml',
                        normal['start'], normal['goal'], 'shortcut')
                    task_field, maximum = graph.field(normal['goal']), normal['max_steps']
                write(OUT/'STATUS.json', dict(status='mixed_v2_on_policy_collection', batch=batch_index+1,
                    release_frame=release, normal_case=normal['id'] if release is None else None,
                    new_ppo_steps=report['ppo_decisions']))
                try:
                    rollout, row = collect(env, runtime, task_field, maximum)
                finally:
                    env.close()
                    del env, runtime
                    gc.collect()
                    trim_allocator()
                assert tensor_digest(actor) == behavior_actor and tensor_digest(learner.critic) == behavior_critic
                path = OUT/f'episode_{len(report["episodes"])+1:02d}.npz'
                np.savez_compressed(path, **rollout)
                row.update(batch=batch_index+1, release_frame=release, full_navigation_case=normal['id'] if release is None else None,
                    initialized_recovery_not_full_mission=release is not None, rollout_path=str(path.relative_to(ROOT)),
                    rollout_sha256=digest(path), behavior_actor_sha256=behavior_actor, behavior_critic_sha256=behavior_critic)
                report['episodes'].append(row)
                episodes.append(rollout)
                save()
                print(json.dumps(row), flush=True)
            batch, normalization = prepare_completed_batch(learner, episodes, gamma=.99)
            updates = []
            for epoch in range(4):
                order = torch.randperm(len(batch['actions']), generator=minibatches).numpy()
                for start in range(0, len(order), 64):
                    selected = order[start:start+64]
                    anchors_selected = np.r_[pools[0][torch.randint(len(pools[0]), (64,), generator=minibatches).numpy()],
                        pools[1][torch.randint(len(pools[1]), (64,), generator=minibatches).numpy()]]
                    arguments = [batch[key][selected] for key in ['observations', 'masks', 'actions', 'old_log_probability', 'advantages', 'returns']]
                    stats = learner.update(*arguments, sampled_temperature=4., rollout_advantages_normalized=True,
                        anchors=anchors[anchors_selected])
                    updates.append(stats)
                    report['new_actor_optimizer_updates'] += int(stats['actor_updated'])
                    report['new_critic_optimizer_updates'] += 1
                    budget.tick()
            assert learner.stop_mismatches() == 0 and tensor_digest(actor.parent) == parent_digest
            report['batches'].append(dict(batch=batch_index+1, normalization=normalization, updates=updates))
            torch.save(dict(actor=actor.state_dict(), critic=learner.critic.state_dict(),
                actor_optimizer=learner.actor_optimizer.state_dict(), critic_optimizer=learner.critic_optimizer.state_dict(),
                frozen_reference_head=learner.reference_head.state_dict(), policy_rng=generator.get_state(),
                minibatch_rng=minibatches.get_state(), torch_rng=torch.get_rng_state(), numpy_rng=np.random.get_state(),
                python_rng=random.getstate(), contract=contract(), ppo_decisions=report['ppo_decisions'],
                parent_checkpoint_sha256=digest(PARENT/'checkpoint.pt'), resume_checkpoint_sha256=digest(resume_source),
                source_complete_decisions=1065, exact_physical_rollout_resume=False,
                deployment_approved=False), MODELS/f'batch_{batch_index+1:02d}.pt')
            save()
            del episodes, batch
            gc.collect()
            trim_allocator()
        snapshot.verify_shared_models()
        del snapshot
    finally:
        base_env.close()
        del base_env, base_runtime
        gc.collect()
        trim_allocator()
    assert report['ppo_decisions'] <= 3104 and report['initialization_physical_steps'] == 1196
    shutil.copy2(MODELS/'batch_02.pt', MODELS/'checkpoint.pt')
    shutil.copy2(PARENT/'parent_actor.pt', MODELS/'parent_actor.pt')
    with torch.no_grad():
        matches = actor(torch.from_numpy(anchors)).eq(torch.from_numpy(ACTIONS[labels])).all(1).numpy()
    assert matches[labels == 0].all()
    export(actor, ROOT/'models/stop_feedback_20261015/long_risk_bc1000_s231016/portable', MODELS/'portable', anchors)
    bundle = MODELS/'portable'
    manifest = read(bundle/'manifest.json')
    for name, value in [('release_observation.json', contract()), ('long_risk_observation.json', risk_contract())]:
        write(bundle/name, value)
        manifest['files'][name] = digest(bundle/name)
    preservation = read(bundle/'navigation_preservation.json')
    preservation.update(schema='guard_aligned_release_head_v2', long_risk_semantics=risk_contract(),
        input_semantics_changed_requires_matching_v2=True, unchanged_runtime_prefix_dimensions=439,
        unchanged_native_suffix=[481, 625], source_model=str(PARENT.relative_to(ROOT)),
        new_ppo_steps=report['ppo_decisions'], training_method='matched_v2_mixed_state_complete_batch_masked_ppo',
        original_hardware_memory_config=True)
    write(bundle/'navigation_preservation.json', preservation)
    manifest.update(required_runtime=RUNTIME_NAME, observation_contract=contract(),
        additional_ppo_steps=report['ppo_decisions'], navigation_preservation_contract=preservation, deployment_approved=False)
    manifest['files']['navigation_preservation.json'] = digest(bundle/'navigation_preservation.json')
    write(bundle/'manifest.json', manifest)
    check_files(bundle, manifest)
    write(bundle/'MODEL_CARD.json', dict(method=preservation['training_method'], new_ppo_steps=report['ppo_decisions'],
        all_reference_stop_labels_retained=True, no_teacher_controller_or_labels=True,
        frozen_components=['parent_actor', 'localizer', 'router', 'directional_docking'],
        training_success_is_not_independent_validation=True, goal_achieved=False, deployment_approved=False,
        manifest_sha256=digest(bundle/'manifest.json')))
    for name, expected in hashes.items():
        assert digest(ROOT/name) == expected, name
    report.update(completed=True, model_path=str(MODELS.relative_to(ROOT)), all_reference_stop_labels_retained=True,
        correct_source_labels=int(matches.sum()), total_source_labels=len(matches),
        source_motion_changes_require_physical_regression=True, inputs_verified=len(hashes),
        checkpoint_sha256=digest(MODELS/'checkpoint.pt'),
        at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(), resource=budget.statistics())
    save()
    write(MODELS/'RESULTS.json', report)
    write(OUT/'STATUS.json', dict(status='complete', new_ppo_steps=report['ppo_decisions'], deployment_approved=False))


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        if OUT.exists():
            write(OUT/'FAILURE.json', dict(error=repr(error), automatically_restarted=False))
        raise
