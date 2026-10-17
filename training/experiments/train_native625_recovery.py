"""Small, native625 masked PPO pilot after the fixed parent regression ends.

Physics prefixes initialize training states and are excluded from PPO. Every
subsequent action is sampled by the trainable actor under the original mask.
Truth is used only for reward, contact and arrival evaluation.
"""
import argparse
from datetime import datetime, timezone
import gc
import json
from pathlib import Path
import random
import subprocess
import time

import numpy as np
import torch

from experiments.guard_reason_release_actor import GuardReasonReleaseActor, export
from experiments.long_risk_actor import digest
from experiments.native625_onpolicy import Native625OnPolicy, generalized_advantage
from experiments.probe_observed_turn_release import physical_step
from experiments.recovery_state_snapshot import RecoverySnapshot, tensor_digest
from experiments.refit_phase_tail_pair import write
from experiments.verify_recovery_snapshot_replay import RecordingState, RecordedActions
from navrl.map import StaticMap
from neural_navigation.experiment_budget import ExperimentBudget
from neural_navigation.goal_pose_runtime import check_files
from neural_navigation.masked_actions import ACTIONS, COMMANDS
from neural_navigation.release_runtime import ReleaseRuntime
from neural_navigation.stress_env import StressNavigationEnv
from neural_navigation.training_distance import TrainingDistanceGraph


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'reports/stop_feedback_20261015'
DEPENDENCY = 'rl-full-mission-retention-validate-20261016.service'
INVOCATION = '155e8a0192d74a169b6ee7ed8e9b13c3'


class SamplingPolicy:
    def __init__(self, learner, generator):
        self.learner, self.generator = learner, generator
        self.last_sample = None

    def predict(self, observation):
        self.last_sample = self.learner.sample(observation, self.generator)
        return ACTIONS[self.last_sample.action].copy()


def dependency_state(unit=DEPENDENCY):
    result = subprocess.run(['systemctl', '--user', 'show', unit,
        '--property=ActiveState,MainPID,InvocationID,ExecMainStatus'],
        text=True, capture_output=True, check=True)
    return dict(x.split('=', 1) for x in result.stdout.splitlines() if '=' in x)


def collect_episode(env, runtime, learner, generator, field, budget, maximum, report):
    sampler = SamplingPolicy(learner, generator)
    runtime.policy.policy = sampler
    env.minimum_gap = env._minimum_gap()
    samples, rewards, reward_terms, labels, reasons = [], [], [], [], []
    previous_command = np.asarray(runtime.context.last_executed) * [.2, .5]
    previous_distance = field.value(env.truth[:2])
    arrived = false_arrival = False
    for index in range(maximum):
        before, before_time = env.truth.copy(), float(env.elapsed)
        result = runtime.step(env.scan, env.observed_odometry, env.measured_velocity,
            env.elapsed, capture_observation=True)
        sample = sampler.last_sample
        np.testing.assert_array_equal(sample.observation, result['policy_observation'])
        sample.record_execution(result)
        goal_reached = result['stop_reason'] == 'estimated_goal_reached'
        if sample.execution_changed and not goal_reached:
            raise ValueError('A masked sampled command was changed; stop before PPO update')
        terminal = False
        terms = dict(geodesic_progress=0., time=0., travel=0., action_change=0., terminal=0.)
        if goal_reached:
            arrived = bool(np.linalg.norm(env.truth[:2] - env.goal) < .2)
            false_arrival = not arrived
            terms['terminal'] = 10. if arrived else -20.
            terminal = True
        else:
            env.step(sample.executed_command / [.2, .5])
            report['ppo_physical_steps'] += 1
            distance = field.value(env.truth[:2])
            terms['geodesic_progress'] = 5. * (previous_distance - distance)
            terms['time'] = -.01 * (env.elapsed - before_time)
            terms['travel'] = -.02 * float(np.linalg.norm(env.truth[:2] - before[:2]))
            terms['action_change'] = -.001 * float(np.abs(sample.executed_command - previous_command).sum())
            previous_distance = distance
            if env.collision_events or env.hard_guard_activations or env.minimum_gap < .04 - 1e-4:
                terms['terminal'] = -20.
                terminal = True
            elif index + 1 == maximum:
                # The source task really ends at 140 seconds. This is not an
                # arbitrary sampler truncation that bootstraps after reset.
                terms['terminal'] = -5.
                terminal = True
        previous_command = sample.executed_command.copy()
        samples.append(sample)
        reward_terms.append(terms)
        rewards.append(sum(terms.values()))
        labels.append([before_time, *before, *env.truth, float(env.minimum_gap)])
        reasons.append(str(result['stop_reason']))
        report['ppo_decisions'] += 1
        budget.tick()
        if terminal:
            break
    values = np.asarray([s.old_value for s in samples], np.float32)
    terminal = np.zeros(len(samples), bool)
    terminal[-1] = True
    # All collected episodes are complete finite tasks; no reset observation
    # is used as a final-state value.
    next_values = np.r_[values[1:], 0.].astype(np.float32)
    advantages, returns = generalized_advantage(rewards, values, next_values, terminal, terminal)
    rollout = dict(observations=np.asarray([s.observation for s in samples]),
        masks=np.asarray([s.mask for s in samples]), actions=np.asarray([s.action for s in samples]),
        old_log_probability=np.asarray([s.old_log_probability for s in samples], np.float32),
        old_values=values, rewards=np.asarray(rewards), next_values=next_values,
        terminated=terminal, episode_ends=terminal.copy(),
        executed_commands=np.asarray([s.executed_command for s in samples]),
        execution_changed=np.asarray([s.execution_changed for s in samples]),
        advantages=advantages.numpy(), returns=returns.numpy(), labels=np.asarray(labels),
        stop_reasons=np.asarray(reasons), temperature=np.asarray(learner.temperature))
    eligible = learner.evaluate(rollout['observations'])['eligible'].detach().numpy()
    return rollout, dict(safe_success=bool(arrived and not false_arrival and not env.collision_events
            and not env.hard_guard_activations and env.minimum_gap >= .04 - 1e-4),
        physical_arrival=arrived, false_arrival=false_arrival, collisions=int(env.collision_events),
        physical_guards=int(env.hard_guard_activations), minimum_dynamic_gap_m=float(env.minimum_gap),
        final_xy_error_m=float(np.linalg.norm(env.truth[:2] - env.goal)),
        training_reward=float(sum(rewards)), sampled_decisions=len(samples),
        eligible_actor_decisions=int(eligible.sum()),
        only_stop_frames=int((~rollout['masks'][:, 1:].any(1)).sum()),
        voluntary_wait_frames=int(((rollout['actions'] == 0) & rollout['masks'][:, 1:].any(1)).sum()),
        reward_terms={key:float(sum(row[key] for row in reward_terms)) for key in reward_terms[0]},
        initialized_recovery_not_full_mission=True, training_episode_not_independent_validation=True)


def main(args):
    output, models = args.report.resolve(), args.models.resolve()
    output.mkdir(parents=True, exist_ok=False)
    models.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    model = args.model.resolve()
    bundle = model / 'portable'
    source = BASE / 'context_broad_diagnostic_20261016/physical/hard3_w11_training_stall_t106.npz'
    tasks_path = BASE / 'context_feedback_longfit_protocol_20261016/tasks.json'
    dataset_path = args.anchor_dataset.resolve()
    case = json.loads(tasks_path.read_text())['actual106']['case']
    releases = args.release_frames or [572, 530]
    if len(releases) > 4 or len(set(releases)) != len(releases) or not all(0 < r < case['max_steps'] for r in releases):
        raise ValueError('One to four distinct source prefix lengths are required')
    if not 1 <= args.episodes_per_stage <= 8:
        raise ValueError('At most eight complete training episodes per initialization stage')
    maximum_decisions = sum(case['max_steps']-r for r in releases)*args.episodes_per_stage
    manifest = json.loads((bundle / 'manifest.json').read_text())
    files = [Path(__file__), model / 'checkpoint.pt', model / 'parent_actor.pt',
        model / 'RESULTS.json', bundle / 'manifest.json', source, tasks_path, dataset_path,
        ROOT / 'training/experiments/native625_onpolicy.py',
        ROOT / 'training/experiments/test_native625_onpolicy.py',
        ROOT / 'training/experiments/test_native625_collector.py',
        ROOT / 'training/experiments/recovery_state_snapshot.py',
        ROOT / 'training/experiments/probe_observed_turn_release.py',
        ROOT / 'training/experiments/verify_recovery_snapshot_replay.py',
        ROOT / 'training/neural_navigation/static_tracking_kernel.cpp']
    files += [bundle / name for name in manifest['files']]
    for folder in ['neural_navigation', 'navrl']:
        files += list((ROOT / 'training' / folder).rglob('*.py'))
    for name in ['guard_reason_release_actor', 'context_release_actor', 'release_actor', 'long_risk_actor']:
        files.append(ROOT / ('training/experiments/' + name + '.py'))
    files += [p for p in (ROOT / 'training' / case['scene']).iterdir() if p.is_file()]
    hashes = {str(p.relative_to(ROOT)): digest(p) for p in set(files)}
    write(output / 'PLAN.json', dict(inputs=hashes, dependency=args.dependency_unit,
        expected_dependency_invocation=args.dependency_invocation, no_parallel_physical_job=True,
        dependency_must_preserve_normal18_before_training=True,
        releases=[dict(prefix_frames=r, episodes=args.episodes_per_stage) for r in releases],
        maximum_ppo_decisions=maximum_decisions, maximum_initialization_physical_steps=sum(releases),
        sampling_temperature=4., learning_rate=.0003, ppo_epochs=4, minibatch_size=64,
        private_sampling_and_minibatch_rngs=True,
        gamma=.99, gae_lambda=.95, anchor_kl_strength=.05, balanced_anchor_stop_motion=True,
        rewards=dict(geodesic_progress=5., seconds=-.01, travel_m=-.02,
            command_change_l1=-.001, physical_arrival=10., collision_or_false_arrival=-20.,
            source_finite_task_deadline=-5.),
        no_wait_penalty_that_rewards_forcing_a_stop_only_state=True,
        sampled_action_likelihood_and_original_mask_retained=True,
        unchanged625_observations_actions_body4cm_masks_dynamics_and_safety=True,
        simulator_truth_only_for_reward_and_evaluation=True,
        full_horizon_tasks_terminal_not_artificial_truncation=True,
        rollout_initialization_not_ppo_experience=True,
        original_actor_greedy_parity_at_initialization=True,
        bc_optimizer_not_reused_for_new_ppo_objective=True,
        reference_head_and_parent_frozen=True, no_automatic_deployment=True))
    (output / 'driver_snapshot.py').write_text(Path(__file__).read_text())
    while True:
        state = dependency_state(args.dependency_unit)
        if state['InvocationID'] not in ('', args.dependency_invocation):
            raise RuntimeError('Validation dependency was restarted')
        if state['ActiveState'] in ('inactive', 'failed') and state['MainPID'] == '0':
            validation_path = args.parent_validation.resolve()
            validation = json.loads(validation_path.read_text())
            if state['ActiveState'] != 'inactive' or state['ExecMainStatus'] != '0' or not validation['completed']:
                raise RuntimeError('Pinned parent validation did not finish normally')
            if validation['selected_model'] != str(model.relative_to(ROOT)):
                raise RuntimeError('The validation belongs to a different model')
            if not validation.get('normal18_pair', {}).get('all_regression_checks_passed', False):
                write(output / 'RESULTS.json', dict(completed=True, ppo_started=False,
                    reason='Parent normal18 gate failed or incomplete; inspect before PPO',
                    validation_sha256=digest(validation_path), new_ppo_steps=0,
                    goal_achieved=False, deployment_approved=False))
                write(output / 'STATUS.json', dict(status='parent_gate_not_passed_no_training'))
                return
            rows = [row for group in validation.get('groups', {}).values()
                for row in group.get('rows', [])]
            if any(row.get('collisions', 0) or row.get('physical_guards', 0)
                    or row.get('false_arrival', False) for row in rows):
                write(output / 'RESULTS.json', dict(completed=True, ppo_started=False,
                    reason='Parent physical validation found a safety failure; inspect before PPO',
                    validation_sha256=digest(validation_path), new_ppo_steps=0,
                    goal_achieved=False, deployment_approved=False))
                write(output / 'STATUS.json', dict(status='parent_safety_failed_no_training'))
                return
            write(output / 'DEPENDENCIES.json', dict(validation_sha256=digest(validation_path),
                exact_parent_validation_complete=True, normal18_retention_gate_passed=True))
            break
        write(output / 'STATUS.json', dict(status='waiting_for_exact_parent_validation',
            physical_simulation_started=False, new_ppo_steps=0, dependency=state))
        time.sleep(5.)
    for name, expected in hashes.items():
        assert digest(ROOT / name) == expected, name
    budget = ExperimentBudget(.35, memory_mib=2800, directory=output)
    torch.manual_seed(231017)
    actor = GuardReasonReleaseActor(torch.jit.load(str(model / 'parent_actor.pt'), map_location='cpu')).eval()
    actor.load_state_dict(torch.load(model / 'checkpoint.pt', map_location='cpu', weights_only=False)['actor'])
    learner = Native625OnPolicy(actor, learning_rate=.0003, temperature=4.)
    frozen_parent = tensor_digest(actor.parent)
    generator = torch.Generator().manual_seed(231017)
    minibatch_generator = torch.Generator().manual_seed(231018)
    with np.load(source, allow_pickle=False) as archive:
        reference = {key: archive[key].copy() for key in archive.files}
    with np.load(dataset_path, allow_pickle=False) as archive:
        anchors, anchor_actions = archive['observations'].copy(), archive['choices'].copy()
    with torch.no_grad():
        np.testing.assert_array_equal(actor(torch.from_numpy(anchors)).numpy(), ACTIONS[anchor_actions])
    stop_pool, move_pool = [np.flatnonzero(condition) for condition in (anchor_actions == 0, anchor_actions != 0)]
    field = TrainingDistanceGraph(StaticMap(ROOT / 'training' / case['scene'] / 'decision_map.yaml', radius=.3)).field(case['goal'])
    report = dict(completed=False, ppo_started=True, episodes=[], ppo_decisions=0,
        ppo_physical_steps=0, initialization_physical_steps=0, new_optimizer_updates=0,
        new_actor_optimizer_updates=0, new_critic_optimizer_updates=0,
        goal_achieved=False, deployment_approved=False)
    try:
        for release in releases:
            env = StressNavigationEnv(case=case, components=bundle,
                scene_dir=ROOT / 'training' / case['scene'], route_mode='shortcut', max_steps=case['max_steps'])
            env.reset(seed=case['seed'], options=dict(start=case['start'], goal=case['goal']))
            runtime = ReleaseRuntime(bundle, ROOT / 'training' / case['scene'] / 'decision_map.yaml',
                case['start'], case['goal'], 'shortcut')
            runtime.policy.policy = RecordedActions(RecordingState(reference['choices']))
            for index in range(release):
                physical_step(env, runtime, budget, reference, index)
                report['initialization_physical_steps'] += 1
                if index % 50 == 0:
                    write(output / 'STATUS.json', dict(status='exact_physical_initialization',
                        release_frames=release, matched_frames=index+1, new_ppo_steps=report['ppo_decisions']))
            snapshot = RecoverySnapshot(env, runtime)
            env.close()
            del env, runtime
            gc.collect()
            for _ in range(args.episodes_per_stage):
                env, runtime = snapshot.restore()
                episode = len(report['episodes']) + 1
                write(output / 'STATUS.json', dict(status='on_policy_recovery', episode=episode,
                    release_frames=release, new_ppo_steps=report['ppo_decisions']))
                try:
                    rollout, row = collect_episode(env, runtime, learner, generator, field, budget,
                        case['max_steps']-release, report)
                finally:
                    env.close()
                    del env, runtime
                    gc.collect()
                rollout_path = output / f'episode_{episode:02d}.npz'
                np.savez_compressed(rollout_path, **rollout)
                updates = []
                for _epoch in range(4):
                    order = torch.randperm(len(rollout['actions']), generator=minibatch_generator).numpy()
                    for start in range(0, len(order), 64):
                        selected = order[start:start+64]
                        anchor_indices = np.r_[stop_pool[torch.randint(len(stop_pool), (64,), generator=minibatch_generator).numpy()],
                            move_pool[torch.randint(len(move_pool), (64,), generator=minibatch_generator).numpy()]]
                        statistics = learner.update(rollout['observations'][selected], rollout['masks'][selected],
                            rollout['actions'][selected], rollout['old_log_probability'][selected],
                            rollout['advantages'][selected], rollout['returns'][selected], sampled_temperature=4.,
                            anchors=anchors[anchor_indices])
                        updates.append(statistics)
                        report['new_actor_optimizer_updates'] += int(statistics['actor_updated'])
                        report['new_critic_optimizer_updates'] += 1
                        report['new_optimizer_updates'] += int(statistics['actor_updated']) + 1
                        budget.tick()
                row.update(episode=episode, release_frames=release,
                    source_trace=str(source.relative_to(ROOT)), rollout_path=str(rollout_path.relative_to(ROOT)),
                    rollout_sha256=digest(rollout_path), updates=updates)
                report['episodes'].append(row)
                checkpoint = models / f'episode_{episode:02d}.pt'
                torch.save(dict(actor=learner.actor.state_dict(), critic=learner.critic.state_dict(),
                    actor_optimizer=learner.actor_optimizer.state_dict(), critic_optimizer=learner.critic_optimizer.state_dict(),
                    frozen_reference_head=learner.reference_head.state_dict(),
                    policy_rng=generator.get_state(), minibatch_rng=minibatch_generator.get_state(),
                    torch_rng=torch.get_rng_state(), numpy_rng=np.random.get_state(), python_rng=random.getstate(),
                    ppo_decisions=report['ppo_decisions'], episode=episode, release_frames=release,
                    sampling_temperature=4., parent_checkpoint_sha256=digest(model / 'checkpoint.pt'),
                    exact_physical_rollout_resume=False, no_deployment_approval=True), checkpoint)
                report.update(new_ppo_steps=report['ppo_decisions'], resource=budget.statistics())
                write(output / 'RESULTS.json', report)
                print(json.dumps({k:v for k,v in row.items() if k!='updates'}), flush=True)
                assert tensor_digest(actor.parent) == frozen_parent
            snapshot.verify_shared_models()
            del snapshot
            gc.collect()
        assert report['ppo_decisions'] <= maximum_decisions
        assert report['initialization_physical_steps'] == sum(releases)
        with torch.no_grad():
            decisions = learner.actor(torch.from_numpy(anchors)).numpy()
        matches = (decisions == ACTIONS[anchor_actions]).all(1)
        safe_stops = bool(matches[anchor_actions == 0].all())
        report.update(correct_source_labels=int(matches.sum()), total_source_labels=len(matches),
            all_reference_stop_labels_retained=safe_stops,
            source_motion_differences_are_reported_not_independent_validation=True,
            diagnostic_portable_exported=False)
        if safe_stops:
            checks = np.r_[anchors, reference['observations'][530:]]
            export(learner.actor, ROOT / 'models/stop_feedback_20261015/long_risk_bc1000_s231016/portable',
                models / 'portable', checks)
            portable = models / 'portable'
            preservation = json.loads((portable / 'navigation_preservation.json').read_text())
            preservation.update(new_ppo_steps=report['ppo_decisions'],
                training_method='native625_masked_on_policy_ppo_with_reference_kl',
                sampling_temperature=4., greedy_action_rule_unchanged=True)
            write(portable / 'navigation_preservation.json', preservation)
            manifest = json.loads((portable / 'manifest.json').read_text())
            manifest.update(additional_ppo_steps=report['ppo_decisions'], deployment_approved=False,
                navigation_preservation_contract=preservation)
            manifest['files']['navigation_preservation.json'] = digest(portable / 'navigation_preservation.json')
            write(portable / 'manifest.json', manifest)
            card = json.loads((portable / 'MODEL_CARD.json').read_text())
            card.update(method='native625_masked_on_policy_ppo_with_reference_kl',
                new_ppo_steps=report['ppo_decisions'], deployment_approved=False,
                manifest_sha256=digest(portable / 'manifest.json'))
            write(portable / 'MODEL_CARD.json', card)
            check_files(portable, manifest)
            report['diagnostic_portable_exported'] = True
        for name, expected in hashes.items():
            assert digest(ROOT / name) == expected, name
        report.update(completed=True, at=datetime.now(timezone.utc).isoformat(),
            new_ppo_steps=report['ppo_decisions'], input_hashes_verified=len(hashes), resource=budget.statistics())
        write(output / 'RESULTS.json', report)
        write(output / 'STATUS.json', dict(status='complete', goal_achieved=False, deployment_approved=False))
    finally:
        assert tensor_digest(actor.parent) == frozen_parent


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--models', type=Path, required=True)
    parser.add_argument('--model', type=Path,
        default=ROOT / 'models/stop_feedback_20261015/full_mission_retention_pair_s231016/retained500')
    parser.add_argument('--anchor-dataset', type=Path,
        default=BASE / 'full_mission_retention_protocol_20261016/demonstrations.npz')
    parser.add_argument('--parent-validation', type=Path,
        default=BASE / 'full_mission_retention_validation_20261016/RESULTS.json')
    parser.add_argument('--dependency-unit', default=DEPENDENCY)
    parser.add_argument('--dependency-invocation', default=INVOCATION)
    parser.add_argument('--release-frames', type=int, action='append')
    parser.add_argument('--episodes-per-stage', type=int, default=8)
    args = parser.parse_args()
    try:
        main(args)
    except Exception as error:
        if args.report.is_dir():
            write(args.report / 'FAILURE.json', dict(error=repr(error), automatically_restarted=False))
        raise
