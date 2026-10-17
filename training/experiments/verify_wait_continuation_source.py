"""Verify an old RL policy's complete continuation from an actual new wait.

The learner's failed prefix only initializes the state. No action from that
prefix is admitted as a positive label. Subsequent commands use current raw
sensor state and the frozen439 RL actor, with every original guard retained.
"""
import argparse
from datetime import datetime, timezone
import gc
import json
from pathlib import Path

import numpy as np
import torch

from experiments.long_risk_actor import digest
from experiments.probe_observed_turn_release import physical_step
from experiments.refit_phase_tail_pair import write
from experiments.replay_reference_missions625 import FrozenReference
from experiments.verify_recovery_snapshot_replay import RecordingState, RecordedActions
from neural_navigation.experiment_budget import ExperimentBudget
from neural_navigation.release_runtime import ReleaseRuntime
from neural_navigation.stress_env import StressNavigationEnv


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'reports/stop_feedback_20261015'


def main(args):
    output, model = args.output.resolve(), args.model.resolve()
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    budget = ExperimentBudget(args.cpu_cores, memory_mib=2800, directory=output)
    bundle = model / 'portable'
    tasks_path = BASE / 'context_feedback_longfit_protocol_20261016/tasks.json'
    case = next(c for c in json.loads(tasks_path.read_text())['normal18'] if c['id'] == 'independent_dynamic_000')
    source = BASE / 'full_mission_retention_validation_20261016/physical/normal_other16_independent_dynamic_000.npz'
    manifest = json.loads((bundle / 'manifest.json').read_text())
    files = [Path(__file__), source, tasks_path, model / 'parent_actor.pt', bundle / 'manifest.json',
        ROOT / 'training/experiments/probe_observed_turn_release.py',
        ROOT / 'training/experiments/replay_reference_missions625.py',
        ROOT / 'training/experiments/verify_recovery_snapshot_replay.py',
        ROOT / 'training/experiments/recovery_state_snapshot.py']
    files += [bundle / name for name in manifest['files']]
    for folder in ['navrl', 'neural_navigation']:
        files += list((ROOT / 'training' / folder).rglob('*.py'))
    files += [ROOT / 'training/neural_navigation/static_tracking_kernel.cpp']
    files += [p for p in (ROOT / 'training' / case['scene']).iterdir() if p.is_file()]
    hashes = {str(p.relative_to(ROOT)): digest(p) for p in set(files)}
    write(output / 'PLAN.json', dict(inputs=hashes, case=case, prefix_frames=args.prefix_frames,
        prefix_initialization_is_not_positive_training_data=True,
        teacher='frozen439_RL_actor_on_current625_prefix', no_forced_motion=True,
        original_actions_body4cm_sensors_masks_dynamics_and_deadline=True,
        maximum_new_physical_steps=700, complete_safe_continuation_required_for_source_admission=True,
        no_new_candidate_success_or_independent_generalization=True,
        no_optimizer=True, new_ppo_steps=0, deployment_approved=False))
    (output / 'driver_snapshot.py').write_text(Path(__file__).read_text())
    with np.load(source, allow_pickle=False) as archive:
        reference = {key: archive[key].copy() for key in archive.files}
    env = StressNavigationEnv(case=case, components=bundle,
        scene_dir=ROOT / 'training' / case['scene'], route_mode='shortcut', max_steps=case['max_steps'])
    env.reset(seed=case['seed'], options=dict(start=case['start'], goal=case['goal']))
    runtime = ReleaseRuntime(bundle, ROOT / 'training' / case['scene'] / 'decision_map.yaml',
        case['start'], case['goal'], 'shortcut')
    runtime.policy.policy = RecordedActions(RecordingState(reference['choices']))
    report = dict(completed=False, new_physical_steps=0, new_ppo_steps=0,
        goal_achieved=False, deployment_approved=False)
    try:
        for frame in range(args.prefix_frames):
            physical_step(env, runtime, budget, reference, frame)
            report['new_physical_steps'] += 1
            if frame % 25 == 0:
                write(output / 'STATUS.json', dict(status='exact_learner_physical_prefix', matched_frames=frame+1))
        # Release at a learner-visited, measured waiting state. No time or
        # privileged world state enters any subsequent teacher decision.
        runtime.policy.policy = FrozenReference(torch.jit.load(str(model / 'parent_actor.pt'), map_location='cpu').eval())
        env.minimum_gap = env._minimum_gap()
        released_at = float(env.elapsed)
        packets = []
        reasons = []
        arrived = false_arrival = False
        for frame in range(args.prefix_frames, case['max_steps']):
            result, packet = physical_step(env, runtime, budget)
            packets.append(packet)
            reasons.append(str(result['stop_reason']))
            if result['stop_reason'] == 'estimated_goal_reached':
                arrived = bool(np.linalg.norm(env.truth[:2] - env.goal) < .2)
                false_arrival = not arrived
                break
            report['new_physical_steps'] += 1
            if env.collision_events or env.hard_guard_activations or env.minimum_gap < .04 - 1e-4:
                break
            if len(packets) % 25 == 0:
                write(output / 'STATUS.json', dict(status='frozen_RL_continuation',
                    frames=len(packets), elapsed=float(env.elapsed)))
        success = bool(arrived and not false_arrival and not env.collision_events
            and not env.hard_guard_activations and env.minimum_gap >= .04 - 1e-4)
        target = output / 'continuation625.npz'
        arrays = {key: np.asarray([packet[key] for packet in packets]) for key in packets[0]}
        arrays['stop_reasons'] = np.asarray(reasons)
        np.savez_compressed(target, **arrays)
        assert report['new_physical_steps'] <= 700
        for name, expected in hashes.items():
            assert digest(ROOT / name) == expected, name
        budget.tick()
        report.update(completed=True, at=datetime.now(timezone.utc).isoformat(),
            safe_success=success, source_admitted=success, reference_prefix_exact=True,
            prefix_frames=args.prefix_frames, complete_continuation_frames=len(packets),
            released_at_s=released_at, continuation_seconds=float(env.elapsed-released_at),
            collisions=int(env.collision_events), physical_guards=int(env.hard_guard_activations),
            false_arrival=false_arrival, minimum_post_release_dynamic_gap_m=float(env.minimum_gap),
            final_xy_error_m=float(np.linalg.norm(env.truth[:2]-env.goal)),
            actual_wait_state_not_original_teacher_only_distribution=True,
            teacher_success_not_new_candidate_success=True,
            trace_path=str(target.relative_to(ROOT)), trace_sha256=digest(target),
            inputs_verified=len(hashes), resource=budget.statistics())
        write(output / 'RESULTS.json', report)
        write(output / 'STATUS.json', dict(status='complete', source_admitted=success))
        print(json.dumps(report), flush=True)
    finally:
        env.close()
        del env, runtime
        gc.collect()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--prefix-frames', type=int, default=175)
    parser.add_argument('--cpu-cores', type=float, default=.2)
    args = parser.parse_args()
    try:
        main(args)
    except Exception as error:
        if args.output.is_dir():
            write(args.output / 'FAILURE.json', dict(error=repr(error), automatically_restarted=False))
        raise
