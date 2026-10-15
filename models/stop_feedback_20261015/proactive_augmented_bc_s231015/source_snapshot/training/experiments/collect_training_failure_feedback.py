"""Mine deterministic recovery inputs from recorded training timeouts only."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time
import numpy as np
import torch
from stable_baselines3 import PPO
from experiments.recovery_curriculum import RecoveryNavigationCurriculum
from experiments.fixed_training_replay import reset_training_episode
from experiments.feasibility_feedback_ppo import feasibility_loss, FEEDBACK_REASONS
from neural_navigation.safety_context import RECOVERY_SCHEMA, validate_contract, OBSERVATION_DIM


def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--policy', type=Path, required=True)
    p.add_argument('--training-log', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists(): p.error('Fresh output required')
    # This collector accepts the training monitor's episode format, not a
    # development/evaluation manifest or cases copied from its results.
    records = [json.loads(line) for line in args.training_log.read_text().splitlines()]
    if not records or any('training_case_json' not in row or 'worker' not in row for row in records):
        p.error('Actual training monitor episodes required')
    records = [row for row in records if not row['verified_mission_success'] and not row['safety_termination']]
    if not records: p.error('No recorded training timeouts')
    hardware = json.loads((args.policy.parent/'hardware.json').read_text())
    validate_contract(hardware['policy_observation'])
    if hardware['policy_observation']['schema'] != RECOVERY_SCHEMA: p.error('V3 policy required')
    torch.set_num_threads(1)
    model = PPO.load(args.policy, device='cpu')
    env = RecoveryNavigationCurriculum(args.policy.parent, stage='robust',
        route_mode='shortcut', motion_profiles=True, footprint_clearance_cost=1.)
    args.output.mkdir(parents=True)
    snapshots = args.output/'source_snapshots'; snapshots.mkdir()
    for name in ['collect_training_failure_feedback.py', 'fixed_training_replay.py']:
        shutil.copy2(Path(__file__).parent/name, snapshots/name)
    meta = dict(kind='training', deterministic_collection=True, ppo_rollout_eligible=False,
        simulator_truth_in_samples=False, evaluation_manifest_consumed=False,
        collection_schema='rejected_sensor_feedback_v2', status='running', episodes=[],
        policy_sha256=digest(args.policy), training_log=str(args.training_log.resolve()),
        training_log_sha256=digest(args.training_log), requested_episodes=len(records),
        sample_count=0, started_unix=time.time())
    def save(): (args.output/'status.json').write_text(json.dumps(meta, indent=2))
    save()
    inputs, actions, commands, episodes, times, reasons = [], [], [], [], [], []
    try:
        for index, source in enumerate(records):
            case = json.loads(source['training_case_json'])
            obs, _ = reset_training_episode(env, case)
            count = 0; no_probe = 0; rejected_with_probe = 0
            for step in range(env.max_steps):
                action = model.predict(obs, deterministic=True)[0]
                with torch.no_grad():
                    _, eligible = feasibility_loss(torch.tensor(obs[None]), torch.tensor(action[None]))
                before = obs.copy(); timestamp = env.active.elapsed
                obs, _, done, truncated, info = env.step(action)
                command = info['commanded_velocity']
                idle = abs(command[0]) < .01 and abs(command[1]) < .05
                accepted = np.any(before[235:244] > .5)
                no_probe += int(idle and not accepted)
                rejected = info['training_actions_overridden'] and info['execution_guard_reason'] in FEEDBACK_REASONS and idle
                rejected_with_probe += int(rejected and accepted)
                if eligible.item() > .5 and rejected:
                    inputs.append(before); actions.append(action.copy()); commands.append(command.copy())
                    episodes.append(index); times.append(timestamp); reasons.append(info['execution_guard_reason'])
                    count += 1
                if done or truncated: break
            meta['episodes'].append(dict(index=index, source_training_step=source['step'], case=case,
                steps=step+1, collected_frames=count, idle_no_probe_frames=no_probe,
                rejected_with_probe_frames=rejected_with_probe,
                verified_success=bool(info['verified_mission_success']),
                safety_termination=bool(info['safety_termination']), timeout=bool(truncated)))
            meta['sample_count'] = len(inputs); save()
            print(json.dumps(meta['episodes'][-1]), flush=True)
        np.savez_compressed(args.output/'sensor_feedback.npz',
            observations=np.asarray(inputs, dtype=np.float32).reshape(-1, OBSERVATION_DIM),
            proposed_actions=np.asarray(actions, dtype=np.float32).reshape(-1, 2),
            commanded_velocity=np.asarray(commands, dtype=np.float32).reshape(-1, 2),
            episode_ids=np.asarray(episodes, dtype=np.int32), timestamps=np.asarray(times, dtype=np.float64),
            current_reasons=np.asarray(reasons, dtype='U48'))
        meta.update(status='complete', finished_unix=time.time(),
            sample_file_sha256=digest(args.output/'sensor_feedback.npz'),
            limitation='Original training resets recreated; current deterministic actions differ from historical stochastic training. '
                'Captured sensor inputs are auxiliary data, not PPO returns or a safety certificate.')
        save()
    except Exception as error:
        meta.update(status='failed', error=str(error)); save(); raise
    finally:
        env.close()


if __name__ == '__main__': main()
