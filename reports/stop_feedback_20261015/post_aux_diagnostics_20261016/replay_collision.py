"""Reproduce 25 actual sampled actions; inspect native contact without a policy.

The fixed replay is a diagnostic, never a trained-model navigation score.
Simulator poses and contacts are logged only, not supplied to a controller.
"""
from datetime import datetime
import hashlib
import json
from pathlib import Path
from unittest.mock import patch
from zoneinfo import ZoneInfo

import mujoco
import numpy as np
import torch

from experiments.prefix_recovery_training import PrefixRecoveryTraining
from experiments.recovery_sampling import ScheduledRecoveryCurriculum
from experiments.stop_feedback_navigation import StopFeedbackWrapper
from experiments.train_localization_recovery import training_case
from neural_navigation.decision_feedback import WAIT_SCHEMA
from neural_navigation.experiment_budget import ExperimentBudget


ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    destination = OUT/'collision_replay'
    destination.mkdir(exist_ok=False)
    plan = json.loads((OUT/'PLAN.json').read_text())
    for name, expected in plan['inputs'].items():
        if digest(ROOT/name) != expected:
            raise ValueError('Recorded diagnostic input changed: '+name)
    torch.set_num_threads(1)
    budget = ExperimentBudget(.15, memory_mib=2800, directory=destination)
    with np.load(OUT/'recorded_collision_episode7.npz',allow_pickle=False) as data:
        expected_obs = data['observations'].copy()
        expected_masks = data['masks'].copy()
        choices = data['choices'].copy()
        expected_info = [json.loads(str(value)) for value in data['diagnostics_json']]
    spec = json.loads((OUT/'collision_specification.json').read_text())
    physical = ScheduledRecoveryCurriculum(
        ROOT/'models/stop_feedback_20261015/baseline/portable',stage='robust',
        route_mode='shortcut',motion_profiles=True,
        recorded_cases=[training_case(spec['case'])],recorded_episodes_per_cycle=1)
    wrapped = StopFeedbackWrapper(physical,reward_scale=1/60,feedback_schema=WAIT_SCHEMA)
    env = PrefixRecoveryTraining(wrapped,[spec],destination,budget)
    contacts, rows, scans, odometry = [], [], [], []
    human_positions, observed, sampled_masks = [], [], []
    active_index = -1
    original_forward = mujoco.mj_forward
    try:
        obs, _ = env.reset(seed=231015)
        if env.initialization_frames != 30:
            raise ValueError('Expected complete 30-frame early initialization')
        e = physical.active

        def capture_forward(model,data):
            result = original_forward(model,data)
            if model is e.model and data is e.data:
                for contact in data.contact:
                    pair = {int(contact.geom1),int(contact.geom2)}
                    if not pair & e.robot_geoms:
                        continue
                    if not (pair & e.wall_geoms or pair & set(e.human_geoms)):
                        continue
                    names = [mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_GEOM,int(g))
                             for g in (contact.geom1,contact.geom2)]
                    contacts.append(dict(replay_index=active_index,
                        data_time_s=float(data.time),source='native_mujoco_contact',
                        geom_names=names,signed_distance_m=float(contact.dist),
                        contact_position=np.asarray(contact.pos).tolist(),
                        robot_pose=e._pose().tolist(),
                        humans=[p.tolist() for p in e._humans()]))
            return result

        with patch('mujoco.mj_forward',side_effect=capture_forward):
            for i, choice in enumerate(choices):
                active_index = i
                np.testing.assert_allclose(obs,expected_obs[i],rtol=0,atol=2e-6)
                mask = env.action_masks()
                np.testing.assert_array_equal(mask,expected_masks[i])
                if not mask[int(choice)]:
                    raise ValueError('Never replay an inadmissible action')
                state = dict(index=i,sensor_timestamp_s=float(physical.core.elapsed),
                    sampled_choice=int(choice),legal_actions=np.flatnonzero(mask).tolist(),
                    true_pose=e.truth.tolist(),estimated_pose=physical.core.estimate.tolist(),
                    actual_velocity=e.velocity.tolist(),
                    human_velocities=[e.data.qvel[e.addresses[n][1]].tolist() for n in e.names[1:]],
                    robot_to_human_gap_m=[e.robot_gap(p) for p in e._humans()],
                    original_sample_probability=None,
                    maximum_observation_difference=float(np.abs(obs-expected_obs[i]).max()))
                observed.append(obs.copy());sampled_masks.append(mask.copy())
                scans.append(physical.core.scan.copy());odometry.append(physical.core.odom.copy())
                human_positions.append(np.asarray(e._humans()))
                obs,_,done,truncated,info = env.step(int(choice))
                expected = expected_info[i]
                np.testing.assert_allclose(info['commanded_velocity'],expected['executed_command'],
                    rtol=0,atol=2e-8)
                np.testing.assert_allclose(info['native_reward'],expected['native_reward'],
                    rtol=0,atol=2e-6)
                if info['decision_feedback_previous_reason'] != expected['executed_reason']:
                    raise ValueError('Execution reason differs from actual training')
                if bool(done or truncated) != bool(expected['done']):
                    raise ValueError('Replay ended differently from the actual sample')
                state.update(executed_command=np.asarray(info['commanded_velocity']).tolist(),
                    executed_reason=info['decision_feedback_previous_reason'],
                    true_pose_after=e.truth.tolist(),actual_velocity_after=e.velocity.tolist(),
                    collisions_after=int(info['collision_events']),
                    physical_guards_after=int(e.hard_guard_activations),
                    native_reward=float(info['native_reward']),terminated=bool(done),truncated=bool(truncated))
                rows.append(state)
                budget.tick()
        if not rows[-1]['terminated'] or rows[-1]['collisions_after'] != 1:
            raise ValueError('The exact recorded native-contact termination was not reproduced')
        path = destination/'TRACE.npz'
        np.savez_compressed(path,observations=np.asarray(observed),masks=np.asarray(sampled_masks),
            choices=choices,scans=np.asarray(scans),odometry=np.asarray(odometry),
            human_positions=np.asarray(human_positions))
        result = dict(completed=True,at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
            all25sampled_states_masks_commands_and_rewards_reproduced=True,
            warmup_frames_excluded=env.initialization_frames,recorded_steps_replayed=len(choices),
            physical_contact_inspection_only=True,model_navigation_evaluation=False,
            collisions=rows[-1]['collisions_after'],physical_guards=rows[-1]['physical_guards_after'],
            frames=rows,native_contacts=contacts,
            first_native_contact=contacts[0] if contacts else None,
            raw_trace_sha256=digest(path),runtime_and_training_code_changed=False,
            source_program_sha256=digest(Path(__file__)),resource=budget.statistics(),
            new_ppo_steps=0,goal_achieved=False,deployment_approved=False)
        (destination/'RESULTS.json').write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(dict(completed=True,exact_recorded_states=len(rows),
            collisions=result['collisions'],first_native_contact=result['first_native_contact'],
            resource=result['resource']),indent=2),flush=True)
    finally:
        env.close()


if __name__=='__main__':
    main()
