"""Freeze checked early-avoidance and normal-navigation examples before fitting.

The reference report is a reused development set. Once used here its cases
are training/retention data, not independent generalization evidence.
"""
import argparse
from collections import Counter
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil

import numpy as np
import torch

from experiments.proactive_recovery import ProactiveRecoveryPolicy, contract
from experiments.recovery_demonstrations import convert_observations, load_demonstrations
from experiments.stop_recovery_policy import warm_start_recovery
from experiments.stop_feedback_navigation import FeedbackInterface
from navrl.geometry import ROOT
from neural_navigation.decision_feedback import WAIT_SCHEMA


PROJECT = ROOT.parent
CHECKPOINT = PROJECT/'models/quality_recovery_20261014/policy_2000.zip'
SOURCE = PROJECT/'reports/stop_feedback_20261015/turn_recovery_20261015/RESULTS.json'


class Interface(FeedbackInterface):
    feedback_schema = WAIT_SCHEMA


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2)+'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    os.chdir(PROJECT)
    torch.set_num_threads(1)
    model = warm_start_recovery(CHECKPOINT, Interface(), seed=231015,
        policy_class=ProactiveRecoveryPolicy)
    reference = PROJECT/'reports/stop_feedback_20261015/prefix_recovery_threeway_20261016/RESULTS.json'
    prior = json.loads(reference.read_text())
    if not prior['completed']:
        raise ValueError('Completed development reference required')
    index = output/'normal_navigation'
    index.mkdir()
    rows, retained, excluded = [], [], []
    specifications = [dict(report=str(SOURCE.relative_to(PROJECT)), id='recover_forward',
        sensor_intervals_s=[[0., 19.8], [20., 140.]])]
    for old in prior['rows']:
        if old['variant'] != 'bc_only':
            continue
        row = dict(old, specification=dict(id=old['case_id']+'_bc_only'))
        identity = row['specification']['id']
        if (not row['safe_success'] or not row['physical_arrival'] or row['false_arrival']
                or row['collisions'] or row['physical_guards'] or row['timeout']):
            excluded.append(dict(id=identity, reason='not_a_complete_physically_safe_episode'))
            continue
        path = reference.parent/(identity+'.npz')
        if digest(path) != row['trace_sha256']:
            raise ValueError('Reference trace changed: '+identity)
        with np.load(path, allow_pickle=False) as data:
            obs = convert_observations(data['observations'], data['masks'], data['commands'], data['labels'][:, 0])
            supported = model.policy.recovery_support(torch.from_numpy(obs)).numpy()
        if not supported.any():
            excluded.append(dict(id=identity, reason='no_trainable_approach_or_recovery_states'))
            continue
        destination = index/path.name
        try:
            os.link(path, destination)
        except OSError:
            shutil.copyfile(path, destination)
        retained.append(dict(id=identity, source_trace_sha256=digest(destination),
            supported_frames=int(supported.sum())))
        rows.append(row)
        specifications.append(dict(report=str((index/'RESULTS.json').relative_to(PROJECT)), id=identity))
    write(index/'RESULTS.json',dict(completed=True, scope='Reused safe development traces for normal-task retention; not a new evaluation.',
        source_report=str(reference.relative_to(PROJECT)), source_report_sha256=digest(reference), rows=rows))
    write(output/'demonstrations.json', specifications)
    original = json.loads((PROJECT/'reports/localization_mapping_20261015/blind_repair_20261015/stop_go_training_cases.json').read_text())
    development = json.loads((reference.parent/'cases.json').read_text())
    task = dict(next(c for c in development if c['id']=='motion_profile_015'),
        id='train_failure_motion_profile_015')
    # Also expose the original failing task from reset, not only its frozen
    # 20-second state. The old course did not include this exact task.
    cases = [task]+original
    from experiments.train_localization_recovery import training_case
    for case in cases:
        training_case(case)
    write(output/'training_cases.json',cases)
    obs, masks, choices, sources = load_demonstrations(specifications, model.policy)
    write(output/'DATA.json',dict(completed=True,created_at=datetime.now().astimezone().isoformat(),
        policy_contract=contract(), source_episodes=sources, retained_normal_tasks=retained,
        excluded_reference_rows=excluded, demonstration_frames=len(obs),
        action_counts={str(k):v for k,v in Counter(choices.tolist()).items()},
        original_development_set_now_used_for_retention=True,
        simulator_truth_is_network_input=False,
        label_intervals_only_select_offline_examples=True,
        hero_interval_explanation='Keep the early turn and ordinary driving context; exclude the one immediate pre-release stop at 19.8 s. Later real waits remain. No claim that every other pre-release stop was unnecessary.',
        unchanged_body_speed_masks_guard_and_reward=True, deployment_approved=False))
    write(output/'PLAN.json',dict(frozen_at=datetime.now().astimezone().isoformat(),
        preceding_goal_turn_classification='progress',
        preceding_evidence='Committed/pushed move snapshot, completed independent steady-base 2k training and actual causal/sampling audits; all 108 migration checks passed. Two complete prefixes reached safely but no extra deterministic PPO gain; the known full-task collision persists.',
        objective='Learn why stopping occurs and choose useful legal recovery or early avoidance, while preserving valid stopping, normal navigation and directional docking.',
        hypothesis='Stop-only support cannot learn seven legal early turns in a complete safe trajectory. Expand learnable context and retain checked normal-task choices.',
        scope='Multi-component feasibility experiment: support plus explicitly curated demonstrations. Additional PPO must be compared with its own BC initialization.',
        code_contract=contract(),
        bc=dict(seed=231015,maximum_updates=400,positive_and_stop_groups_equal_weight=True),
        before_ppo=['Inspect BC fitting on every source and the seven early turns.',
            'Evaluate the original failing motion_profile_015 from the actual reset without a steering prefix.',
            'Retain failures; scripted probes are diagnostic, never scored as learned navigation.'],
        required_validation=['Original 14 complete missions including six directional tasks.',
            'Two full prefix continuations, including valid waiting statistics.',
            'Previously frozen held-out conditions with failed initialization retained.',
            'Fresh held-out failure families/maps before deployment acceptance.',
            'PPO sampled actions/old likelihoods/original masks, frozen base and actual executed history.'],
        unresolved_training_stall='All w11 prefix diagnostic continuations tried so far time out; none supplies a safe expert.',
        no_forced_runtime_maneuver_or_minimum_speed=True,no_guard_or_body_relaxation=True,
        inputs_sha256={str(p.relative_to(PROJECT)):digest(p) for p in [CHECKPOINT,SOURCE,reference]},
        primary_soft_cpu_cores=.35,auxiliary_soft_cpu_cores=.15,
        deployment_approved=False))
    print(json.dumps(dict(output=str(output),examples=len(obs),normal_tasks=len(retained),
        action_counts=Counter(choices.tolist())),indent=2),flush=True)


if __name__ == '__main__':
    main()
