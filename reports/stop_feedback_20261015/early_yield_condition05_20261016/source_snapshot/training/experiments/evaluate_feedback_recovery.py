"""Evaluate learned continuation from reproducible, physically reached stalls.

A frozen sensor/action prefix initializes the complete runtime history. Every
prefix command must remain admissible and reproduce its source sensor inputs.
After release, the model alone selects commands. These are recovery tests,
not full missions driven exclusively by the evaluated model from the start.
"""
import argparse
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from experiments.evaluate_docking_navigation import evaluate
from neural_navigation.decision_feedback_runtime import StopFeedbackRuntime
from neural_navigation.experiment_budget import ExperimentBudget
from neural_navigation.masked_actions import ACTIONS, masks_from_observation


class SensorPrefixActor:
    def __init__(self, actor, core, prefix, release_s):
        self.actor, self.core, self.prefix = actor, core, prefix
        self.release_s = release_s
        self.cursor = 0
        self.released_frames = 0

    def predict(self, observation):
        if self.cursor == len(self.prefix['times']):
            if self.core.elapsed < self.release_s-1e-8:
                raise ValueError('The policy was released before its declared sensor time')
            self.released_frames += 1
            return self.actor.predict(observation)
        i = self.cursor
        if abs(self.core.elapsed-self.prefix['times'][i]) > 1e-8:
            raise ValueError('The frozen prefix skipped or repeated a sensor frame')
        for key, actual in [('scans', self.core.scan), ('odometry', self.core.odom),
                            ('measured_velocity', self.core.measured_velocity)]:
            if not np.allclose(actual, self.prefix[key][i], rtol=0, atol=2e-8, equal_nan=True):
                raise ValueError('The frozen prefix failed to reproduce its '+key)
        choice = int(self.prefix['choices'][i])
        if not masks_from_observation(np.asarray(observation)[:282])[choice]:
            raise ValueError('A frozen prefix command is no longer admissible')
        self.cursor += 1
        return ACTIONS[choice].copy()


def load_prefix(specification):
    path = Path(specification['trace'])
    release = float(specification['release_s'])
    with np.load(path, allow_pickle=False) as data:
        times = data['labels'][:, 0]
        n = int(np.count_nonzero(times < release-1e-8))
        if (not np.isfinite(release) or n < 1 or n >= len(times)
                or abs(times[n]-release) > 1e-8 or np.any(np.diff(times) <= 0)):
            raise ValueError('A release frame inside a monotonic source trace is required')
        prefix = {key: data[key][:n].copy() for key in
                  ['choices', 'scans', 'odometry', 'measured_velocity']}
        prefix['times'] = times[:n].copy()
        # Physical labels are kept outside the actor and used only to verify
        # that both models truly start continuation in the same physical state.
        expected_poses = data['labels'][:n+1, 1:4].copy()
    if not np.isin(prefix['choices'], np.arange(12)).all():
        raise ValueError('Frozen discrete action index is invalid')
    return prefix, expected_poses


def evaluate_recovery(specification, bundle, budget):
    prefix, expected_poses = load_prefix(specification)
    created = []

    class PrefixRuntime(StopFeedbackRuntime):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            actor = SensorPrefixActor(self.policy.policy, self.core, prefix, specification['release_s'])
            self.policy.policy = actor
            created.append(actor)

    with patch('neural_navigation.goal_pose_runtime.resolve_runtime', return_value=PrefixRuntime):
        row, trace = evaluate(specification['case'], bundle, budget)
    if len(created) != 1 or created[0].cursor != len(prefix['times']) or not created[0].released_frames:
        raise ValueError('Recovery evaluation did not finish its initialization prefix')
    n = len(prefix['times'])
    np.testing.assert_array_equal(trace['choices'][:n], prefix['choices'])
    np.testing.assert_allclose(trace['labels'][:n+1, 1:4], expected_poses, atol=2e-8, rtol=0)
    choices = trace['choices'][n:]
    masks = trace['masks'][n:]
    waits = (choices == 0) & masks[:, 1:].any(-1)
    times = trace['labels'][n:, 0]
    durations = np.diff(np.r_[times, row['seconds']])
    longest = run = 0.
    for waiting, duration in zip(waits, durations):
        run = run+duration if waiting else 0.
        longest = max(longest, run)
    row.update(case_id=specification['id'], original_case_id=specification['case']['id'],
        recovery_test=True, not_a_full_mission_score=True,
        frozen_prefix_frames=n, prefix_sensor_and_physical_parity_verified=True,
        release_sensor_timestamp_s=specification['release_s'],
        post_release_seconds=row['seconds']-specification['release_s'],
        model_controlled_frames=len(choices),
        post_release_wait_with_choices_frames=int(waits.sum()),
        post_release_only_stop_frames=int((~masks[:, 1:].any(-1)).sum()),
        longest_post_release_wait_with_choices_s=float(longest),
        prefix_trace_sha256=hashlib.sha256(Path(specification['trace']).read_bytes()).hexdigest())
    trace['model_controlled'] = np.arange(len(trace['choices'])) >= n
    return row, trace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['bundle', 'specifications', 'output']:
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--label', default='reference')
    parser.add_argument('--cpu-cores', type=float, default=.15)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    budget = ExperimentBudget(args.cpu_cores, memory_mib=3072, directory=args.output)
    specifications = json.loads(args.specifications.read_text())
    (args.output/'specifications.json').write_text(json.dumps(specifications, indent=2)+'\n')
    from experiments.experiment_provenance import record_provenance
    record_provenance(args.output, args.bundle)
    report = dict(completed=False, deployment_approved=False, rows=[],
        label=args.label, specifications_sha256=hashlib.sha256(args.specifications.read_bytes()).hexdigest(),
        bundle_manifest_sha256=hashlib.sha256((args.bundle/'manifest.json').read_bytes()).hexdigest(),
        scope='Learned continuation after a fixed, independently verified sensor/action prefix; reused development traces.',
        limitations=['Correlated recovery states from previously used trajectories, not independent missions.',
                     'Prefix is scripted initialization, not a claim that the model chose the earlier maneuver.'])
    ids = set()
    for specification in specifications:
        name = specification['id']
        if not name or name in ids or Path(name).name != name:
            raise ValueError('Unique simple recovery case names are required')
        ids.add(name)
        row, trace = evaluate_recovery(specification, args.bundle, budget)
        path = args.output/(name+'.npz')
        np.savez_compressed(path, **trace)
        row.update(variant=args.label, trace_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        report['rows'].append(row); report['resource'] = budget.statistics()
        (args.output/'RESULTS.json').write_text(json.dumps(report, indent=2)+'\n')
        print(json.dumps(row), flush=True)
    report['completed'] = True
    (args.output/'RESULTS.json').write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__': main()
