"""Frozen initialization-prefix tests for the explicitly versioned release-context runtime.

Prefix commands only initialize the common physical/sensor state; the learned
actor controls every frame after release. These are not full-mission scores.
"""
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np

from experiments.evaluate_docking_navigation import evaluate
from experiments.evaluate_feedback_recovery import SensorPrefixActor, load_prefix
from experiments.release_actor import RUNTIME_NAME
from experiments.long_risk_actor import digest
from neural_navigation.goal_pose_runtime import resolve_runtime


def evaluate_recovery(specification, bundle, budget):
    manifest = json.loads((Path(bundle) / 'manifest.json').read_text())
    if manifest['required_runtime'] != RUNTIME_NAME:
        raise ValueError('Only the explicitly versioned long-release-context runtime is supported')
    runtime_type = resolve_runtime(RUNTIME_NAME)
    prefix, expected_poses = load_prefix(specification)
    created = []

    class PrefixRuntime(runtime_type):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            actor = SensorPrefixActor(self.policy.policy, self.core, prefix, specification['release_s'])
            self.policy.policy = actor
            created.append(actor)

    with patch('neural_navigation.goal_pose_runtime.resolve_runtime', return_value=PrefixRuntime):
        row, trace = evaluate(specification['case'], bundle, budget)
    if len(created) != 1 or created[0].cursor != len(prefix['times']) or not created[0].released_frames:
        raise ValueError('Common prefix was not completed and released')
    n = len(prefix['times'])
    np.testing.assert_array_equal(trace['choices'][:n], prefix['choices'])
    np.testing.assert_allclose(trace['labels'][:n+1, 1:4], expected_poses, rtol=0, atol=2e-8)
    choices, masks = trace['choices'][n:], trace['masks'][n:]
    waits = (choices == 0) & masks[:, 1:].any(axis=1)
    times = trace['labels'][n:, 0]
    durations = np.diff(np.r_[times, row['seconds']])
    longest = run = 0.
    for waiting, duration in zip(waits, durations):
        run = run + duration if waiting else 0.
        longest = max(longest, run)
    row.update(case_id=specification['id'], original_case_id=specification['case']['id'],
        recovery_test=True, not_a_full_mission_score=True, frozen_prefix_frames=n,
        prefix_sensor_and_physical_parity_verified=True,
        release_sensor_timestamp_s=specification['release_s'],
        post_release_seconds=row['seconds'] - specification['release_s'],
        model_controlled_frames=len(choices), post_release_wait_with_choices_frames=int(waits.sum()),
        post_release_only_stop_frames=int((~masks[:, 1:].any(axis=1)).sum()),
        longest_post_release_wait_with_choices_s=float(longest),
        prefix_trace_sha256=digest(specification['trace']),
        prefix_is_initialization_only=True, no_teacher_after_release=True)
    trace['model_controlled'] = np.arange(len(trace['choices'])) >= n
    return row, trace
