"""Explicit v2 release runtime for actors trained on aligned forecast history."""
import json
from pathlib import Path

import numpy as np

from neural_navigation.aligned_risk_observation import AlignedRiskHistory, contract as risk_contract
from neural_navigation.decision_feedback import contract as feedback_contract, WAIT_SCHEMA
from neural_navigation.docking_contract import docking_contract
from neural_navigation.goal_pose import contract as goal_contract
from neural_navigation.goal_pose_runtime import check_files
from neural_navigation.quality_prediction_context import contract as quality_contract
from neural_navigation.release_observation import contract as original_release_contract
from neural_navigation.release_runtime import ReleaseInput, ReleaseRuntime
from neural_navigation.runtime import SensorNavigationRuntime


RUNTIME_NAME = 'neural_navigation.aligned_release_runtime.AlignedReleaseRuntime'


def contract():
    return dict(original_release_contract(), schema='risk_and_measured_release_context_v2_guard_aligned',
        parent_contract=risk_contract(), requires_retraining_after_v1=True)


class AlignedReleaseRuntime(ReleaseRuntime):
    def __init__(self, bundle, *args, **kwargs):
        bundle = Path(bundle)
        manifest = json.loads((bundle / 'manifest.json').read_text())
        if manifest.get('required_runtime') != RUNTIME_NAME or manifest['observation_contract'] != contract():
            raise ValueError('Aligned v2 actor, runtime and observation contract required')
        components = {'release_observation.json': contract(), 'long_risk_observation.json': risk_contract(),
            'decision_feedback.json': feedback_contract(WAIT_SCHEMA),
            'quality_observation.json': quality_contract(), 'goal_pose_observation.json': goal_contract()}
        for name, expected in components.items():
            if json.loads((bundle / name).read_text()) != expected:
                raise ValueError('Aligned release input component mismatch: ' + name)
        docking = manifest['docking_contract']
        if (manifest['quality_prefix_contract'] != quality_contract()
                or docking != docking_contract(docking.get('activation_distance_m'))
                or json.loads((bundle / 'docking_contract.json').read_text()) != docking):
            raise ValueError('Frozen quality/docking contract mismatch')
        check_files(bundle, manifest)
        # Skip the v1-only constructor. The inherited step still performs the
        # same physical command checks and records actual execution feedback.
        SensorNavigationRuntime.__init__(self, bundle, *args, **kwargs)
        if self.context is None or self.policy.observation_dimensions != 282:
            raise ValueError('Native V4 guard prefix required')
        if self.guard.response_gain != 20. or not np.array_equal(self.guard.response_acceleration, [.5, 1.]):
            raise ValueError('Prediction and execution motion response must match')
        self.context.capture_decision_feedback = True
        self.policy.observation_dimensions = 625
        self.policy = ReleaseInput(self.policy, self.core, self.guard, self.context)
        self.policy.history = AlignedRiskHistory()
