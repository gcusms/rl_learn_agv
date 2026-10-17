"""Explicit release-context runtime; the existing guard alone authorizes motion."""
import json
from pathlib import Path

import numpy as np

from neural_navigation.decision_feedback import contract as feedback_contract, WAIT_SCHEMA
from neural_navigation.docking_contract import docking_contract
from neural_navigation.goal_pose import append_goal_features, contract as goal_contract
from neural_navigation.goal_pose_runtime import check_files
from neural_navigation.long_risk_observation import contract as risk_contract
from neural_navigation.long_risk_runtime import LongRiskInput
from neural_navigation.quality_prediction_context import contract as quality_contract
from neural_navigation.release_observation import augment, contract, RUNTIME_DIM
from neural_navigation.runtime import SensorNavigationRuntime


RUNTIME_NAME = 'neural_navigation.release_runtime.ReleaseRuntime'


class ReleaseInput(LongRiskInput):
    def predict(self, prefix):
        quality = self.prediction.augment(prefix, self.core)
        heading = append_goal_features(quality, self.core)
        feedback = self.feedback.augment(heading, self.core, self.guard, self.context)
        old = self.history.augment(feedback, self.core, self.guard)
        self.last_observation = augment(old, self.core.scan, self.core.lidar)
        return self.policy.predict(self.last_observation)


class ReleaseRuntime(SensorNavigationRuntime):
    def __init__(self, bundle, *args, **kwargs):
        bundle = Path(bundle)
        manifest = json.loads((bundle / 'manifest.json').read_text())
        if manifest['required_runtime'] != RUNTIME_NAME or manifest['observation_contract'] != contract():
            raise ValueError('Explicit release runtime and observation contract required')
        components = {'release_observation.json': contract(), 'long_risk_observation.json': risk_contract(),
            'decision_feedback.json': feedback_contract(WAIT_SCHEMA),
            'quality_observation.json': quality_contract(), 'goal_pose_observation.json': goal_contract()}
        for name, expected in components.items():
            if json.loads((bundle / name).read_text()) != expected:
                raise ValueError('Release input component mismatch: ' + name)
        docking = manifest['docking_contract']
        if (manifest['quality_prefix_contract'] != quality_contract()
                or docking != docking_contract(docking.get('activation_distance_m'))
                or json.loads((bundle / 'docking_contract.json').read_text()) != docking):
            raise ValueError('Frozen quality/docking contract mismatch')
        check_files(bundle, manifest)
        super().__init__(bundle, *args, **kwargs)
        if self.context is None or self.policy.observation_dimensions != 282:
            raise ValueError('Native V4 guard prefix required')
        if self.guard.response_gain != 20. or not np.array_equal(self.guard.response_acceleration, [.5, 1.]):
            raise ValueError('Prediction and execution motion response must match')
        self.context.capture_decision_feedback = True
        self.policy.observation_dimensions = RUNTIME_DIM
        self.policy = ReleaseInput(self.policy, self.core, self.guard, self.context)

    def step(self, *args, **kwargs):
        result = super().step(*args, **kwargs)
        actual = self.policy.last_observation
        previous = self.policy.feedback.previous_reason
        self.policy.feedback.record(result)
        result.update(policy_observation_dimensions=RUNTIME_DIM,
            goal_pose_features=actual[300:303].tolist(),
            decision_feedback_features=actual[303:439].tolist(),
            decision_feedback_previous_reason=previous,
            decision_feedback_executed_reason=self.policy.feedback.previous_reason,
            long_risk_history_features=actual[439:481].tolist(),
            measured_scan_quality_features=actual[481:].tolist(),
            current_action_rejection_reasons=list(self.context.guard_probe_frame['reasons']))
        if 'policy_observation' in result:
            result['policy_observation'] = actual.tolist()
        return result
