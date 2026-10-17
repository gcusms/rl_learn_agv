"""Versioned long-risk actor runtime with the original physical safety checks."""
import json
from pathlib import Path

import numpy as np

from neural_navigation.decision_feedback import contract as feedback_contract, WAIT_SCHEMA
from neural_navigation.decision_feedback_runtime import FeedbackInput
from neural_navigation.docking_contract import docking_contract
from neural_navigation.goal_pose import append_goal_features, contract as goal_contract
from neural_navigation.goal_pose_runtime import check_files
from neural_navigation.long_risk_observation import LongRiskHistory, contract, RUNTIME_DIM
from neural_navigation.quality_prediction_context import contract as quality_contract
from neural_navigation.runtime import SensorNavigationRuntime


RUNTIME_NAME = 'neural_navigation.long_risk_runtime.LongRiskRuntime'


class LongRiskInput(FeedbackInput):
    def __init__(self, policy, core, guard, context):
        super().__init__(policy, core, guard, context, WAIT_SCHEMA)
        self.history = LongRiskHistory()

    def predict(self, prefix):
        quality = self.prediction.augment(prefix, self.core)
        heading = append_goal_features(quality, self.core)
        feedback = self.feedback.augment(heading, self.core, self.guard, self.context)
        self.last_observation = self.history.augment(feedback, self.core, self.guard)
        return self.policy.predict(self.last_observation)

    def on_runtime_reset(self, context):
        super().on_runtime_reset(context)
        self.history.reset()


class LongRiskRuntime(SensorNavigationRuntime):
    def __init__(self, bundle, *args, **kwargs):
        bundle = Path(bundle)
        manifest = json.loads((bundle / 'manifest.json').read_text())
        if manifest['required_runtime'] != RUNTIME_NAME or manifest['observation_contract'] != contract():
            raise ValueError('Explicit long-risk runtime and observation contract required')
        if json.loads((bundle / 'long_risk_observation.json').read_text()) != contract():
            raise ValueError('Long-risk observation component mismatch')
        if json.loads((bundle / 'decision_feedback.json').read_text()) != feedback_contract(WAIT_SCHEMA):
            raise ValueError('Unchanged V2 feedback component required')
        if (manifest['quality_prefix_contract'] != quality_contract()
                or json.loads((bundle / 'quality_observation.json').read_text()) != quality_contract()
                or json.loads((bundle / 'goal_pose_observation.json').read_text()) != goal_contract()):
            raise ValueError('Unchanged quality and goal contracts required')
        docking = manifest['docking_contract']
        if docking != docking_contract(docking.get('activation_distance_m')) or json.loads(
                (bundle / 'docking_contract.json').read_text()) != docking:
            raise ValueError('Directional docking contract mismatch')
        check_files(bundle, manifest)
        super().__init__(bundle, *args, **kwargs)
        if self.context is None or self.policy.observation_dimensions != 282:
            raise ValueError('Native V4 guard prefix required')
        if self.guard.response_gain != 20. or not np.array_equal(self.guard.response_acceleration, [.5, 1.]):
            raise ValueError('Forecast response and execution response must match')
        self.context.capture_decision_feedback = True
        self.policy.observation_dimensions = RUNTIME_DIM
        self.policy = LongRiskInput(self.policy, self.core, self.guard, self.context)

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
            long_risk_history_features=actual[439:].tolist(),
            current_action_rejection_reasons=list(self.context.guard_probe_frame['reasons']))
        if 'policy_observation' in result:
            result['policy_observation'] = actual.tolist()
        return result
