"""Portable directional runtime for the explicit appended decision contract."""
import json
from pathlib import Path
from neural_navigation.runtime import SensorNavigationRuntime
from neural_navigation.goal_pose_runtime import check_files
from neural_navigation.quality_prediction_context import QualityPredictionContext, contract as quality_contract
from neural_navigation.goal_pose import append_goal_features, contract as goal_contract
from neural_navigation.docking_contract import docking_contract
from neural_navigation.decision_feedback import DecisionFeedback, contract, RUNTIME_DIM


RUNTIME_NAME = 'neural_navigation.decision_feedback_runtime.StopFeedbackRuntime'


class FeedbackInput:
    def __init__(self, policy, core, guard, context):
        self.policy=policy; self.core=core; self.guard=guard; self.context=context
        self.prediction=QualityPredictionContext(guard.half*2)
        self.feedback=DecisionFeedback(); self.last_observation=None

    def predict(self, prefix):
        quality=self.prediction.augment(prefix, self.core)
        heading=append_goal_features(quality, self.core)
        self.last_observation=self.feedback.augment(heading, self.core, self.guard, self.context)
        return self.policy.predict(self.last_observation)

    def on_runtime_reset(self, context):
        self.context=context; self.prediction.reset(); self.feedback.reset()
        self.last_observation=None


class StopFeedbackRuntime(SensorNavigationRuntime):
    def __init__(self, bundle, *args, **kwargs):
        bundle=Path(bundle); manifest=json.loads((bundle/'manifest.json').read_text())
        if manifest['required_runtime']!=RUNTIME_NAME or manifest['observation_contract']!=contract():
            raise ValueError('Explicit stop decision runtime/observation contract required')
        if json.loads((bundle/'decision_feedback.json').read_text())!=contract():
            raise ValueError('Decision feedback component contract mismatch')
        if (manifest['quality_prefix_contract']!=quality_contract()
            or json.loads((bundle/'quality_observation.json').read_text())!=quality_contract()
            or json.loads((bundle/'goal_pose_observation.json').read_text())!=goal_contract()):
            raise ValueError('Legacy prefix contract mismatch')
        docking=manifest['docking_contract']
        if docking!=docking_contract(docking.get('activation_distance_m')) or json.loads((bundle/'docking_contract.json').read_text())!=docking:
            raise ValueError('Directional docking contract mismatch')
        check_files(bundle, manifest)
        super().__init__(bundle, *args, **kwargs)
        if self.context is None or self.policy.observation_dimensions!=282:
            raise ValueError('Native V4 guard prefix required')
        self.context.capture_decision_feedback=True
        self.policy.observation_dimensions=RUNTIME_DIM
        self.policy=FeedbackInput(self.policy, self.core, self.guard, self.context)

    def step(self, *args, **kwargs):
        result=super().step(*args, **kwargs)
        actual=self.policy.last_observation
        previous_reason=self.policy.feedback.previous_reason
        self.policy.feedback.record(result)
        result['policy_observation_dimensions']=RUNTIME_DIM
        result['goal_pose_features']=actual[300:303].tolist()
        result['decision_feedback_features']=actual[303:].tolist()
        result['decision_feedback_previous_reason']=previous_reason
        result['decision_feedback_executed_reason']=self.policy.feedback.previous_reason
        result['current_action_rejection_reasons']=list(self.context.guard_probe_frame['reasons'])
        if 'policy_observation' in result:
            result['policy_observation']=actual.tolist()
        return result
