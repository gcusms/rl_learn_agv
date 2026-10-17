"""Portable directional actors: no Gym, PPO, MuJoCo or ROS imports.

Legacy manifests retain their pinned runtime names. Deployment may resolve those
names here without importing the training modules that originally defined them.
"""
import hashlib
import json
from pathlib import Path
from neural_navigation.runtime import SensorNavigationRuntime
from neural_navigation.goal_pose import append_goal_features,contract
from neural_navigation.docking_contract import docking_contract
from neural_navigation.quality_prediction_context import QualityPredictionContext,contract as quality_contract


def check_files(bundle,manifest):
    for optional in ('localization_contract.json','localization_kernel.so','routing_contract.json','uncertainty_contract.json','decision_feedback.json'):
        if (bundle/optional).exists() and optional not in manifest['files']:
            raise ValueError('Missing localization component hash: '+optional)
    for name,digest in manifest['files'].items():
        if Path(name).name!=name or hashlib.sha256((bundle/name).read_bytes()).hexdigest()!=digest:
            raise ValueError('Bundle integrity failure')


class GoalPoseInput:
    def __init__(self,policy,core,prediction):
        self.policy=policy;self.core=core;self.prediction=prediction;self.last_observation=None
    def predict(self,prefix):
        self.last_observation=append_goal_features(self.prediction.augment(prefix,self.core),self.core)
        return self.policy.predict(self.last_observation)


class GoalPoseRuntime(SensorNavigationRuntime):
    def __init__(self,bundle,*args,**kwargs):
        bundle=Path(bundle);manifest=json.loads((bundle/'manifest.json').read_text())
        if manifest['required_runtime']!='experiments.goal_pose_navigation.GoalPoseRuntime':
            raise ValueError('Goal-pose runtime mismatch')
        self._initialize_goal_runtime(bundle,manifest,*args,**kwargs)

    def _initialize_goal_runtime(self,bundle,manifest,*args,**kwargs):
        if (manifest['observation_contract']!=contract() or manifest['quality_prefix_contract']!=quality_contract()
            or json.loads((bundle/'goal_pose_observation.json').read_text())!=contract()
            or json.loads((bundle/'quality_observation.json').read_text())!=quality_contract()):
            raise ValueError('Goal/quality contract mismatch')
        check_files(bundle,manifest)
        SensorNavigationRuntime.__init__(self,bundle,*args,**kwargs)
        if self.policy.observation_dimensions!=282:raise ValueError('Native V4 prefix required')
        self.policy.observation_dimensions=303
        self.policy=GoalPoseInput(self.policy,self.core,QualityPredictionContext(self.guard.half*2))

    def step(self,*args,**kwargs):
        result=super().step(*args,**kwargs);actual=self.policy.last_observation
        result['policy_observation_dimensions']=303;result['goal_pose_features']=actual[300:].tolist()
        if 'policy_observation' in result:result['policy_observation']=actual.tolist()
        return result


class DockingRuntime(GoalPoseRuntime):
    def __init__(self,bundle,*args,**kwargs):
        bundle=Path(bundle);manifest=json.loads((bundle/'manifest.json').read_text())
        actual=manifest['docking_contract'];expected=docking_contract(actual.get('activation_distance_m'))
        if (manifest['required_runtime']!='experiments.docking_navigation.DockingRuntime'
            or actual!=expected or json.loads((bundle/'docking_contract.json').read_text())!=expected):
            raise ValueError('Docking contract mismatch')
        self._initialize_goal_runtime(bundle,manifest,*args,**kwargs)


def resolve_runtime(name):
    """Resolve explicit legacy or feedback runtime names; never guess dimensions."""
    if name=='neural_navigation.long_risk_runtime.LongRiskRuntime':
        from neural_navigation.long_risk_runtime import LongRiskRuntime
        return LongRiskRuntime
    if name=='neural_navigation.decision_feedback_runtime.StopFeedbackRuntime':
        from neural_navigation.decision_feedback_runtime import StopFeedbackRuntime
        return StopFeedbackRuntime
    return {'experiments.goal_pose_navigation.GoalPoseRuntime':GoalPoseRuntime,
            'experiments.docking_navigation.DockingRuntime':DockingRuntime}.get(name)
