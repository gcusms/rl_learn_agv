"""Opt-in quality actor runtime; retains the existing sensor core and guard."""
import hashlib
import json
from pathlib import Path
from neural_navigation.runtime import SensorNavigationRuntime
from neural_navigation.quality_prediction_context import QualityPredictionContext,contract,TOTAL_DIM


class QualityPolicyInput:
    def __init__(self,policy,core,prediction):
        self.policy=policy;self.core=core;self.prediction=prediction
        self.last_observation=None

    def predict(self,prefix):
        observation=self.prediction.augment(prefix,self.core)
        self.last_observation=observation.copy()
        return self.policy.predict(observation)


class QualitySensorNavigationRuntime(SensorNavigationRuntime):
    sensor_core_variants=('native',)
    def __init__(self,bundle,*args,**kwargs):
        bundle=Path(bundle)
        if json.loads((bundle/'quality_observation.json').read_text())!=contract():
            raise ValueError('Quality observation contract mismatch')
        manifest=json.loads((bundle/'manifest.json').read_text())
        if manifest.get('sensor_core','native') not in self.sensor_core_variants:
            raise ValueError('Bundle requires a different sensor core runtime')
        if manifest['observation_contract']!=contract() or not manifest['requires_quality_feature_generator']:
            raise ValueError('Quality bundle manifest required')
        for name,expected in manifest['files'].items():
            if Path(name).name!=name or hashlib.sha256((bundle/name).read_bytes()).hexdigest()!=expected:
                raise ValueError('Bundle hash or filename mismatch')
        super().__init__(bundle,*args,**kwargs)
        if self.policy.observation_dimensions!=282:raise ValueError('Native V4 prefix required')
        prediction=QualityPredictionContext(self.guard.half*2)
        self.policy.observation_dimensions=TOTAL_DIM
        self.policy=QualityPolicyInput(self.policy,self.core,prediction)

    def step(self,*args,**kwargs):
        result=super().step(*args,**kwargs)
        actual=self.policy.last_observation
        result['policy_observation_dimensions']=len(actual)
        result['quality_prediction_features']=actual[282:].tolist()
        if 'policy_observation' in result:result['policy_observation']=actual.tolist()
        return result
