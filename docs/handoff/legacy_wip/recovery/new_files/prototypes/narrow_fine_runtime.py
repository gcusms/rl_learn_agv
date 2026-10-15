"""Opt-in portable 20-action runtime, with the exact training sensor interface."""
import hashlib,json
from pathlib import Path
import numpy as np
import torch
from neural_navigation.runtime import execute_policy_action
from neural_navigation.quality_prediction_context import QualityPredictionContext
from prototypes.narrow_alignment import AlignmentNarrowRuntime
from prototypes.narrow_observation import augment,masks,ACTIONS,contract

class FineSensorRuntime(AlignmentNarrowRuntime):
    def __init__(self,bundle,*args,**kwargs):
        bundle=Path(bundle)
        if json.loads((bundle/'fine_observation_action_contract.json').read_text())!=contract():raise ValueError('Fine interface contract mismatch')
        super().__init__(bundle,*args,**kwargs)
        self.fine_actor=torch.jit.load(str(bundle/'actor.pt'),map_location='cpu').eval()
        self.fine_quality=QualityPredictionContext(self.guard.half*2)
    def step(self,scan,odom,velocity,timestamp,received_timestamp=None,scan_valid=True,rear_clearance_known=False,capture_observation=False):
        if rear_clearance_known:raise ValueError('No rear/reverse fine commands supported')
        base,diagnostics=self.core.update(scan,odom,velocity,timestamp)
        prefix=self.context.augment(base,self.core,self.guard,timestamp,received_timestamp=received_timestamp,scan_valid=scan_valid)
        observation=augment(self.fine_quality.augment(prefix,self.core),self.core,self.guard,timestamp,received_timestamp=received_timestamp,scan_valid=scan_valid)
        with torch.no_grad():proposal=self.fine_actor(torch.from_numpy(observation)[None])[0].numpy()
        distances=np.max(np.abs(ACTIONS-proposal),axis=1);index=int(distances.argmin())
        if distances[index]>2e-6 or not masks(observation)[index]:raise ValueError('Unknown or rejected fine actor command')
        result=execute_policy_action(self.core,self.guard,proposal,timestamp,received_timestamp,scan_valid)
        self.context.record(result);result['fine_choice']=index;result['diagnostics']=diagnostics
        result['policy_observation_dimensions']=318
        if capture_observation:result['policy_observation']=observation.tolist()
        return result
