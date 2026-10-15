"""Actual checkpoint migration, likelihood and serialization checks."""
import json
import tempfile
from pathlib import Path
import numpy as np
import torch
from sb3_contrib import MaskablePPO
from experiments.recovery_temperature_policy import migrate_policy, RecoveryTemperaturePolicy


def identical(a,b):
    if isinstance(a,torch.Tensor):torch.testing.assert_close(a,b,rtol=0,atol=0)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for key in a:identical(a[key],b[key])
    elif isinstance(a,(list,tuple)):
        assert len(a)==len(b)
        for x,y in zip(a,b):identical(x,y)
    else:assert a==b


def main():
    root=Path(__file__).resolve().parents[2]
    checkpoint=root/'training/runs/safety_context_20261011/quality_recovery_4000/candidate/policy_2000.zip'
    torch.set_num_threads(1)
    model=MaskablePPO.load(checkpoint,device='cpu')
    weights={k:v.clone() for k,v in model.policy.state_dict().items()}
    optimizer=model.policy.optimizer.state_dict()
    rng=np.random.default_rng(91)
    observations=rng.normal(size=(128,300)).astype(np.float32)
    masks=rng.random((128,12))>.7;masks[:,0]=True
    before=model.predict(observations,action_masks=masks,deterministic=True)[0]
    state=torch.get_rng_state().clone()
    migrate_policy(model)
    identical(weights,model.policy.state_dict());identical(optimizer,model.policy.optimizer.state_dict())
    identical(state,torch.get_rng_state())
    np.testing.assert_array_equal(before,model.predict(observations,action_masks=masks,deterministic=True)[0])
    obs=torch.from_numpy(observations)
    actions,_,old_log=model.policy(obs,action_masks=masks)
    _,new_log,_=model.policy.evaluate_actions(obs,actions,action_masks=masks)
    torch.testing.assert_close(old_log,new_log,atol=1e-6,rtol=0)
    with tempfile.TemporaryDirectory() as temporary:
        path=Path(temporary)/'policy.zip';model.save(path)
        restored=MaskablePPO.load(path,device='cpu')
        assert isinstance(restored.policy,RecoveryTemperaturePolicy)
        assert restored.policy.recovery_temperature==3.
        identical(weights,restored.policy.state_dict());identical(optimizer,restored.policy.optimizer.state_dict())
        np.testing.assert_array_equal(before,restored.predict(observations,action_masks=masks,deterministic=True)[0])
    print(json.dumps(dict(passed=True,checkpoint=str(checkpoint),inputs=128,
        exact_weights=True,exact_adam=True,rng_preserved=True,greedy_parity=True,
        rollout_update_log_probability_parity=True,save_reload=True,
        limitation='Compatibility checks; recovery behavior remains unproven.')))


if __name__=='__main__':main()
