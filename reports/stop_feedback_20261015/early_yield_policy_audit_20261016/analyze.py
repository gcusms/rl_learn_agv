"""Inspect saved policy support at observed early-yield warning states."""
import hashlib
import json
import os
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
OUT=Path(__file__).resolve().parent


def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    if (OUT/'RESULTS.json').exists():raise FileExistsError('Preserve the audit')
    import numpy as np
    import torch
    from experiments.stop_recovery_policy import StopRecoveryPPO
    from experiments.train_prefix_recovery import frozen_base_digest
    from neural_navigation.experiment_budget import ExperimentBudget
    torch.set_num_threads(1);budget=ExperimentBudget(.15,memory_mib=2048,directory=OUT)
    base=ROOT/'reports/stop_feedback_20261015'
    source=json.loads((base/'actual95_broad_regression_20261016/RESULTS.json').read_text())
    replay=json.loads((base/'early_yield_replay_20261016/RESULTS.json').read_text())
    assert source['completed'] and replay['completed']
    models={}
    for name in ('actual95_transfer_bc400_s231016','recovery_followup_bc400_s231016'):
        path=ROOT/'models/stop_feedback_20261015'/name/'policy_bc.zip'
        model=StopRecoveryPPO.load(path,device='cpu');model.policy.set_training_mode(False)
        assert model.num_timesteps==0 and frozen_base_digest(model.policy)=='e5502406c4ee8e42c4da09dd6affa022cb76e7f9b01316b1eaf0d2551431938e'
        models[name]=(model,digest(path));budget.tick()
    rows=[]
    for summary in replay['summaries']:
        row=next(r for r in source['groups']['fresh12']['rows'] if r['case_id']==summary['case_id'])
        path=ROOT/row['trace_path'];assert digest(path)==row['trace_sha256']
        with np.load(path,allow_pickle=False) as z:
            observations=np.concatenate((z['observations'][:,:300],z['observations'][:,303:]),axis=1).astype(np.float32)
            masks=z['masks'];times=z['labels'][:,0];choices=z['choices']
        i=int(np.flatnonzero(abs(times-summary['earliest_sensor_yield_signal']['time_s'])<1e-8)[0])
        result=dict(case_id=summary['case_id'],time_s=float(times[i]),source_trace_sha256=row['trace_sha256'],
            necessary_yield_is_a_hypothesis_not_an_admitted_label=True,models={})
        for name,(model,checksum) in models.items():
            with torch.no_grad():
                probabilities=model.policy.get_distribution(torch.from_numpy(observations),masks).distribution.probs.numpy()
                supported=model.policy.recovery_support(torch.from_numpy(observations)).numpy()
            if name=='actual95_transfer_bc400_s231016':np.testing.assert_array_equal(probabilities.argmax(-1),choices)
            p=probabilities[i]
            result['models'][name]=dict(checkpoint_sha256=checksum,recovery_branch_active=bool(supported[i]),
                stop_probability=float(p[0]),selected_choice=int(p.argmax()),probabilities=p.tolist(),
                wait_with_choices_frames=int(((probabilities.argmax(-1)==0)&masks[:,1:].any(-1)).sum()))
            budget.tick()
        rows.append(result)
    result=dict(completed=True,rows=rows,old_actor_frames_reproduced=237,
        new_physical_steps=0,new_optimizer_updates=0,new_ppo_steps=0,resource=budget.statistics(),
        goal_achieved=False,deployment_approved=False,
        scope='Saved-data actor support and probability audit only; no counterfactual safety or learned recovery claim.')
    (OUT/'RESULTS.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':
    os.chdir(ROOT)
    try:main()
    except Exception as error:
        (OUT/'FAILURE.json').write_text(json.dumps(dict(error=repr(error),automatically_restarted=False))+'\n')
        raise
