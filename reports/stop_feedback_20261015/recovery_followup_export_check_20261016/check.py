"""Independently load saved weights, optimizer and representative export calls."""
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import torch

from experiments.stop_recovery_policy import StopRecoveryPPO
from experiments.train_prefix_recovery import frozen_base_digest
from neural_navigation.experiment_budget import ExperimentBudget
from neural_navigation.masked_actions import ACTIONS


ROOT=Path(__file__).resolve().parents[3]
OUT=Path(__file__).resolve().parent
MODEL=ROOT/'models/stop_feedback_20261015/recovery_followup_bc400_s231016'
PARENT=ROOT/'models/stop_feedback_20261015/actual95_transfer_bc400_s231016'


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    if (OUT/'RESULTS.json').exists():raise FileExistsError('Do not overwrite export verification')
    torch.set_num_threads(1)
    budget=ExperimentBudget(.15,memory_mib=2048,directory=OUT)
    manifest=json.loads((MODEL/'bc_portable/manifest.json').read_text())
    for name,expected in manifest['files'].items():assert digest(MODEL/'bc_portable'/name)==expected,name
    final=json.loads((MODEL/'RESULTS.json').read_text());assert final['completed'] and final['exported_for_bounded_evaluation']
    assert digest(MODEL/'policy_bc.zip')==final['checkpoint_sha256']
    model=StopRecoveryPPO.load(MODEL/'policy_bc.zip',device='cpu');model.policy.set_training_mode(False)
    assert model.num_timesteps==0
    assert frozen_base_digest(model.policy)=='e5502406c4ee8e42c4da09dd6affa022cb76e7f9b01316b1eaf0d2551431938e'
    with np.load(MODEL/'demonstrations.npz',allow_pickle=False) as z:data={k:z[k].copy() for k in z.files}
    with np.load(PARENT/'demonstrations.npz',allow_pickle=False) as z:
        for key in ['observations','masks','choices']:np.testing.assert_array_equal(data[key][:2421],z[key])
    with torch.no_grad():
        distribution=model.policy.get_distribution(torch.from_numpy(data['observations']),data['masks']).distribution
        choices=distribution.probs.argmax(-1).numpy()
    np.testing.assert_array_equal(choices,data['choices'])
    assert int((data['choices'][:2421]==0).sum())==57
    budget.tick()
    continuation=torch.load(MODEL/'continuation_state.pt',map_location='cpu',weights_only=False)
    model.policy.optimizer.load_state_dict(continuation['optimizer'])
    assert continuation['torch_rng'].dtype==torch.uint8 and continuation['numpy_rng'][0]=='MT19937'
    actor=torch.jit.load(str(MODEL/'bc_portable/actor.pt'),map_location='cpu').eval()
    selected=np.unique(np.r_[np.flatnonzero(data['choices']==0),np.arange(2421,len(data['choices'])),
                             np.linspace(0,2420,128,dtype=int)])
    for i in selected:
        x=data['observations'][i]
        runtime=np.r_[x[:300],np.zeros(3,np.float32),x[300:]].astype(np.float32)
        assert runtime.shape==(439,)
        with torch.no_grad():predicted=actor(torch.from_numpy(runtime[None])).numpy()[0]
        np.testing.assert_array_equal(predicted,ACTIONS[data['choices'][i]])
        budget.tick()
    result=dict(completed=True,at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
        checkpoint_sha256=digest(MODEL/'policy_bc.zip'),all_saved_examples_correct=len(choices),
        old2421_inputs_masks_and_labels_bit_identical=True,old_stop_labels_retained=57,
        optimizer_and_rng_loaded=True,frozen_base_unchanged=True,
        exported_decisions_checked=len(selected),all_stops_and_all84_new_targets_in_export_check=True,
        new_physical_steps=0,new_optimizer_updates=0,new_ppo_steps=0,resource=budget.statistics(),
        actual106_repair_not_claimed=True,goal_achieved=False,deployment_approved=False)
    (OUT/'RESULTS.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    os.chdir(ROOT)
    try:main()
    except Exception as error:
        (OUT/'FAILURE.json').write_text(json.dumps(dict(error=repr(error),automatically_restarted=False))+'\n')
        raise
