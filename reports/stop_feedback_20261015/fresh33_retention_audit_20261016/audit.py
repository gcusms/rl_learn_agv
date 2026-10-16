"""Check the efficiency-regressed task against its complete safe parent trace.

This is reused development data. Check current model decisions on the parent
history; do not confuse open-loop agreement with closed-loop recovery.
"""
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import torch

from experiments.recovery_demonstrations import convert_observations
from experiments.stop_recovery_policy import StopRecoveryPPO
from neural_navigation.experiment_budget import ExperimentBudget


ROOT=Path(__file__).resolve().parents[3]
OUT=Path(__file__).resolve().parent
BASE=ROOT/'reports/stop_feedback_20261015'
OLD=ROOT/'models/stop_feedback_20261015/early_transfer_bc800_s231015'
NEW=ROOT/'models/stop_feedback_20261015/actual95_transfer_bc400_s231016'


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    if (OUT/'PLAN.json').exists():raise FileExistsError('Fresh audit required')
    old_report=BASE/'actual_stop_validation_20261016/RESULTS_main.json'
    new_report=BASE/'actual95_broad_regression_20261016/RESULTS.json'
    old=json.loads(old_report.read_text());new=json.loads(new_report.read_text())
    assert old['completed'] and new['completed']
    before=next(r for r in old['groups']['fresh']['rows'] if r['case_id']=='localization_nav_33')
    after=next(r for r in new['groups']['fresh12']['rows'] if r['case_id']=='localization_nav_33')
    assert before['safe_success'] and after['safe_success']
    source=BASE/'early_transfer_verified_fresh_20261016/localization_nav_33_early_transfer_verified.npz'
    assert digest(source)==before['trace_sha256']
    paths=[Path(__file__),source,old_report,new_report,OLD/'policy_bc.zip',NEW/'policy_bc.zip',
           NEW/'demonstrations.npz',ROOT/'training/experiments/recovery_demonstrations.py']
    inputs={str(p.relative_to(ROOT)):digest(p) for p in paths}
    (OUT/'PLAN.json').write_text(json.dumps(dict(inputs=inputs,new_physical_steps=0,
        new_optimizer_updates=0,new_ppo_steps=0,old_source_complete_safe=True,
        source_selection='Only the development task that exceeded the declared per-task efficiency threshold.',
        no_unseen_data_claim=True),indent=2)+'\n')
    torch.set_num_threads(1)
    budget=ExperimentBudget(.15,memory_mib=2048,directory=OUT)
    with np.load(source,allow_pickle=False) as z:data={k:z[k].copy() for k in ['labels','observations','masks','commands','choices']}
    obs=convert_observations(data['observations'],data['masks'],data['commands'],data['labels'][:,0])
    parent=StopRecoveryPPO.load(OLD/'policy_bc.zip',device='cpu')
    candidate=StopRecoveryPPO.load(NEW/'policy_bc.zip',device='cpu')
    parent.policy.set_training_mode(False);candidate.policy.set_training_mode(False)
    targets=data['choices'].astype(np.int64);keep=[];predicted=[]
    for i,x in enumerate(obs):
        x=torch.from_numpy(x[None]);mask=data['masks'][i:i+1]
        with torch.no_grad():
            supported=bool(candidate.policy.recovery_support(x)[0])
            old_choice=int(parent.policy.get_distribution(x,mask).distribution.probs.argmax(-1)[0])
            new_choice=int(candidate.policy.get_distribution(x,mask).distribution.probs.argmax(-1)[0])
        if supported:assert old_choice==targets[i],(i,old_choice,int(targets[i]))
        keep.append(supported);predicted.append(new_choice);budget.tick()
    keep=np.asarray(keep,bool);predicted=np.asarray(predicted)
    mismatch=np.flatnonzero(keep&(predicted!=targets))
    with np.load(NEW/'demonstrations.npz',allow_pickle=False) as z:
        old_data={k:z[k].copy() for k in ['observations','masks','choices']}
    arrays={k:np.concatenate([old_data[k],a[keep]]) for k,a in
            [('observations',obs),('masks',data['masks']),('choices',targets)]}
    known={};conflicts=[]
    for i,(x,mask,label) in enumerate(zip(arrays['observations'],arrays['masks'],arrays['choices'])):
        key=hashlib.sha256(x.tobytes()+mask.tobytes()).digest()
        old_value=known.setdefault(key,(int(label),i))
        if old_value[0]!=int(label):conflicts.append([old_value[1],i])
    if not conflicts:np.savez_compressed(OUT/'demonstrations.npz',**arrays)
    result=dict(completed=True,at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
        parent_seconds=before['seconds'],candidate_seconds=after['seconds'],
        candidate_time_relative_increase=after['seconds']/before['seconds']-1.,
        supported_safe_parent_examples=int(keep.sum()),old_examples=len(old_data['choices']),
        old_prefix_bit_identical=True,old_stop_labels_preserved=int((old_data['choices']==0).sum()),
        disagreements=[dict(frame=int(i),time_s=float(data['labels'][i,0]),parent_choice=int(targets[i]),
            current_choice=int(predicted[i]),legal=np.flatnonzero(data['masks'][i]).tolist()) for i in mismatch],
        exact_input_conflicts=conflicts,source_admissible_for_separate_bounded_fitting=bool(keep.any() and len(mismatch) and not conflicts),
        both_source_and_regressed_task_are_reused_development=True,
        new_physical_steps=0,new_optimizer_updates=0,new_ppo_steps=0,resource=budget.statistics(),
        actual106_repair_not_claimed=True,goal_achieved=False,deployment_approved=False)
    for name,expected in inputs.items():assert digest(ROOT/name)==expected,name
    (OUT/'RESULTS.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    os.chdir(ROOT)
    try:main()
    except Exception as error:
        (OUT/'FAILURE.json').write_text(json.dumps(dict(error=repr(error),automatically_restarted=False))+'\n')
        raise
