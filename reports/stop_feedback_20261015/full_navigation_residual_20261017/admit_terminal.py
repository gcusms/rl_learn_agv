"""Verify identity migration and admit only a complete safe demonstration."""
from datetime import datetime
import hashlib
import json
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent
BASE=HERE.parent
SOURCE=ROOT/'models/stop_feedback_20261015/v2_behavior_retention_s231017/portable'


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def read(path):return json.loads(path.read_text())
def write(path,value):
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');temporary.replace(path)


def main():
    assert not (HERE/'ADMISSION_TERMINAL.json').exists()
    demo_dir=BASE/'sensor01_reference_terminal_demonstration_20261017'
    demo_result=read(demo_dir/'RESULTS.json')
    assert demo_result['completed'] and demo_result['full_safe_demonstration']
    row=demo_result['row']
    assert row['safe_success'] and not any(row[k] for k in ('collisions','false_arrival','physical_guards','timeout'))
    assert row['protocol_valid'] and row['seconds']<=140.
    demo_path=ROOT/row['trace_path'];assert digest(demo_path)==row['trace_sha256']
    old_dir=BASE/'v2_behavior_retention_20261017'
    old_data=read(old_dir/'DATA.json')
    old_archive=old_dir/'demonstrations.npz'
    assert digest(old_archive)==old_data['archive_sha256']
    validation=read(BASE/'v2_behavior_retention_validation_20261017/RESULTS.json')
    physical=[ROOT/r['trace_path'] for g in validation['groups'].values() for r in g['rows']]
    manifest=read(SOURCE/'manifest.json')
    paths=[Path(__file__),HERE/'actor.py',demo_dir/'RESULTS.json',demo_path,old_archive,
        old_dir/'DATA.json',SOURCE/'manifest.json']+[SOURCE/n for n in manifest['files']]+physical
    hashes={str(p.relative_to(ROOT)):digest(p) for p in set(paths)}
    write(HERE/'ADMISSION_TERMINAL_PLAN.json',dict(inputs=hashes,source_parent=str(SOURCE.relative_to(ROOT)),
        full_old4282_preserved=True,only_complete_safe_development_demonstration_with_protected_terminal_parent=True,
        no_training_or_physical_execution_in_this_audit=True,
        original_body4cm_actions_masks_deadlines_and_terminal_branch_unchanged=True,
        soft_cpu_cores=.35,rss_budget_mib=2800,hard_service_memory_mib=3072))
    import numpy as np
    import torch
    from actor import FullNavigationResidualActor,contract
    from neural_navigation.experiment_budget import ExperimentBudget
    from neural_navigation.masked_actions import masks_from_observation
    from experiments.native625_onpolicy import Native625OnPolicy
    torch.set_num_threads(1);torch.manual_seed(231017)
    budget=ExperimentBudget(.35,2800,HERE)
    parent=torch.jit.load(str(SOURCE/'actor.pt'),map_location='cpu').eval()
    actor=FullNavigationResidualActor(parent).eval()
    old=np.load(old_archive,allow_pickle=False);demo=np.load(demo_path,allow_pickle=False)
    assert len(old['actions'])==4282 and 100<=len(demo['choices'])<=250
    demo_count=len(demo['choices'])
    observations=np.concatenate([old['observations'],demo['observations']]).astype(np.float32)
    actions=np.r_[old['actions'],demo['choices']].astype(np.int64)
    masks=np.concatenate([old['masks'],demo['masks']]).astype(bool)
    np.testing.assert_array_equal(masks,masks_from_observation(observations[:,:282]))
    assert masks[np.arange(len(actions)),actions].all()
    labels={};conflicts=[]
    for i,(obs,label) in enumerate(zip(observations,actions)):
        previous=labels.setdefault(obs.tobytes(),(int(label),i))
        if previous[0]!=int(label):conflicts.append([previous[1],i])
    parity_frames=0;newly_trainable=0;outside_mismatches=[];parent_choices=[];supports=[]
    with torch.no_grad():
        for start in range(0,len(observations),32):
            x=torch.from_numpy(observations[start:start+32])
            before=parent(x);after=actor(x)
            np.testing.assert_array_equal(before.numpy(),after.numpy())
            choice=(before[:,None]-actor.actions[None]).abs().amax(2).argmin(1)
            valid=actor.support(x);previous_support=parent.support(x)
            newly_trainable+=int((valid&~previous_support).sum())
            parent_choices.extend(choice.tolist());supports.extend(valid.tolist())
            parity_frames+=len(x);budget.tick()
        for path in physical:
            with np.load(path,allow_pickle=False) as z:
                for start in range(0,len(z['observations']),32):
                    x=torch.from_numpy(z['observations'][start:start+32].copy())
                    np.testing.assert_array_equal(actor(x).numpy(),parent(x).numpy())
                    parity_frames+=len(x);budget.tick()
    parent_choices=np.asarray(parent_choices);supports=np.asarray(supports)
    outside_mismatches=np.flatnonzero((~supports)&(parent_choices!=actions)).tolist()
    changed_demo=(parent_choices[4282:]!=actions[4282:])
    old_parent_match=bool(np.array_equal(parent_choices[:4282],actions[:4282]))
    # Exercise the actual PPO context/likelihood path on newly controllable
    # recorded states, without an optimizer update.
    learner=Native625OnPolicy(actor,temperature=1.)
    probe=np.flatnonzero(changed_demo&supports[4282:])+4282
    assert len(probe)>0
    terms=learner.evaluate(observations[probe],masks[probe],actions[probe])
    assert bool(terms['eligible'].all()) and bool(torch.isfinite(terms['log_probability']).all())
    assert all(not p.requires_grad for p in actor.parent.parameters())
    admitted=not conflicts and not outside_mismatches and old_parent_match
    archive=HERE/'training_data_terminal.npz'
    if admitted:
        np.savez_compressed(archive,observations=observations,actions=actions,masks=masks,
            source_group=np.r_[np.zeros(4282,np.int64),np.ones(demo_count,np.int64)])
    for name,expected in hashes.items():assert digest(ROOT/name)==expected,name
    budget.tick()
    result=dict(completed=True,at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
        admitted=admitted,contract=contract(),identity_action_frames_checked=parity_frames,
        old_reference_rows=4282,complete_safe_demo_rows=demo_count,total_rows=len(actions),
        exact_input_label_conflicts=conflicts,outside_support_label_mismatches=outside_mismatches,
        old4282_parent_labels_exact=old_parent_match,newly_trainable_reference_states=newly_trainable,
        demo_changed_choices=int(changed_demo.sum()),
        demo_changed_choices_outside_new_support=int((changed_demo&~supports[4282:]).sum()),
        actual_native625_ppo_eligible_label_frames_checked=len(probe),
        action_masks_parent_terminal_and_invalid_behavior_preserved=True,
        archive_sha256=digest(archive) if admitted else None,
        no_training_updates=True,new_physical_episodes=0,new_ppo_steps=0,
        source_development_not_independent_test=True,resource=budget.statistics(),
        goal_achieved=False,deployment_approved=False)
    write(HERE/'ADMISSION_TERMINAL.json',result)
    print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__=='__main__':
    try:main()
    except Exception as error:
        write(HERE/'ADMISSION_TERMINAL_FAILURE.json',dict(error=repr(error),no_auto_restart=True));raise
