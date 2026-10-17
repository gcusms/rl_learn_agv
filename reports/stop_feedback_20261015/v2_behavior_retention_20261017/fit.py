"""Bounded auxiliary fit retaining the v2 PPO's complete safe gain reference.

One changed factor: successful-reference coverage. Guard, observations,
hardware, actions and frozen navigation components remain byte-identical.
"""
import copy
from datetime import datetime
import hashlib
import json
from pathlib import Path
import random
import shutil
from zoneinfo import ZoneInfo


ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent
SOURCE=ROOT/'models/stop_feedback_20261015/guard_aligned_mixed_ppo_resume_s231017'
MODEL=ROOT/'models/stop_feedback_20261015/v2_behavior_retention_s231017'


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path,value):
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');tmp.replace(path)


def main():
    assert not (HERE/'FIT_RESULTS.json').exists() and not MODEL.exists()
    paths=[Path(__file__),HERE/'DATA.json',HERE/'demonstrations.npz',SOURCE/'checkpoint.pt',SOURCE/'parent_actor.pt']
    for folder in ['experiments','neural_navigation','navrl']:
        paths+=list((ROOT/'training'/folder).rglob('*.py'))
    manifest=json.loads((SOURCE/'portable/manifest.json').read_text())
    paths+=[SOURCE/'portable/manifest.json']+[SOURCE/'portable'/name for name in manifest['files']]
    hashes={str(p.relative_to(ROOT)):digest(p) for p in set(paths)}
    write(HERE/'FIT_PLAN.json',dict(at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),inputs=hashes,
        source_model=str(SOURCE.relative_to(ROOT)),seed=231017,
        only_changed_factor='complete_safe_development_reference_coverage',
        maximum_accepted_supervised_updates=1200,maximum_proposed_updates=1800,learning_rate=.00015,
        loss='masked_cross_entropy_over_all_supported_reference_rows',
        original557_stop_labels_checked_after_every_proposed_update=True,
        reject_and_backtrack_optimizer_and_head_if_original_stop_lost=True,
        complete_all4282_reference_labels_required_before_physics=True,
        no_new_ppo_steps=True,source_prior_additional_ppo_decisions=1993,
        unchanged_guard_memory_footprint4cm_actions_dynamics_deadlines_and_observation_semantics=True,
        frozen_components=['parent_actor','localizer','router','directional_docking'],
        reused_dynamic00_and_dynamic01_are_now_training_development_cases=True,
        full_original_regression_safety_efficiency_late106_and_fresh_seed_gates_preserved=True,
        no_automatic_promotion=True,soft_cpu_cores=.35,memory_hard_max_mib=3072,
        goal_achieved=False,deployment_approved=False))
    import gc
    import numpy as np
    import torch
    import torch.nn.functional as F
    from experiments.guard_reason_release_actor import GuardReasonReleaseActor
    from experiments.train_long_risk_actor import state_digest
    from neural_navigation.aligned_release_runtime import AlignedReleaseRuntime, contract
    from neural_navigation.experiment_budget import ExperimentBudget
    from neural_navigation.goal_pose_runtime import check_files
    torch.set_num_threads(1);torch.manual_seed(231017);np.random.seed(231017);random.seed(231017)
    budget=ExperimentBudget(.35,2800,HERE)
    data=json.loads((HERE/'DATA.json').read_text());assert data['admitted_rows']==4282
    assert digest(HERE/'demonstrations.npz')==data['archive_sha256']
    with np.load(HERE/'demonstrations.npz',allow_pickle=False) as z:
        x=torch.from_numpy(z['observations'].copy());target=torch.from_numpy(z['actions'].astype(np.int64))
        legal=torch.from_numpy(z['masks'].copy());source_groups=z['source_group'].copy()
    actor=GuardReasonReleaseActor(torch.jit.load(str(SOURCE/'parent_actor.pt'),map_location='cpu')).eval()
    checkpoint=torch.load(SOURCE/'checkpoint.pt',map_location='cpu',weights_only=False)
    actor.load_state_dict(checkpoint['actor']);del checkpoint;gc.collect()
    frozen_hash=state_digest(actor.parent)
    features=[];supports=[];baselines=[]
    with torch.no_grad():
        for start in range(0,len(x),32):
            a,b=actor.features(x[start:start+32]);features.append(a.detach());baselines.append(b.detach())
            supports.append(actor.support(x[start:start+32]));budget.tick()
    features=torch.cat(features);support=torch.cat(supports);baseline=torch.cat(baselines)
    nominal=(baseline[:,None]-actor.actions[None]).abs().amax(2).argmin(1)
    assert torch.equal(actor.masks(x),legal)
    assert legal[torch.arange(len(x)),target].all()
    assert not ((~support)&(nominal!=target)).any(),'Frozen parent mismatch; do not fit unsupported references'
    anchor_stop=(target==0)&(torch.arange(len(x))<3799)
    assert int(anchor_stop.sum())==557
    optimizer=torch.optim.Adam(actor.head.parameters(),lr=.00015)

    @torch.no_grad()
    def metrics():
        logits=(actor.head(features)+2.*features[:,76:88]).masked_fill(~legal,-1e8)
        selected=torch.where(support,logits.argmax(1),nominal)
        correct=selected==target
        alternatives=logits.clone();alternatives.scatter_(1,target[:,None],-1e8)
        margin=logits.gather(1,target[:,None]).flatten()-alternatives.amax(1)
        return dict(correct_labels=int(correct.sum()),total_labels=len(x),
            original_stop_correct=int(correct[anchor_stop].sum()),original_stop_total=557,
            all_stop_correct=int(correct[target==0].sum()),all_stop_total=int((target==0).sum()),
            outside_support_exact=bool(correct[~support].all()),
            minimum_supported_label_margin=float(margin[support].min()),
            groups={str(g):dict(total=int((source_groups==g).sum()),correct=int(correct[torch.from_numpy(source_groups==g)].sum()))
                for g in np.unique(source_groups)})

    def stop_retained():
        with torch.no_grad():
            selected=(actor.head(features[anchor_stop])+2.*features[anchor_stop,76:88]).masked_fill(~legal[anchor_stop],-1e8).argmax(1)
            selected=torch.where(support[anchor_stop],selected,nominal[anchor_stop])
            return bool((selected==0).all())

    initial=metrics();assert initial['original_stop_correct']==557
    accepted=rejected=0;history=[dict(accepted_updates=0,**initial)]
    for proposed in range(1,1801):
        previous_head=copy.deepcopy(actor.head.state_dict());previous_optimizer=copy.deepcopy(optimizer.state_dict())
        logits=(actor.head(features[support])+2.*features[support,76:88]).masked_fill(~legal[support],-1e8)
        loss=F.cross_entropy(logits,target[support])
        optimizer.zero_grad(set_to_none=True);loss.backward()
        torch.nn.utils.clip_grad_norm_(actor.head.parameters(),1.);optimizer.step()
        if not stop_retained():
            actor.head.load_state_dict(previous_head);optimizer.load_state_dict(previous_optimizer)
            for group in optimizer.param_groups:group['lr']*=.5
            rejected+=1
            if optimizer.param_groups[0]['lr']<1e-8:break
        else:accepted+=1
        if proposed%5==0:budget.tick()
        if proposed%20==0 or accepted>=1200:
            row=dict(accepted_updates=accepted,proposed_updates=proposed,rejected_updates=rejected,loss=float(loss.detach()),**metrics())
            history.append(row);write(HERE/'FIT_STATUS.json',dict(status='auxiliary_reference_fit',**row))
            if row['correct_labels']==len(x) and row['minimum_supported_label_margin']>=.01:break
        if accepted>=1200:break
    final=metrics();fit_passed=final['correct_labels']==len(x) and final['original_stop_correct']==557
    assert state_digest(actor.parent)==frozen_hash
    MODEL.mkdir(exist_ok=False,parents=True);shutil.copy2(SOURCE/'parent_actor.pt',MODEL/'parent_actor.pt')
    torch.save(dict(actor=actor.state_dict(),optimizer=optimizer.state_dict(),torch_rng=torch.get_rng_state(),
        numpy_rng=np.random.get_state(),python_rng=random.getstate(),source_checkpoint_sha256=digest(SOURCE/'checkpoint.pt'),
        data_sha256=data['archive_sha256'],contract=contract(),new_supervised_updates=accepted,
        source_prior_additional_ppo_steps=1993,new_ppo_steps=0,exact_physical_rollout_resume=False,
        deployment_approved=False),MODEL/'checkpoint.pt')
    export_passed=runtime_admission=False
    if fit_passed:
        bundle=MODEL/'portable';shutil.copytree(SOURCE/'portable',bundle)
        sample=x[::max(1,len(x)//64)][:64]
        traced=torch.jit.trace_module(actor,{'forward':(sample,),'logits':(sample,),'support':(sample,)})
        traced.save(str(bundle/'actor.pt'))
        with torch.no_grad():
            for start in range(0,len(x),32):
                chunk=x[start:start+32]
                np.testing.assert_array_equal(traced(chunk).numpy(),actor(chunk).numpy())
                expected=actor.actions[target[start:start+32]]
                np.testing.assert_array_equal(actor(chunk).numpy(),expected.numpy())
                budget.tick()
        export_passed=True
        preservation=json.loads((bundle/'navigation_preservation.json').read_text())
        preservation.update(auxiliary_safe_reference_fit=dict(source=str(SOURCE.relative_to(ROOT)),
            rows=len(x),new_supervised_updates=accepted,new_ppo_steps=0,source_prior_additional_ppo_steps=1993,
            original557_stops_retained=True,original3799_rows_preserved=True,
            complete_reference_gain_dynamic00_retained=True,known_dynamic01_reference_added=True,
            runtime_physics_gates_not_yet_passed=True))
        write(bundle/'navigation_preservation.json',preservation)
        new_manifest=copy.deepcopy(manifest);new_manifest.update(deployment_approved=False,
            navigation_preservation_contract=preservation,additional_ppo_steps=1993,
            parent_manifest_sha256=digest(SOURCE/'portable/manifest.json'))
        for name in ['actor.pt','navigation_preservation.json']:new_manifest['files'][name]=digest(bundle/name)
        write(bundle/'manifest.json',new_manifest)
        write(bundle/'MODEL_CARD.json',dict(method='auxiliary_complete_safe_reference_fit_after_v2_PPO',
            new_ppo_steps=0,source_prior_additional_ppo_steps=1993,new_supervised_updates=accepted,
            deployment_approved=False,physics_gates_not_yet_passed=True,
            normal_gain_and_failure_cases_are_development_not_unseen_tests=True))
        check_files(bundle,new_manifest)
        for name in manifest['files']:
            if name not in ['actor.pt','navigation_preservation.json']:assert digest(bundle/name)==digest(SOURCE/'portable'/name),name
        del traced;gc.collect()
        runtime=AlignedReleaseRuntime(bundle,ROOT/'training/assets/room_navigation_scripted/decision_map.yaml',
            [6.225,2.725,0.],[7.125,5.725],'shortcut')
        assert runtime.guard.predictive_constraint.config==json.loads((SOURCE/'portable/hardware.json').read_text())['near_field_memory']['predictive_action_constraint']
        runtime_admission=True;del runtime;gc.collect();budget.tick()
    for name,expected in hashes.items():assert digest(ROOT/name)==expected,name
    result=dict(completed=True,at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),training_started=True,
        fit_passed=fit_passed,export_parity_passed=export_passed,strict_runtime_admission_passed=runtime_admission,
        proposed_updates=proposed,accepted_supervised_updates=accepted,rejected_supervised_updates=rejected,
        initial=initial,final=final,history=history,original557_stop_labels_retained=True,
        model=str(MODEL.relative_to(ROOT)),checkpoint_sha256=digest(MODEL/'checkpoint.pt'),
        new_ppo_steps=0,source_prior_additional_ppo_steps=1993,new_physical_episodes=0,
        unchanged_guard_memory_observation_actions_footprint_and_frozen_components=True,
        inputs=hashes,resource=budget.statistics(),goal_achieved=False,deployment_approved=False)
    write(HERE/'FIT_RESULTS.json',result);write(MODEL/'RESULTS.json',result)
    write(HERE/'FIT_STATUS.json',dict(status='complete',fit_passed=fit_passed,new_supervised_updates=accepted))
    print(json.dumps({k:result[k] for k in ['fit_passed','accepted_supervised_updates','initial','final','resource']},ensure_ascii=False),flush=True)


if __name__=='__main__':
    try:main()
    except Exception as error:
        write(HERE/'FIT_FAILURE.json',dict(error=repr(error),no_auto_restart=True));raise
