"""Bounded residual pretraining from a verified complete safe demonstration."""
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
SOURCE=ROOT/'models/stop_feedback_20261015/v2_behavior_retention_s231017/portable'
RESUME=ROOT/'models/stop_feedback_20261015/full_navigation_margin_s231017'
MODEL=ROOT/'models/stop_feedback_20261015/full_navigation_margin_resume_s231017'


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def read(path):return json.loads(path.read_text())
def write(path,value):
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');temporary.replace(path)


def main():
    assert not MODEL.exists() and not (HERE/'MARGIN_RESUME_FIT_RESULTS.json').exists()
    admission=read(HERE/'ADMISSION_TERMINAL.json')
    assert admission['completed'] and admission['admitted']
    archive=HERE/'training_data_terminal.npz';assert digest(archive)==admission['archive_sha256']
    manifest=read(SOURCE/'manifest.json')
    paths=[Path(__file__),HERE/'actor.py',HERE/'ADMISSION_TERMINAL.json',archive,SOURCE/'manifest.json',RESUME/'checkpoint.pt',RESUME/'RESULTS.json']
    paths += [SOURCE/name for name in manifest['files']]
    hashes={str(p.relative_to(ROOT)):digest(p) for p in set(paths)}
    write(HERE/'MARGIN_RESUME_FIT_PLAN.json',dict(at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),inputs=hashes,
        source_parent=str(SOURCE.relative_to(ROOT)),model=str(MODEL.relative_to(ROOT)),seed=231017,
        maximum_accepted_updates=3600,maximum_proposed_updates=4500,source_accepted_updates=1200,learning_rate=.00015,
        loss='new_demo_cross_entropy_plus_old_reference_margin_hinge',old_margin=.2,old_hinge_weight=4.,original_all658_stop_labels_checked_after_each_update=True,
        reject_and_restore_optimizer_and_head_if_old_stop_lost=True,
        require_all_reference_labels_before_physics=True,
        ordinary_and_recovery_navigation_head_trainable=True,parent_and_terminal_docking_frozen=True,
        same625_observation_masks_actions_body4cm_memory_predictor_route_dynamics_and_deadline=True,
        new_ppo_steps=0,no_automatic_promotion=True,
        original_fixed28_known_safety_hard97_6_106_and_conditional_fresh8_gates_preserved=True,
        soft_cpu_cores=.35,rss_budget_mib=2800,hard_service_memory_mib=3072,
        goal_achieved=False,deployment_approved=False))
    import numpy as np
    import torch
    import torch.nn.functional as F
    from actor import FullNavigationResidualActor,contract
    from neural_navigation.aligned_release_runtime import AlignedReleaseRuntime
    from neural_navigation.experiment_budget import ExperimentBudget
    from neural_navigation.goal_pose_runtime import check_files
    torch.set_num_threads(1);torch.manual_seed(231017);np.random.seed(231017);random.seed(231017)
    budget=ExperimentBudget(.35,2800,HERE)
    data=np.load(archive,allow_pickle=False)
    x=torch.from_numpy(data['observations'].copy());target=torch.from_numpy(data['actions'].copy())
    legal=torch.from_numpy(data['masks'].copy());groups=torch.from_numpy(data['source_group'].copy())
    actor=FullNavigationResidualActor(torch.jit.load(str(SOURCE/'actor.pt'),map_location='cpu')).eval()
    resumed=torch.load(RESUME/'checkpoint.pt',map_location='cpu',weights_only=False)
    assert resumed['data_sha256']==admission['archive_sha256'] and resumed['new_supervised_updates']==1200
    actor.load_state_dict(resumed['actor'])
    parent_state={name:value.detach().clone() for name,value in actor.parent.state_dict().items()}
    features=[];supports=[];baselines=[]
    with torch.no_grad():
        for start in range(0,len(x),32):
            f,b=actor.features(x[start:start+32]);features.append(f);baselines.append(b)
            supports.append(actor.support(x[start:start+32]));budget.tick()
    features=torch.cat(features);baseline=torch.cat(baselines);support=torch.cat(supports)
    nominal=(baseline[:,None]-actor.actions[None]).abs().amax(2).argmin(1)
    assert torch.equal(actor.masks(x),legal)
    assert legal[torch.arange(len(x)),target].all()
    assert not ((~support)&(nominal!=target)).any()
    stop_anchor=(groups==0)&(target==0);assert int(stop_anchor.sum())==658
    supported_groups=groups[support]
    old_rows=supported_groups==0;new_rows=supported_groups==1
    optimizer=torch.optim.Adam(actor.head.parameters(),lr=.00015)
    optimizer.load_state_dict(resumed['optimizer'])
    torch.set_rng_state(resumed['torch_rng']);np.random.set_state(resumed['numpy_rng']);random.setstate(resumed['python_rng'])

    @torch.no_grad()
    def metrics():
        logits=(actor.head(features)+2.*features[:,76:88]).masked_fill(~legal,-1e8)
        selected=torch.where(support,logits.argmax(1),nominal)
        correct=selected==target
        alternatives=logits.clone();alternatives.scatter_(1,target[:,None],-1e8)
        margin=logits.gather(1,target[:,None]).flatten()-alternatives.amax(1)
        return dict(correct_labels=int(correct.sum()),total_labels=len(x),
            old4282_correct=int(correct[groups==0].sum()),teacher_correct=int(correct[groups==1].sum()),
            protected_stop_correct=int(correct[stop_anchor].sum()),protected_stop_total=658,
            all_stop_correct=int(correct[target==0].sum()),all_stop_total=int((target==0).sum()),
            minimum_supported_margin=float(margin[support].min()),outside_support_parent_exact=bool(correct[~support].all()))

    def retained():
        with torch.no_grad():
            selected=(actor.head(features[stop_anchor])+2.*features[stop_anchor,76:88]).masked_fill(~legal[stop_anchor],-1e8).argmax(1)
            return bool((torch.where(support[stop_anchor],selected,nominal[stop_anchor])==0).all())

    initial=metrics();assert initial['correct_labels']==4446 and initial['protected_stop_correct']==658
    accepted=rejected=0;history=[dict(accepted_updates=0,**initial)]
    proposed=0
    for proposed in range(1,4501):
        previous_head=copy.deepcopy(actor.head.state_dict());previous_optimizer=copy.deepcopy(optimizer.state_dict())
        logits=(actor.head(features[support])+2.*features[support,76:88]).masked_fill(~legal[support],-1e8)
        selected_targets=target[support]
        teacher_loss=F.cross_entropy(logits[new_rows],selected_targets[new_rows])
        old_logits=logits[old_rows]
        true_old=old_logits.gather(1,selected_targets[old_rows,None]).flatten()
        alternatives=old_logits.scatter(1,selected_targets[old_rows,None],-1e8).amax(1)
        retention_loss=F.relu(.2-(true_old-alternatives)).mean()
        loss=teacher_loss+4.*retention_loss
        optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(actor.head.parameters(),1.)
        optimizer.step()
        if not retained():
            actor.head.load_state_dict(previous_head);optimizer.load_state_dict(previous_optimizer)
            for group in optimizer.param_groups:group['lr']*=.5
            rejected+=1
            if optimizer.param_groups[0]['lr']<1e-8:break
        else:accepted+=1
        if proposed%5==0:budget.tick()
        if proposed%20==0 or accepted>=3600:
            row=dict(accepted_updates=accepted,proposed_updates=proposed,rejected_updates=rejected,
                loss=float(loss.detach()),**metrics())
            history.append(row);write(HERE/'MARGIN_RESUME_FIT_STATUS.json',dict(status='residual_supervised_fit',**row))
            if row['correct_labels']==len(x) and row['minimum_supported_margin']>=.01:break
        if accepted>=3600:break
    final=metrics();fit_passed=final['correct_labels']==len(x) and final['protected_stop_correct']==658
    for name,value in actor.parent.state_dict().items():assert torch.equal(value,parent_state[name]),name
    MODEL.mkdir(parents=True,exist_ok=False)
    shutil.copy2(SOURCE/'actor.pt',MODEL/'parent625_actor.pt')
    shutil.copy2(HERE/'actor.py',MODEL/'training_actor.py')
    torch.save(dict(actor=actor.state_dict(),optimizer=optimizer.state_dict(),torch_rng=torch.get_rng_state(),
        numpy_rng=np.random.get_state(),python_rng=random.getstate(),actor_contract=contract(),
        training_parent_file='parent625_actor.pt',parent_sha256=digest(SOURCE/'actor.pt'),
        data_sha256=admission['archive_sha256'],new_supervised_updates=accepted,source_supervised_updates=1200,cumulative_supervised_updates=1200+accepted,new_ppo_steps=0,
        exact_physical_rollout_resume=False,deployment_approved=False),MODEL/'checkpoint.pt')
    export_passed=runtime_passed=False
    if fit_passed:
        bundle=MODEL/'portable';shutil.copytree(SOURCE,bundle)
        sample=x[::max(1,len(x)//64)][:64]
        traced=torch.jit.trace_module(actor,{'forward':(sample,),'logits':(sample,),'support':(sample,)})
        traced.save(str(bundle/'actor.pt'))
        with torch.no_grad():
            for start in range(0,len(x),32):
                chunk=x[start:start+32]
                np.testing.assert_array_equal(traced(chunk).numpy(),actor.actions[target[start:start+32]].numpy())
                budget.tick()
        export_passed=True
        preservation=read(bundle/'navigation_preservation.json')
        preservation.update(schema='full_navigation_residual_v1',learning_support=contract(),
            full625_parent_sha256=digest(SOURCE/'actor.pt'),old4282_labels_retained=True,
            original658_stops_retained=True,new_supervised_updates=accepted,new_ppo_steps=0,
            full_safe_teacher_case='pre_ppo_holdout_sensor_01',not_independent_generalization=True,
            physical_validation_not_yet_passed=True)
        write(bundle/'navigation_preservation.json',preservation)
        write(bundle/'full_navigation_training_contract.json',contract())
        new_manifest=copy.deepcopy(manifest)
        new_manifest.update(deployment_approved=False,navigation_preservation_contract=preservation,
            parent_manifest_sha256=digest(SOURCE/'manifest.json'))
        for name in ('actor.pt','navigation_preservation.json','full_navigation_training_contract.json'):
            new_manifest['files'][name]=digest(bundle/name)
        write(bundle/'manifest.json',new_manifest)
        write(bundle/'MODEL_CARD.json',dict(method='full_navigation_residual_supervised_initialization',
            training_contract=contract(),new_ppo_steps=0,new_supervised_updates=accepted,
            deployment_approved=False,physical_validation_not_yet_passed=True))
        check_files(bundle,new_manifest)
        for name in manifest['files']:
            if name not in ('actor.pt','navigation_preservation.json'):assert digest(bundle/name)==digest(SOURCE/name),name
        runtime=AlignedReleaseRuntime(bundle,ROOT/'training/assets/room_navigation_scripted/decision_map.yaml',
            [6.225,2.725,0.],[7.125,5.725],'shortcut')
        assert runtime.policy.policy.observation_dimensions==625
        runtime_passed=True;del runtime;budget.tick()
    for name,expected in hashes.items():assert digest(ROOT/name)==expected,name
    result=dict(completed=True,at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),fit_passed=fit_passed,
        initial=initial,final=final,accepted_supervised_updates=accepted,source_supervised_updates=1200,cumulative_supervised_updates=1200+accepted,rejected_updates=rejected,
        proposed_updates=proposed,history=history,export_parity_passed=export_passed,
        strict_runtime_admission_passed=runtime_passed,model=str(MODEL.relative_to(ROOT)),
        checkpoint_sha256=digest(MODEL/'checkpoint.pt'),new_ppo_steps=0,new_physical_episodes=0,
        parent_and_terminal_pose_actor_frozen=True,original_guard_body4cm_actions_and_map_components_unchanged=True,
        resource=budget.statistics(),goal_achieved=False,deployment_approved=False)
    write(HERE/'MARGIN_RESUME_FIT_RESULTS.json',result);write(MODEL/'RESULTS.json',result)
    write(HERE/'MARGIN_RESUME_FIT_STATUS.json',dict(status='complete',fit_passed=fit_passed,new_supervised_updates=accepted))
    print(json.dumps({k:v for k,v in result.items() if k!='history'},ensure_ascii=False),flush=True)


if __name__=='__main__':
    try:main()
    except Exception as error:
        write(HERE/'MARGIN_RESUME_FIT_FAILURE.json',dict(error=repr(error),no_auto_restart=True));raise
