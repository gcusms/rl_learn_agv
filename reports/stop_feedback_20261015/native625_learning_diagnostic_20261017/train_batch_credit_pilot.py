"""Bounded, serial follow-up: frozen-policy collection, full credit and STOP retention."""
import argparse
from datetime import datetime,timezone
import gc
import hashlib
import json
from pathlib import Path
import random
import shutil
import subprocess
import time


ROOT=Path(__file__).resolve().parents[3]
BASE=ROOT/'reports/stop_feedback_20261015'
HERE=Path(__file__).resolve().parent
DEPENDENCIES={
    'rl-native625-suite-ppo-20261016.service':'d74b80d255c343d599183aa14c78fe84',
    'rl-native625-suite-post-validate-20261016.service':'c53b2d88cbb343e7b4dd997a44d83643',
    'rl-native625-holdout-pair-20261016.service':'fcf4b6663c2b44deb5591fe73641c07e',
}
PARENT=ROOT/'models/stop_feedback_20261015/normal_suite_wait_pair_v2_s231016/wait_continuation'


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path,value):
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
    temporary.replace(path)


def states():
    result={}
    for unit,expected in DEPENDENCIES.items():
        output=subprocess.check_output(['systemctl','--user','show',unit,
            '--property=MainPID,ActiveState,InvocationID,ExecMainStatus'],text=True)
        observed=dict(line.split('=',1) for line in output.splitlines() if '=' in line)
        if observed['InvocationID'] not in ('',expected):
            raise RuntimeError('A pinned predecessor was replaced: '+unit)
        result[unit]=observed
    return result


def main(args):
    output,models=args.output.resolve(),args.models.resolve()
    output.mkdir(parents=True,exist_ok=False)
    source=BASE/'context_broad_diagnostic_20261016/physical/hard3_w11_training_stall_t106.npz'
    tasks_path=BASE/'context_feedback_longfit_protocol_20261016/tasks.json'
    anchors_path=BASE/'normal_suite_wait_protocol_v2_20261016/demonstrations.npz'
    parent_result=BASE/'normal_suite_wait_validation_v2_20261016/RESULTS.json'
    case=read(tasks_path)['actual106']['case']
    files=[Path(__file__),HERE/'retained_stop_ppo.py',HERE/'batch_credit_ppo.py',
        HERE/'CONSTRAINT_AND_CREDIT_TESTS.json',source,tasks_path,anchors_path,parent_result,
        PARENT/'checkpoint.pt',PARENT/'parent_actor.pt']
    manifest=read(PARENT/'portable/manifest.json')
    files += [PARENT/'portable/manifest.json']+[PARENT/'portable'/name for name in manifest['files']]
    for folder in ['experiments','neural_navigation','navrl']:
        files += list((ROOT/'training'/folder).rglob('*.py'))
    files += [ROOT/'training/neural_navigation/static_tracking_kernel.cpp']
    files += [p for p in (ROOT/'training'/case['scene']).iterdir() if p.is_file()]
    hashes={str(p.relative_to(ROOT)):digest(p) for p in set(files)}
    write(output/'PLAN.json',dict(inputs=hashes,dependencies=DEPENDENCIES,
        episodes_per_collection=args.episodes,collection_batches=1,release_frames=450,
        maximum_ppo_decisions=(case['max_steps']-450)*args.episodes,
        maximum_initialization_physical_steps=450,sampling_temperature=4.,gamma=.99,gae_lambda=1.,
        learning_rate=.0003,epochs=4,minibatch_size=64,
        one_policy_and_critic_frozen_during_complete_episode_collection=True,
        actor_advantages_normalized_once_over_collection_batch=True,
        all_reference_stop_labels_checked_after_every_proposed_actor_update=True,
        unchanged_rewards_sensors_actions_masks_body4cm_and_dynamics=True,
        new_sensor_logging_for_independent_guard_diagnosis=True,
        original_scan_unknown_semantics_and_safety_unchanged=True,
        no_parallel_physical_job=True,no_automatic_deployment=True))
    while True:
        observed=states()
        done=all(s['MainPID']=='0' and s['ActiveState'] in ('inactive','failed') for s in observed.values())
        if done:
            break
        write(output/'STATUS.json',dict(status='waiting_for_exact_serial_predecessors',
            dependencies=observed,new_ppo_steps=0,new_physical_episodes=0))
        time.sleep(5.)
    previous=read(BASE/'native625_suite_ppo_20261016/RESULTS.json')
    if not previous.get('completed') or any(s['ActiveState']=='failed' or s['ExecMainStatus']!='0' for s in observed.values()):
        write(output/'RESULTS.json',dict(completed=True,training_started=False,
            reason='A predecessor is incomplete or failed; inspect before new physics',
            dependencies=observed,new_ppo_steps=0,goal_achieved=False,deployment_approved=False))
        write(output/'STATUS.json',dict(status='not_started_predecessor_failed'))
        return
    post=read(BASE/'native625_suite_post_validation_20261016/RESULTS.json')
    holdout=read(BASE/'native625_frozen_holdout_validation_20261016/RESULTS.json')
    if (post.get('all_development_retention_passed') and post.get('hard_state_comparison',{}).get('all_safe_success')
            and holdout.get('all_frozen_checks_passed')):
        write(output/'RESULTS.json',dict(completed=True,training_started=False,
            reason='Predecessor already passed all hard, retention and frozen checks; review rather than automatically retrain',
            new_ppo_steps=0,goal_achieved=False,deployment_approved=False))
        write(output/'STATUS.json',dict(status='not_needed_predecessor_passed_all_checks'))
        return
    assert read(HERE/'CONSTRAINT_AND_CREDIT_TESTS.json')['passed']
    assert read(parent_result)['normal18_pair']['all_regression_checks_passed']
    for name,expected in hashes.items():
        assert digest(ROOT/name)==expected,name

    import numpy as np
    import torch
    from batch_credit_ppo import BatchCreditOnPolicy,prepare_completed_batch
    from experiments.guard_reason_release_actor import GuardReasonReleaseActor,export
    from experiments.recovery_state_snapshot import RecoverySnapshot,tensor_digest
    from experiments.train_native625_recovery import collect_episode,physical_step
    from experiments.verify_recovery_snapshot_replay import RecordingState,RecordedActions
    from navrl.map import StaticMap
    from neural_navigation.experiment_budget import ExperimentBudget
    from neural_navigation.goal_pose_runtime import check_files
    from neural_navigation.masked_actions import ACTIONS
    from neural_navigation.release_runtime import ReleaseRuntime
    from neural_navigation.stress_env import StressNavigationEnv
    from neural_navigation.training_distance import TrainingDistanceGraph

    models.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1);torch.manual_seed(231017)
    budget=ExperimentBudget(.35,memory_mib=2800,directory=output)
    with np.load(anchors_path,allow_pickle=False) as saved:
        anchors,labels=saved['observations'].copy(),saved['choices'].copy()
    actor=GuardReasonReleaseActor(torch.jit.load(str(PARENT/'parent_actor.pt'),map_location='cpu')).eval()
    actor.load_state_dict(torch.load(PARENT/'checkpoint.pt',map_location='cpu',weights_only=False)['actor'])
    learner=BatchCreditOnPolicy(actor,anchors[labels==0],learning_rate=.0003,temperature=4.)
    generator=torch.Generator().manual_seed(231017)
    minibatches=torch.Generator().manual_seed(231018)
    frozen_parent=tensor_digest(actor.parent)
    behavior_actor=tensor_digest(actor)
    behavior_critic=tensor_digest(learner.critic)
    with np.load(source,allow_pickle=False) as saved:
        reference={key:saved[key].copy() for key in saved.files}
    field=TrainingDistanceGraph(StaticMap(ROOT/'training'/case['scene']/'decision_map.yaml',radius=.3)).field(case['goal'])
    report=dict(completed=False,training_started=True,episodes=[],ppo_decisions=0,ppo_physical_steps=0,
        initialization_physical_steps=0,new_ppo_steps=0,new_actor_optimizer_updates=0,new_critic_optimizer_updates=0,
        goal_achieved=False,deployment_approved=False)

    def save():
        report.update(new_ppo_steps=report['ppo_decisions'],resource=budget.statistics(),
            new_optimizer_updates=report['new_actor_optimizer_updates']+report['new_critic_optimizer_updates'])
        write(output/'RESULTS.json',report)

    env=StressNavigationEnv(case=case,components=PARENT/'portable',
        scene_dir=ROOT/'training'/case['scene'],route_mode='shortcut',max_steps=case['max_steps'])
    env.reset(seed=case['seed'],options=dict(start=case['start'],goal=case['goal']))
    runtime=ReleaseRuntime(PARENT/'portable',ROOT/'training'/case['scene']/'decision_map.yaml',case['start'],case['goal'],'shortcut')
    runtime.policy.policy=RecordedActions(RecordingState(reference['choices']))
    for index in range(450):
        physical_step(env,runtime,budget,reference,index)
        report['initialization_physical_steps']+=1
        if index%50==0:
            write(output/'STATUS.json',dict(status='exact_physical_initialization',matched_frames=index+1,new_ppo_steps=0))
    snapshot=RecoverySnapshot(env,runtime)
    env.close();del env,runtime;gc.collect()
    rollouts=[]
    for episode in range(1,args.episodes+1):
        env,runtime=snapshot.restore()
        sensors={key:[] for key in ['raw_scans','input_odometry','input_measured_velocity',
            'per_action_guard_reasons','per_action_guard_executed']}
        original_step=runtime.step

        def record_step(scan,odometry,velocity,timestamp,**kwargs):
            sensors['raw_scans'].append(np.asarray(scan).copy())
            sensors['input_odometry'].append(np.asarray(odometry).copy())
            sensors['input_measured_velocity'].append(np.asarray(velocity).copy())
            result=original_step(scan,odometry,velocity,timestamp,**kwargs)
            frame=runtime.context.guard_probe_frame
            sensors['per_action_guard_reasons'].append([str(r) for r in frame['reasons']])
            sensors['per_action_guard_executed'].append(np.asarray(frame['executed']).copy())
            return result

        runtime.step=record_step
        write(output/'STATUS.json',dict(status='frozen_policy_collection',episode=episode,new_ppo_steps=report['ppo_decisions']))
        try:
            rollout,row=collect_episode(env,runtime,learner,generator,field,budget,case['max_steps']-450,report)
        finally:
            runtime.step=original_step
            env.close();del env,runtime,original_step,record_step;gc.collect()
        assert tensor_digest(actor)==behavior_actor and tensor_digest(learner.critic)==behavior_critic
        rollout.update({key:np.asarray(value) for key,value in sensors.items()})
        assert all(len(value)==len(rollout['actions']) for value in sensors.values())
        path=output/f'episode_{episode:02d}.npz'
        np.savez_compressed(path,**rollout)
        row.update(episode=episode,release_frames=450,rollout_path=str(path.relative_to(ROOT)),
            rollout_sha256=digest(path),behavior_actor_sha256=behavior_actor,behavior_critic_sha256=behavior_critic)
        report['episodes'].append(row);rollouts.append(rollout);save()
        print(json.dumps({k:v for k,v in row.items() if k!='reward_terms'}),flush=True)
    snapshot.verify_shared_models();del snapshot;gc.collect()
    report['collection_actor_and_critic_frozen_verified']=True
    batch,normalization=prepare_completed_batch(learner,rollouts,gamma=.99)
    report['batch_normalization']=normalization
    np.savez_compressed(output/'collection_batch.npz',**batch)
    del rollouts;gc.collect();budget.tick()
    pools=[np.flatnonzero(labels==0),np.flatnonzero(labels!=0)]
    updates=[]
    for epoch in range(4):
        order=torch.randperm(len(batch['actions']),generator=minibatches).numpy()
        for start in range(0,len(order),64):
            selected=order[start:start+64]
            anchor_indices=np.r_[pools[0][torch.randint(len(pools[0]),(64,),generator=minibatches).numpy()],
                pools[1][torch.randint(len(pools[1]),(64,),generator=minibatches).numpy()]]
            arguments=[batch[key][selected] for key in ['observations','masks','actions','old_log_probability','advantages','returns']]
            statistics=learner.update(*arguments,sampled_temperature=4.,rollout_advantages_normalized=True,anchors=anchors[anchor_indices])
            updates.append(statistics)
            report['new_actor_optimizer_updates']+=int(statistics['actor_updated'])
            report['new_critic_optimizer_updates']+=1
            budget.tick()
        report['updates']=updates;save()
        write(output/'STATUS.json',dict(status='batch_ppo_update',completed_epochs=epoch+1,new_ppo_steps=report['ppo_decisions']))
    assert learner.stop_mismatches()==0 and tensor_digest(actor.parent)==frozen_parent
    torch.save(dict(actor=actor.state_dict(),critic=learner.critic.state_dict(),
        actor_optimizer=learner.actor_optimizer.state_dict(),critic_optimizer=learner.critic_optimizer.state_dict(),
        frozen_reference_head=learner.reference_head.state_dict(),policy_rng=generator.get_state(),
        minibatch_rng=minibatches.get_state(),torch_rng=torch.get_rng_state(),numpy_rng=np.random.get_state(),
        python_rng=random.getstate(),ppo_decisions=report['ppo_decisions'],normalization=normalization,
        parent_checkpoint_sha256=digest(PARENT/'checkpoint.pt'),exact_physical_rollout_resume=False,
        no_deployment_approval=True),models/'checkpoint.pt')
    shutil.copy2(PARENT/'parent_actor.pt',models/'parent_actor.pt')
    with torch.no_grad():
        decisions=actor(torch.from_numpy(anchors)).numpy()
    matches=(decisions==ACTIONS[labels]).all(1)
    assert matches[labels==0].all()
    checks=np.r_[anchors,reference['observations'][530:]]
    export(actor,ROOT/'models/stop_feedback_20261015/long_risk_bc1000_s231016/portable',models/'portable',checks)
    portable=models/'portable'
    preservation=read(portable/'navigation_preservation.json')
    preservation.update(new_ppo_steps=report['ppo_decisions'],
        training_method='frozen_complete_batch_mc_advantage_and_reference_stop_guard',
        gamma=.99,gae_lambda=1.,normalization_scope=normalization['normalization_scope'],
        reference_stop_retention_is_not_physical_safety_acceptance=True)
    write(portable/'navigation_preservation.json',preservation)
    manifest=read(portable/'manifest.json')
    manifest.update(additional_ppo_steps=report['ppo_decisions'],deployment_approved=False,
        training_method='frozen_complete_batch_mc_advantage_and_reference_stop_guard',
        navigation_preservation_contract=preservation)
    manifest['files']['navigation_preservation.json']=digest(portable/'navigation_preservation.json')
    write(portable/'manifest.json',manifest)
    card=read(portable/'MODEL_CARD.json')
    card.update(method=manifest['training_method'],new_ppo_steps=report['ppo_decisions'],
        manifest_sha256=digest(portable/'manifest.json'),deployment_approved=False)
    write(portable/'MODEL_CARD.json',card)
    check_files(portable,manifest)
    for name,expected in hashes.items():
        assert digest(ROOT/name)==expected,name
    report.update(completed=True,at=datetime.now(timezone.utc).isoformat(),
        correct_source_labels=int(matches.sum()),total_source_labels=len(matches),
        all_reference_stop_labels_retained=True,diagnostic_portable_exported=True,
        source_motion_changes_are_not_physical_acceptance=True,input_hashes_verified=len(hashes),
        physical_evaluation_still_required=True)
    assert report['ppo_decisions'] <= (case['max_steps']-450)*args.episodes
    save();write(output/'STATUS.json',dict(status='complete',new_ppo_steps=report['ppo_decisions'],goal_achieved=False,deployment_approved=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--episodes',type=int,choices=[4,8],default=8)
    parser.add_argument('--output',type=Path,default=BASE/'native625_batch_credit_pilot_20261017')
    parser.add_argument('--models',type=Path,default=ROOT/'models/stop_feedback_20261015/native625_batch_credit_s231017')
    args=parser.parse_args()
    try:
        main(args)
    except Exception as error:
        if args.output.exists():
            write(args.output/'FAILURE.json',dict(error=repr(error),goal_achieved=False,deployment_approved=False))
        raise
