"""Append a completed early-yield source, retaining every old example exactly.

No failed source triggers optimizer updates. The new residual is saved beside
the parent and can be exported only after all old/new decisions are retained.
"""
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
from zoneinfo import ZoneInfo

ROOT=Path(__file__).resolve().parents[3]
OUT=Path(__file__).resolve().parent
BASE=ROOT/'reports/stop_feedback_20261015'
PARENT=ROOT/'models/stop_feedback_20261015/recovery_followup_bc400_s231016'
TARGET=ROOT/'models/stop_feedback_20261015/early_yield_bc400_s231016'
BUNDLE=ROOT/'models/stop_feedback_20261015/baseline/portable'
SOURCE=BASE/'early_yield_physical_20261016'
DEPENDENCY='rl-early-yield-physical-20261016.service'


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def stamp():return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()


def write(name,value):
    path=OUT/name;temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(path)


def state():
    raw=subprocess.check_output(['systemctl','--user','show',DEPENDENCY,
        '-p','MainPID','-p','ActiveState','-p','Result','-p','InvocationID'],text=True)
    return dict(line.split('=',1) for line in raw.splitlines())


def main():
    if TARGET.exists() or (OUT/'PLAN.json').exists():raise FileExistsError('Fresh bounded fit required')
    predecessor=state()
    manifest=json.loads((BUNDLE/'manifest.json').read_text())
    paths=[Path(__file__),PARENT/'policy_bc.zip',PARENT/'demonstrations.npz',PARENT/'RESULTS.json',
        SOURCE/'run.py',SOURCE/'PLAN.json',SOURCE/'case.json',SOURCE/'expert_config.json',
        BASE/'early_yield_replay_20261016/RESULTS.json',BASE/'actual106_memory_context_20261016/replay.py',
        BUNDLE/'manifest.json',*[BUNDLE/name for name in manifest['files']]]
    for folder in ('neural_navigation','navrl','experiments'):
        paths+=list((ROOT/'training'/folder).rglob('*.py'))
    inputs={str(p.relative_to(ROOT)):digest(p) for p in set(paths)}
    write('PLAN.json',dict(at=stamp(),inputs=inputs,predecessor=DEPENDENCY,
        predecessor_invocation=predecessor['InvocationID'],maximum_supervised_updates=400,seed=231016,
        parent_checkpoint_sha256=digest(PARENT/'policy_bc.zip'),required_old_examples=2505,
        required_old_stop_examples=57,exact_old_input_mask_label_retention=True,
        complete_safe_physical_source_required=True,minimum_post_yield_sampled_clearance_m=.04,
        minimum_supported_model_motion_frames_after_teacher_release=10,
        failed_or_partial_trajectories_are_not_positive_examples=True,
        unchanged_frozen_base_localizer_router_docking_body_actions_masks_and_guard=True,
        changed_factor='Complete-safe early-yield training data only',new_ppo_steps=0,
        goal_achieved=False,deployment_approved=False))
    while True:
        current=state()
        if current['InvocationID'] not in ('',predecessor['InvocationID']):raise RuntimeError('Source restarted')
        if current['ActiveState'] in ('inactive','failed'):
            if current['ActiveState']!='inactive' or current['Result']!='success' or current['MainPID']!='0':
                raise RuntimeError('Physical source job failed')
            break
        write('STATUS.json',dict(at=stamp(),status='waiting_for_complete_source',optimization_started=False,predecessor=current))
        time.sleep(5)
    def verify():
        for name,checksum in inputs.items():assert digest(ROOT/name)==checksum,name
    verify()
    source_path=SOURCE/'RESULTS.json';source=json.loads(source_path.read_text());assert source['completed']
    def skip(reason,**extra):
        result=dict(completed=True,skipped=True,reason=reason,source_report_sha256=digest(source_path),
            additional_supervised_updates=0,new_ppo_steps=0,exported_for_bounded_evaluation=False,
            goal_achieved=False,deployment_approved=False,**extra)
        write('SOURCE_ADMISSION.json',result);write('RESULTS.json',result)
        write('STATUS.json',dict(at=stamp(),status='complete_without_fitting',reason=reason))
    if not source.get('full_safe_expert_trajectory_exists'):
        skip('No complete safe early-yield source; no optimizer started.');return
    assert len(source['rows'])==1
    row=source['rows'][0]
    assert row['safe_success'] and row['physical_arrival'] and row['protocol_valid']
    assert not any(row[k] for k in ('false_arrival','collisions','physical_guards','timeout'))
    trace_path=ROOT/row['trace_path'];assert digest(trace_path)==row['trace_sha256']
    import importlib.util
    import numpy as np
    import torch
    from experiments.experiment_provenance import record_provenance
    from experiments.recovery_demonstrations import convert_observations,initialize_from_demonstrations
    from experiments.stop_recovery_policy import StopRecoveryPPO,export_recovery
    from experiments.train_prefix_recovery import frozen_base_digest
    from neural_navigation.experiment_budget import ExperimentBudget
    from neural_navigation.masked_actions import COMMANDS,masks_from_observation
    torch.set_num_threads(1);torch.manual_seed(231016);np.random.seed(231016)
    budget=ExperimentBudget(.35,memory_mib=3072,directory=OUT)
    with np.load(trace_path,allow_pickle=False) as z:trace={k:z[k].copy() for k in z.files}
    starts=[e['time_s'] for e in row['events'] if e['type']=='begin_yield']
    releases=[e['time_s'] for e in row['events'] if e['type']=='release_to_model']
    if not starts or not releases:
        skip('Source never returned from observed yield to the model.');return
    times=trace['labels'][:,0];begin=min(starts);release=min(releases)
    x=convert_observations(trace['observations'],trace['masks'],trace['commands'],times)
    np.testing.assert_array_equal(masks_from_observation(x[:,:282]),trace['masks'])
    assert trace['masks'][np.arange(len(times)),trace['choices']].all()
    away=np.linalg.norm(x[:,218:220]*12.,axis=1)>.3
    np.testing.assert_allclose(trace['commands'][away],COMMANDS[trace['choices'][away]],rtol=0,atol=2e-8)
    spec=importlib.util.spec_from_file_location('yield_source_geometry',BASE/'actual106_memory_context_20261016/replay.py')
    geometry=importlib.util.module_from_spec(spec);spec.loader.exec_module(geometry)
    case=json.loads((SOURCE/'case.json').read_text())
    scene=json.loads((ROOT/'training'/case['scene']/'physical_scene.json').read_text())
    half=np.asarray(json.loads((BUNDLE/'hardware.json').read_text())['robot_footprint_m'])/2
    walls=np.asarray(scene['walls'],float);radii=np.asarray([h['radius_m'] for h in scene['dynamic_obstacles']])
    gaps=[geometry.physical_gap(trace['labels'][i,1:4],half,walls,trace['human_positions'][i],radii)
        for i in np.flatnonzero(times>=begin-1e-8)]
    minimum=float(min(gaps))
    if minimum<.04:
        skip('Reached, but post-yield sampled physical clearance was below4cm.',minimum_sampled_gap_m=minimum);return
    model=StopRecoveryPPO.load(PARENT/'policy_bc.zip',device='cpu');model.policy.set_training_mode(False)
    frozen=frozen_base_digest(model.policy)
    assert frozen=='e5502406c4ee8e42c4da09dd6affa022cb76e7f9b01316b1eaf0d2551431938e' and model.num_timesteps==0
    with torch.no_grad():supported=model.policy.recovery_support(torch.from_numpy(x)).numpy()
    keep=supported & (times>=begin-1e-8)
    post=supported & trace['model_controlled'] & (times>=release-1e-8) & (trace['choices']!=0)
    if int(post.sum())<10 or not np.any(trace['choices'][keep]==0) or not np.any(trace['choices'][keep]!=0):
        skip('Insufficient supported necessary waiting or post-release learned motion.',supported_post_release_motion=int(post.sum()));return
    with np.load(PARENT/'demonstrations.npz',allow_pickle=False) as z:
        old={key:z[key].copy() for key in ('observations','masks','choices')}
    assert len(old['choices'])==2505 and int((old['choices']==0).sum())==57
    data={key:np.concatenate([old[key],value[keep]]) for key,value in
          [('observations',x),('masks',trace['masks']),('choices',trace['choices'])]}
    for key in old:np.testing.assert_array_equal(data[key][:2505],old[key])
    seen={}
    for i,(features,mask,label) in enumerate(zip(data['observations'],data['masks'],data['choices'])):
        key=hashlib.sha256(features.tobytes()+mask.tobytes()).digest();before=seen.setdefault(key,(int(label),i))
        if before[0]!=int(label):raise ValueError('Conflicting exact labels: '+repr((before[1],i)))
    admission=dict(admitted=True,source_report_sha256=digest(source_path),source_trace=str(trace_path.relative_to(ROOT)),
        source_trace_sha256=digest(trace_path),minimum_post_yield_sampled_gap_m=minimum,
        sampled_geometry_is_not_a_continuous_hardware_guarantee=True,first_sensor_yield_s=begin,first_release_s=release,
        total_examples=len(data['choices']),old_examples_retained=2505,old_stops_retained=57,
        new_examples=int(keep.sum()),new_stop_targets=int((trace['choices'][keep]==0).sum()),
        new_motion_targets=int((trace['choices'][keep]!=0).sum()),supported_model_motion_after_release=int(post.sum()),
        expert_trajectory_is_not_a_learned_model_success=True,inputs_contain_no_truth_or_future_data=True)
    write('SOURCE_ADMISSION.json',admission)
    TARGET.mkdir();np.savez_compressed(TARGET/'demonstrations.npz',**data)
    (TARGET/'config.json').write_text(json.dumps(dict(source=admission,maximum_updates=400,seed=231016),indent=2)+'\n')
    record_provenance(TARGET,BUNDLE,[Path(__file__)])
    with torch.no_grad():
        previous=model.policy.get_distribution(torch.from_numpy(old['observations']),old['masks']).distribution.probs.argmax(-1).numpy()
    np.testing.assert_array_equal(previous,old['choices'])
    write('STATUS.json',dict(at=stamp(),status='bounded_supervised_fit',examples=len(data['choices']),new_ppo_steps=0))
    fitted=initialize_from_demonstrations(model,data['observations'],data['masks'],data['choices'],budget,maximum_updates=400)
    model.policy.set_training_mode(False)
    with torch.no_grad():predicted=model.policy.get_distribution(torch.from_numpy(data['observations']),data['masks']).distribution.probs.argmax(-1).numpy()
    old_correct=int((predicted[:2505]==old['choices']).sum())
    stop_correct=int((predicted[:2505][old['choices']==0]==0).sum())
    all_correct=int((predicted==data['choices']).sum())
    assert frozen_base_digest(model.policy)==frozen and model.num_timesteps==0
    accepted=old_correct==2505 and stop_correct==57 and all_correct==len(predicted)
    model.demonstration_initialization=dict(fitted,parent_checkpoint_sha256=digest(PARENT/'policy_bc.zip'),source=admission)
    model.save(TARGET/'policy_bc.zip')
    torch.save(dict(optimizer=model.policy.optimizer.state_dict(),torch_rng=torch.get_rng_state(),numpy_rng=np.random.get_state()),TARGET/'continuation_state.pt')
    if accepted:export_recovery(model,BUNDLE,TARGET/'bc_portable')
    verify()
    result=dict(completed=True,at=stamp(),source=admission,fitting=fitted,old_examples_correct=old_correct,
        old_stop_labels_preserved=stop_correct,total_examples=len(predicted),all_examples_correct=all_correct,
        additional_supervised_updates=fitted['updates'],new_ppo_steps=0,
        checkpoint_sha256=digest(TARGET/'policy_bc.zip'),frozen_base_sha256=frozen,
        exported_for_bounded_evaluation=accepted,resource=budget.statistics(),goal_achieved=False,deployment_approved=False)
    (TARGET/'RESULTS.json').write_text(json.dumps(result,indent=2)+'\n');write('RESULTS.json',result)
    write('STATUS.json',dict(at=stamp(),status='complete',exported_for_bounded_evaluation=accepted))


if __name__=='__main__':
    os.chdir(ROOT)
    try:main()
    except Exception as error:
        write('FAILURE.json',dict(at=stamp(),error=repr(error),automatically_restarted=False))
        raise
