"""Bounded data-only fitting after complete physical evidence is available.

Always preserve the exact2421 examples. Add the complete safe parent task
whose efficiency regressed; optionally add the sensor-turn source only if its
entire physical trial succeeds and sampled post106 clearance remains>=4cm.
No failed actual106 trajectory is a positive example. No PPO or physics runs.
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
PARENT=ROOT/'models/stop_feedback_20261015/actual95_transfer_bc400_s231016'
TARGET=ROOT/'models/stop_feedback_20261015/recovery_followup_bc400_s231016'
BUNDLE=ROOT/'models/stop_feedback_20261015/baseline/portable'
FRESH=BASE/'fresh33_retention_audit_20261016'
SOURCE=BASE/'actual106_sensor_turn_probe_20261016'
DEPENDENCY='rl-actual106-sensor-turn-probe-20261016.service'


def stamp():return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def write(name,data):
    p=OUT/name;t=p.with_suffix(p.suffix+'.tmp')
    t.write_text(json.dumps(data,indent=2)+'\n');t.replace(p)


def state():
    output=subprocess.check_output(['systemctl','--user','show',DEPENDENCY,
        '-p','ActiveState','-p','MainPID','-p','Result','-p','InvocationID'],text=True)
    return dict(line.split('=',1) for line in output.splitlines())


def main():
    if TARGET.exists() or (OUT/'PLAN.json').exists():raise FileExistsError('Fresh fitting output required')
    fresh=json.loads((FRESH/'RESULTS.json').read_text())
    assert fresh['completed'] and fresh['source_admissible_for_separate_bounded_fitting']
    dependency=state()
    paths=[Path(__file__),PARENT/'policy_bc.zip',PARENT/'demonstrations.npz',
        FRESH/'RESULTS.json',FRESH/'PLAN.json',FRESH/'demonstrations.npz',
        SOURCE/'run.py',SOURCE/'PLAN.json',SOURCE/'prefix.json',
        BASE/'actual106_memory_context_20261016/replay.py']
    manifest=json.loads((BUNDLE/'manifest.json').read_text())
    assert digest(BUNDLE/'manifest.json')=='f412add64da251ec875e5e3f7b402e38d8153bf3b82425fafd99ef3ccc15ce4f'
    paths += [BUNDLE/'manifest.json']+[BUNDLE/name for name in manifest['files']]
    for folder in ['navrl','neural_navigation','experiments']:
        paths+=list((ROOT/'training'/folder).rglob('*.py'))
    inputs={str(p.relative_to(ROOT)):digest(p) for p in set(paths) if p.is_file()}
    plan=dict(at=stamp(),inputs=inputs,parent=str(PARENT.relative_to(ROOT)),target=str(TARGET.relative_to(ROOT)),
        predecessor=DEPENDENCY,predecessor_invocation=dependency['InvocationID'],
        maximum_supervised_updates=400,seed=231016,only_changed_factor='Complete-safe demonstration data.',
        required_original_examples=2421,required_original_stops=57,
        fresh33_source_admitted=True,actual106_source_rule='Append only if complete safe arrival and post106 sampled truth gap>=0.04m.',
        source_rule_declared_before_reading_the_new_sensor_probe_outcome=True,
        no_failed_positive_examples=True,actions_body_guard_and_observation_contract_unchanged=True,
        no_temperature_calibration=True,new_ppo_steps=0,goal_achieved=False,deployment_approved=False)
    write('PLAN.json',plan)
    while True:
        current=state()
        if current['InvocationID'] not in ('',plan['predecessor_invocation']):raise RuntimeError('Predecessor restarted')
        if current['ActiveState'] in ('inactive','failed'):
            if current['ActiveState']!='inactive' or current['Result']!='success' or current['MainPID']!='0':
                raise RuntimeError('The source experiment did not finish successfully')
            break
        write('STATUS.json',dict(at=stamp(),status='waiting_for_complete_physical_evidence',predecessor=current,
            optimization_started=False,new_ppo_steps=0))
        time.sleep(5)
    def verify():
        for name,expected in inputs.items():assert digest(ROOT/name)==expected,name
    verify()
    source_path=SOURCE/'RESULTS.json';source=json.loads(source_path.read_text());assert source['completed']
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
    TARGET.mkdir();budget=ExperimentBudget(.35,memory_mib=3072,directory=TARGET)
    model=StopRecoveryPPO.load(PARENT/'policy_bc.zip',device='cpu');model.policy.set_training_mode(False)
    assert model.num_timesteps==0
    base=frozen_base_digest(model.policy)
    assert base=='e5502406c4ee8e42c4da09dd6affa022cb76e7f9b01316b1eaf0d2551431938e'
    with np.load(FRESH/'demonstrations.npz',allow_pickle=False) as z:
        data={k:z[k].copy() for k in ['observations','masks','choices']}
    with np.load(PARENT/'demonstrations.npz',allow_pickle=False) as z:
        for key in data:np.testing.assert_array_equal(data[key][:2421],z[key])
    assert len(data['choices'])==2421+fresh['supported_safe_parent_examples']
    extra=dict(admitted=False,source_report_sha256=digest(source_path),reason='The new sensor-turn trial did not safely arrive.')
    row=source['rows'][0]
    if row['safe_success']:
        assert row['physical_arrival'] and row['protocol_valid'] and not row['false_arrival']
        assert not row['collisions'] and not row['physical_guards'] and not row['timeout']
        trace_path=SOURCE/(row['specification']['id']+'.npz')
        assert digest(trace_path)==row['trace_sha256']
        with np.load(trace_path,allow_pickle=False) as z:trace={k:z[k].copy() for k in z.files}
        spec=importlib.util.spec_from_file_location('safe_source_geometry',BASE/'actual106_memory_context_20261016/replay.py')
        geometry=importlib.util.module_from_spec(spec);spec.loader.exec_module(geometry)
        prefix=json.loads((SOURCE/'prefix.json').read_text())
        scene=json.loads((ROOT/'training'/prefix['case']['scene']/'physical_scene.json').read_text())
        hardware=json.loads((BUNDLE/'hardware.json').read_text())
        half=np.asarray(hardware['robot_footprint_m'])/2;walls=np.asarray(scene['walls'],float)
        radii=np.asarray([h['radius_m'] for h in scene['dynamic_obstacles']])
        times=trace['labels'][:,0]
        gaps=[geometry.physical_gap(trace['labels'][i,1:4],half,walls,trace['human_positions'][i],radii)
              for i in np.flatnonzero(times>=106.-1e-8)]
        minimum_gap=float(min(gaps))
        extra.update(minimum_post106_frame_sampled_truth_gap_m=minimum_gap,frame_samples_are_not_continuous_hardware_certification=True)
        if minimum_gap>=.04:
            x=convert_observations(trace['observations'],trace['masks'],trace['commands'],times)
            np.testing.assert_array_equal(masks_from_observation(x[:,:282]),trace['masks'])
            assert trace['masks'][np.arange(len(times)),trace['choices']].all()
            away=np.linalg.norm(x[:,218:220]*12.,axis=1)>.3
            np.testing.assert_allclose(trace['commands'][away],COMMANDS[trace['choices'][away]],atol=2e-8,rtol=0)
            with torch.no_grad():keep=model.policy.recovery_support(torch.from_numpy(x)).numpy()
            keep &= times>=row['specification']['begin_s']-1e-8
            assert keep.any() and np.count_nonzero(trace['choices'][keep])
            for key,array in [('observations',x),('masks',trace['masks']),('choices',trace['choices'])]:
                data[key]=np.concatenate([data[key],array[keep]])
            extra.update(admitted=True,reason='Complete safe expert source passed the declared gate.',
                source_trace=str(trace_path.relative_to(ROOT)),trace_sha256=digest(trace_path),
                new_examples=int(keep.sum()),movement_targets=int(np.count_nonzero(trace['choices'][keep])),
                stop_targets=int((trace['choices'][keep]==0).sum()),
                initialization_and_pre_intervention_waiting_excluded=True,
                experimental_memory_component_handover_required=True,
                expert_success_is_not_a_learned_model_success=True)
        else:extra['reason']='Reached, but sampled post106 physical clearance fell below4cm; exclude this source.'
    seen={}
    for i,(x,mask,label) in enumerate(zip(data['observations'],data['masks'],data['choices'])):
        key=hashlib.sha256(x.tobytes()+mask.tobytes()).digest();old=seen.setdefault(key,(int(label),i))
        if old[0]!=int(label):raise ValueError('Conflicting exact demonstration labels: '+str((old[1],i)))
    np.savez_compressed(TARGET/'demonstrations.npz',**data)
    write('SOURCE_ADMISSION.json',dict(at=stamp(),fresh33_examples=fresh['supported_safe_parent_examples'],actual106=extra,
        total_examples=len(data['choices']),old_prefix_bit_identical=True,original_stops_preserved=57))
    (TARGET/'config.json').write_text(json.dumps(dict(plan,actual106_source=extra,
        total_examples=len(data['choices']),frozen_base_sha256=base),indent=2)+'\n')
    record_provenance(TARGET,BUNDLE,[Path(__file__)])
    write('STATUS.json',dict(at=stamp(),status='bounded_supervised_fitting',examples=len(data['choices']),new_ppo_steps=0))
    x=torch.from_numpy(data['observations'])
    with torch.no_grad():before=model.policy.get_distribution(x,data['masks']).distribution.probs.numpy()
    assert int((before[:2421].argmax(1)==data['choices'][:2421]).sum())==2421
    result=initialize_from_demonstrations(model,data['observations'],data['masks'],data['choices'],budget,maximum_updates=400)
    model.policy.set_training_mode(False)
    with torch.no_grad():after=model.policy.get_distribution(x,data['masks']).distribution.probs.numpy()
    assert frozen_base_digest(model.policy)==base and model.num_timesteps==0
    old_correct=int((after[:2421].argmax(1)==data['choices'][:2421]).sum())
    all_correct=int((after.argmax(1)==data['choices']).sum())
    old_stops=data['choices'][:2421]==0
    stops=int((after[:2421].argmax(1)[old_stops]==0).sum())
    admitted=old_correct==2421 and stops==57 and all_correct==len(data['choices'])
    model.demonstration_initialization=dict(result,parent_checkpoint=str(PARENT.relative_to(ROOT)),
        additional_supervised_updates=result['updates'],actual106_source=extra)
    model.save(TARGET/'policy_bc.zip')
    torch.save(dict(optimizer=model.policy.optimizer.state_dict(),torch_rng=torch.get_rng_state(),
        numpy_rng=np.random.get_state()),TARGET/'continuation_state.pt')
    if admitted:
        export_recovery(model,BUNDLE,TARGET/'bc_portable')
        card_path=TARGET/'bc_portable/MODEL_CARD.json';card=json.loads(card_path.read_text())
        card.update(method='Bounded complete-safe development-data followup; no additionalPPO.',
            additional_supervised_updates=result['updates'],new_ppo_steps=0,
            actual106_expert_source=extra,navigation_acceptance_completed=False,
            actual106_learned_recovery_not_yet_evaluated=True)
        card_path.write_text(json.dumps(card,indent=2)+'\n')
    verify()
    final=dict(completed=True,at=stamp(),fitting=result,total_examples=len(data['choices']),
        old_examples_correct=old_correct,old_examples=2421,old_stop_labels_preserved=stops,
        all_examples_correct=all_correct,actual106_source=extra,
        exported_for_bounded_evaluation=admitted,frozen_base_sha256=base,
        checkpoint_sha256=digest(TARGET/'policy_bc.zip'),resource=budget.statistics(),
        additional_supervised_updates=result['updates'],new_ppo_steps=0,
        deployment_approved=False,goal_achieved=False)
    (TARGET/'RESULTS.json').write_text(json.dumps(final,indent=2)+'\n')
    write('RESULTS.json',final)
    write('STATUS.json',dict(at=stamp(),status='complete',exported_for_bounded_evaluation=admitted,new_ppo_steps=0))
    print(json.dumps({k:v for k,v in final.items() if k!='fitting'}),flush=True)


if __name__=='__main__':
    os.chdir(ROOT)
    try:main()
    except Exception as error:
        write('FAILURE.json',dict(at=stamp(),error=repr(error),automatically_restarted=False,
            deployment_approved=False,goal_achieved=False))
        raise
