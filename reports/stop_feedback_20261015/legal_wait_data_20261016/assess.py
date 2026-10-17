"""Admit verified late-wait experience and preserve the original safe labels."""
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil
from zoneinfo import ZoneInfo

import numpy as np
import torch

from experiments.recovery_demonstrations import load_demonstrations,convert_observations
from experiments.stop_recovery_policy import StopRecoveryPPO
from experiments.train_prefix_recovery import frozen_base_digest


ROOT=Path(__file__).resolve().parents[3]
OUT=Path(__file__).resolve().parent
BASE=ROOT/'reports/stop_feedback_20261015'
INPUTS={}


def pin(path):
    raw=path.read_bytes()
    INPUTS[str(path.relative_to(ROOT))]=hashlib.sha256(raw).hexdigest()
    return raw


def write(name,data):
    (OUT/name).write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')


def main():
    if (OUT/'RESULTS.json').exists() or (OUT/'reference_adapter').exists():
        raise FileExistsError('Do not replace admitted data')
    torch.set_num_threads(1)
    pin(Path(__file__))
    for name in ['training/experiments/recovery_demonstrations.py',
                 'training/experiments/prefix_recovery_training.py',
                 'training/neural_navigation/decision_feedback.py']:
        pin(ROOT/name)
    probe_folder=BASE/'feasible_wait_probe_navigation_20261016'
    probe=json.loads(pin(probe_folder/'RESULTS.json'))
    if not probe['completed'] or len(probe['rows'])!=2:
        raise ValueError('Both predeclared physical probes must finish')
    probe_states={}
    probe_data={}
    for row in probe['rows']:
        if (not row['safe_success'] or row['collisions'] or row['false_arrival']
                or row['physical_guards'] or row['actual_scripted_hold_frames']!=12
                or not row['source_sensor_and_physical_prefix_parity_verified']):
            raise ValueError('A physically unsafe or unverified probe cannot supply data')
        path=probe_folder/(row['case_id']+'.npz')
        if hashlib.sha256(pin(path)).hexdigest()!=row['trace_sha256']:
            raise ValueError('Held physical trace changed')
        with np.load(path,allow_pickle=False) as trace:
            indices=np.flatnonzero(trace['model_controlled'])
            i=int(indices[0])
            obs=convert_observations(trace['observations'],trace['masks'].astype(bool),
                trace['commands'],trace['labels'][:,0])
            if abs(trace['labels'][i,0]-row['release_sensor_timestamp_s'])>1e-8:
                raise ValueError('Release must align with its actual sensor frame')
            probe_states[row['case_id']]=(obs[i].copy(),trace['masks'][i].copy(),int(trace['choices'][i]))
            probe_data[row['case_id']]=row

    # This adapter only adds the legacy demonstration row/filename convention.
    # The recorded observations, masks, actions and physical labels stay byte-identical.
    reference=BASE/'w11_recovery_bc800_navigation_20261016'
    report=json.loads(pin(reference/'RESULTS.json'))
    cases={c['id']:c for c in json.loads(pin(reference/'cases.json'))}
    row=dict(next(r for r in report['rows'] if r['case_id']=='motion_profile_002'))
    if not report['completed'] or not row['safe_success'] or row['collisions']:
        raise ValueError('Complete safe profile002 reference required')
    source=reference/(row['case_id']+'_'+row['variant']+'.npz')
    if hashlib.sha256(pin(source)).hexdigest()!=row['trace_sha256']:
        raise ValueError('Reference physical trace changed')
    adapter=OUT/'reference_adapter';adapter.mkdir()
    shutil.copyfile(source,adapter/'motion_profile_002.npz')
    row['specification']=cases['motion_profile_002']
    (adapter/'RESULTS.json').write_text(json.dumps(dict(completed=True,rows=[row],
        format_adapter_only=True,new_physical_steps=0,
        original_report=str((reference/'RESULTS.json').relative_to(ROOT)),
        original_report_sha256=INPUTS[str((reference/'RESULTS.json').relative_to(ROOT))],
        original_trace=str(source.relative_to(ROOT)),original_trace_sha256=row['trace_sha256'],
        arrays_or_labels_modified=False,deployment_approved=False),ensure_ascii=False,indent=2)+'\n')

    original_path=BASE/'w11_success_augmented_protocol_20261016/demonstrations.json'
    original=json.loads(pin(original_path))
    new_specs=[dict(report=str((probe_folder/'RESULTS.json').relative_to(ROOT)),
        id='legal_pause_train_failure_w11_e7',minimum_sensor_timestamp_s=5.,sensor_intervals_s=[[5.,6.]]),
        dict(report=str((adapter/'RESULTS.json').relative_to(ROOT)),id='motion_profile_002',
             minimum_sensor_timestamp_s=7.4,sensor_intervals_s=[[7.4,8.8]])]
    write('demonstrations.json',original+new_specs)
    held=probe_data['legal_pause_train_failure_w11_e7']
    write('training_prefixes.json',[dict(id=held['case_id'],case=held['specification']['case'],
        release_s=held['release_sensor_timestamp_s'],
        trace=str((probe_folder/(held['case_id']+'.npz')).relative_to(ROOT)))])

    inference={}
    base_digest=None
    admitted=None
    for label,folder,checkpoint in [
        ('reference','w11_recovery_bc800_s231015','policy_bc.zip'),
        ('control','w11_control2000_s231015','policy.zip'),
        ('retained','w11_retained2000_s231015','policy.zip')]:
        path=ROOT/'models/stop_feedback_20261015'/folder/checkpoint
        pin(path)
        model=StopRecoveryPPO.load(path,device='cpu')
        model.policy.set_training_mode(False)
        actual=frozen_base_digest(model.policy)
        if base_digest is not None and actual!=base_digest:
            raise ValueError('The frozen base differs across models')
        base_digest=actual
        inference[label]={}
        for case_id,(obs,mask,choice) in probe_states.items():
            x=torch.from_numpy(obs[None])
            with torch.no_grad():
                probabilities=model.policy.get_distribution(x,mask[None]).distribution.probs.numpy()[0]
                supported=bool(model.policy.recovery_support(x).item())
            greedy=int(probabilities.argmax())
            if label=='reference' and greedy!=choice:
                raise ValueError('The reference model does not reproduce its recorded released action')
            inference[label][case_id]=dict(choice=greedy,stop_probability=float(probabilities[0]),
                actual_recovery_support=supported,probabilities=probabilities.tolist(),
                offline_only_not_new_physical_evaluation=True)
        if label=='reference':
            if not inference[label]['legal_pause_train_failure_w11_e7']['actual_recovery_support']:
                raise ValueError('The admitted legal-wait example must train the recovery branch')
            obs,masks,choices,sources=load_demonstrations(original+new_specs,model.policy)
            old_path=ROOT/'models/stop_feedback_20261015/w11_recovery_bc800_s231015/demonstrations.npz'
            pin(old_path)
            with np.load(old_path,allow_pickle=False) as old:
                n=len(old['choices'])
                for key,value in [('observations',obs),('masks',masks),('choices',choices)]:
                    np.testing.assert_array_equal(value[:n],old[key])
            if n!=2089 or int((choices[:n]==0).sum())!=37:
                raise ValueError('Original2089/37necessary-wait labels must remain intact')
            support=model.policy.recovery_support(torch.from_numpy(obs)).numpy()
            if not support.all():
                raise ValueError('Demonstration inputs must lie in learnable support')
            late=(obs[:,431]*5.>=2.-1e-6)&(choices!=0)
            if not late[n:].any():
                raise ValueError('The new data did not fill any long-legal-wait motion coverage')
            conflicts={};contradictions=0
            for x,m,y in zip(obs,masks,choices):
                key=x.tobytes()+m.tobytes()
                if key in conflicts and conflicts[key]!=int(y):contradictions+=1
                conflicts[key]=int(y)
            if contradictions:
                raise ValueError('Identical inputs have contradictory demonstrated labels')
            np.savez_compressed(OUT/'demonstrations_verified.npz',observations=obs,masks=masks,choices=choices)
            admitted=dict(total_labels=len(choices),old_labels_preserved=n,
                old_necessary_wait_labels=37,new_labels=len(choices)-n,
                long_legal_wait_motion_labels=int(late.sum()),
                new_source_summary=sources[-2:],contradictory_identical_inputs=0,
                all_old_arrays_byte_equivalent=True,scripted_initialization_frames_excluded=True)
        del model
    for path in [OUT/'demonstrations.json',OUT/'training_prefixes.json',OUT/'demonstrations_verified.npz',
                 adapter/'RESULTS.json',adapter/'motion_profile_002.npz']:
        pin(path)
    write('RESULTS.json',dict(completed=True,at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
        data_admitted_for_bounded_initialization=True,model_or_ppo_admitted=False,
        data=admitted,release_state_model_inference=inference,frozen_base_sha256=base_digest,
        inputs=INPUTS,new_physical_steps=0,new_ppo_steps=0,
        original_failed_t32_continuation_not_used_as_success_data=True,
        not_a_resume_of_either2k_model=True,goal_achieved=False,deployment_approved=False))
    print(json.dumps(dict(completed=True,data=admitted,release_state_model_inference=inference),
        ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
