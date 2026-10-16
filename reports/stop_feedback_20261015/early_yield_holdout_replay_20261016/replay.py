"""Replay the two remaining learned-model impacts and inspect sensor-only yield signals.

No physics or counterfactual command is executed. Truth is used only in the
separate post-step clearance score, never by either forecast or actor replay.
"""
from datetime import datetime
import time
import hashlib
import json
import os
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT=Path(__file__).resolve().parents[3]
OUT=Path(__file__).resolve().parent
BASE=ROOT/'reports/stop_feedback_20261015'
BUNDLE=ROOT/'models/stop_feedback_20261015/early_yield_remaining_bc600_s231016/bc_portable'


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def write(name,value):
    path=OUT/name;temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(path)


def main():
    if (OUT/'PLAN.json').exists():raise FileExistsError('Do not restart this frozen replay')
    import numpy as np
    import torch
    from neural_navigation.decision_feedback_runtime import StopFeedbackRuntime
    from neural_navigation.experiment_budget import ExperimentBudget
    from neural_navigation.masked_actions import ACTIONS,COMMANDS,masks_from_observation
    from neural_navigation.motion_risk import predicted_body_risk
    from experiments.stop_recovery_policy import StopRecoveryPPO
    torch.set_num_threads(1)
    budget=ExperimentBudget(.15,memory_mib=2048,directory=OUT)
    source_path=OUT/'FROZEN_FINISHED_CASES.json'
    source=json.loads(source_path.read_text());assert source['completed']
    cases_path=BASE/'early_yield_remaining_protocol_20261016/heldout_cases.json'
    cases=json.loads(cases_path.read_text())
    failures=[r for r in source['rows'] if r['collisions']]
    assert len(failures)==2 and {r['case_id'] for r in failures}=={
        'early_yield_sensor_holdout_02_0','early_yield_sensor_holdout_02_1'}
    manifest=json.loads((BUNDLE/'manifest.json').read_text())
    paths=[Path(__file__),source_path,cases_path,BUNDLE/'manifest.json',*[BUNDLE/n for n in manifest['files']]]
    paths += [ROOT/r['trace_path'] for r in failures]
    for folder in ('neural_navigation','experiments','navrl'):
        paths+=list((ROOT/'training'/folder).rglob('*.py'))
    inputs={str(p.relative_to(ROOT)):digest(p) for p in set(paths)}
    write('PLAN.json',dict(inputs=inputs,maximum_replayed_frames=103,
        horizons_s=[2.4,5.0],finite_measured_motion_response=True,
        velocity_uncertainty_floor_m_s=.025,adaptive_track_uncertainty=True,
        forecast_source='guard scan tracks in continuous odometry frame',
        truth_used_only_for_separate_poststep_scoring=True,
        no_new_physics=True,new_ppo_steps=0,new_optimizer_updates=0,
        goal_achieved=False,deployment_approved=False))
    model=StopRecoveryPPO.load(ROOT/'models/stop_feedback_20261015/early_yield_remaining_bc600_s231016/policy_bc.zip',device='cpu')
    model.policy.set_training_mode(False)
    summaries=[];all_records=[];total=0
    for row in failures:
        case=next(c for c in cases if c['id']==row['case_id'])
        trace_path=ROOT/row['trace_path'];assert digest(trace_path)==row['trace_sha256']
        with np.load(trace_path,allow_pickle=False) as z:data={key:z[key].copy() for key in z.files}
        converted=np.concatenate((data['observations'][:,:300],data['observations'][:,303:]),axis=1).astype(np.float32)
        with torch.no_grad():
            probabilities=model.policy.get_distribution(torch.from_numpy(converted),data['masks']).distribution.probs.numpy()
            supported=model.policy.recovery_support(torch.from_numpy(converted)).numpy()
        np.testing.assert_array_equal(probabilities.argmax(-1),data['choices'])
        runtime=StopFeedbackRuntime(BUNDLE,ROOT/'training'/case['scene']/'decision_map.yaml',
            np.asarray(case['start'])+np.asarray(case.get('initial_prior_offset',[0,0,0])),case['goal'],'shortcut')
        records=[]
        class Replay:
            def __init__(self,actor):self.actor=actor;self.cursor=0
            def predict(self,observation):
                i=self.cursor;now=float(data['labels'][i,0])
                np.testing.assert_allclose(observation,data['observations'][i],rtol=0,atol=2e-6)
                reference=self.actor.predict(observation)
                np.testing.assert_allclose(reference,ACTIONS[data['choices'][i]],rtol=0,atol=2e-6)
                mask=masks_from_observation(np.asarray(observation)[:282])
                choice=int(data['choices'][i]);forward=next((k for k in (1,2,3) if mask[k]),0)
                selected=list(range(12))
                predictor=runtime.guard.predictive_constraint
                assert abs(float(predictor.timestamp)-now)<1e-8
                predictions={};forecast_cpu_ms={}
                for horizon in (2.4,5.0):
                    forecast_start=time.process_time()
                    values=predicted_body_risk(predictor.tracks,np.asarray(runtime.core.odom),now,
                        COMMANDS[selected],runtime.guard.half*2,horizon_s=horizon,period_s=.2,
                        margin_m=.04,uncertainty_m_s=.025,
                        measured_velocity=np.asarray(runtime.core.measured_velocity),
                        adaptive_track_uncertainty=True)
                    forecast_cpu_ms[str(horizon)]=(time.process_time()-forecast_start)*1000.
                    predictions[str(horizon)]={str(k):v for k,v in zip(selected,values)}
                record=dict(case_id=case['id'],time_s=now,choice=choice,
                    legal_choices=np.flatnonzero(mask).tolist(),forward_choice=forward,
                    prediction_input_valid=bool(observation[435]>.5 and observation[438]>.5),
                    tracks=len(predictor.tracks),predictions=predictions,forecast_cpu_ms=forecast_cpu_ms,
                    actor_stop_probability=float(probabilities[i,0]),recovery_branch_active=bool(supported[i]),
                    source_actor_reproduced=True)
                records.append(record);self.cursor+=1
                return ACTIONS[choice].copy()
        replay=Replay(runtime.policy.policy);runtime.policy.policy=replay
        for i in range(len(data['choices'])):
            result=runtime.step(data['scans'][i],data['odometry'][i],data['measured_velocity'][i],
                float(data['labels'][i,0]),capture_observation=True)
            np.testing.assert_allclose(result['command'],data['commands'][i],rtol=0,atol=2e-8)
            np.testing.assert_allclose(runtime.core.estimate,data['labels'][i,4:7],rtol=0,atol=2e-8)
            # Independent diagnostic only. These values never enter Replay.predict.
            pose=data['labels'][i,1:4];c,s=np.cos(pose[2]),np.sin(pose[2])
            humans=np.asarray(data['human_positions'][i])
            local=(humans-pose[:2])@np.array([[c,-s],[s,c]])
            gaps=np.linalg.norm(local-np.clip(local,-runtime.guard.half,runtime.guard.half),axis=1)-.25
            records[-1]['scored_truth_human_gap_m']=float(gaps.min())
            records[-1]['executed_stop_reason']=result['stop_reason']
            total+=1;budget.tick()
            if i%25==0:write('STATUS.json',dict(status='causal_replay',case_id=case['id'],frame=i,total_frames=total))
        assert replay.cursor==len(data['choices'])
        triggers=[]
        for record in records:
            choice=record['choice']
            if not record['prediction_input_valid'] or not choice:continue
            short=record['predictions']['2.4'][str(choice)]
            long=record['predictions']['5.0'][str(choice)]
            stop=record['predictions']['5.0']['0']
            if (long['supported_tracks'] and long['predicted_margin_contact_s'] is not None
                    and stop['predicted_margin_contact_s'] is None):
                triggers.append(dict(time_s=record['time_s'],choice=choice,
                    short_contact_s=short['predicted_margin_contact_s'],long_contact_s=long['predicted_margin_contact_s'],
                    scored_truth_gap_m=record['scored_truth_human_gap_m'],
                    actor_stop_probability=record['actor_stop_probability'],recovery_branch_active=record['recovery_branch_active']))
        tail=0
        for choice in data['choices'][::-1]:
            if choice:break
            tail+=1
        summaries.append(dict(case_id=case['id'],reproduced_frames=len(records),
            source_collisions=row['collisions'],source_terminal_stationary_frames=tail,
            earliest_sensor_yield_signal=triggers[0] if triggers else None,
            sensor_yield_signal_frames=len(triggers),
            no_counterfactual_yield_success_claim=True))
        all_records.extend(records)
        write('PARTIAL.json',dict(completed=False,frames=total,summaries=summaries))
    assert total==103
    for name,checksum in inputs.items():assert digest(ROOT/name)==checksum,name
    write('RESULTS.json',dict(completed=True,at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
        reproduced_observation_actor_command_and_pose_frames=total,summaries=summaries,records=all_records,
        new_physical_steps=0,new_ppo_steps=0,new_optimizer_updates=0,resource=budget.statistics(),
        goal_achieved=False,deployment_approved=False,
        scope='Sensor evidence for an earlier necessary wait; only complete physical validation can establish a safe source.'))
    write('STATUS.json',dict(status='complete',frames=total))


if __name__=='__main__':
    os.chdir(ROOT)
    try:main()
    except Exception as error:
        write('FAILURE.json',dict(error=repr(error),automatically_restarted=False))
        raise
