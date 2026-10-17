"""One gated sensor-only early-yield feasibility episode, after regression.

The offline teacher can only choose legal stop, then returns control to the
unchanged actor after confirmed moving-track clearance. It never supplies a
timed turn, future obstacle pose, simulator truth or a deployed controller.
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
BUNDLE=ROOT/'models/stop_feedback_20261015/actual95_transfer_bc400_s231016/bc_portable'
DEPENDENCY='rl-recovery-followup-broad-20261016.service'


def stamp():return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def write(name,value):
    path=OUT/name;temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(path)


def state():
    raw=subprocess.check_output(['systemctl','--user','show',DEPENDENCY,
        '-p','MainPID','-p','ActiveState','-p','Result','-p','InvocationID'],text=True)
    return dict(line.split('=',1) for line in raw.splitlines())


def main():
    if (OUT/'PLAN.json').exists():raise FileExistsError('No automatic restart or extra trials')
    audit_path=BASE/'early_yield_replay_20261016/RESULTS.json'
    audit=json.loads(audit_path.read_text())
    assert audit['completed'] and audit['reproduced_observation_actor_command_and_pose_frames']==237
    admission=next(r for r in audit['summaries'] if r['case_id']=='proactive_fresh_condition_01')
    signal=admission['earliest_sensor_yield_signal']
    assert signal is not None and signal['short_contact_s'] is None
    cases_path=BASE/'proactive_augmented_protocol_20261016/fresh_cases.json'
    case=next(c for c in json.loads(cases_path.read_text()) if c['id']=='proactive_fresh_condition_01')
    write('case.json',case)
    config=dict(horizon_s=5.,period_s=.2,margin_m=.04,velocity_uncertainty_floor_m_s=.025,
        adaptive_track_uncertainty=True,finite_measured_response=True,
        trigger='nominal moving action predicts contact while stopping predicts no contact',
        release='three consecutive valid frames with a supported moving track, legal forward or nominal motion, and no forecast contact',
        unknown_prediction_releases_yield=False,teacher_override_action=0,
        no_forced_motion_on_release=True,no_absolute_episode_time_trigger=True)
    write('expert_config.json',config)
    manifest=json.loads((BUNDLE/'manifest.json').read_text())
    paths=[Path(__file__),audit_path,cases_path,OUT/'case.json',OUT/'expert_config.json',BUNDLE/'manifest.json']
    paths += [BUNDLE/name for name in manifest['files']]
    for folder in ('neural_navigation','experiments','navrl'):
        paths+=list((ROOT/'training'/folder).rglob('*.py'))
    paths += [p for p in (ROOT/'training'/case['scene']).iterdir() if p.is_file()]
    inputs={str(p.relative_to(ROOT)):digest(p) for p in set(paths)}
    predecessor=state()
    write('PLAN.json',dict(at=stamp(),inputs=inputs,predecessor=DEPENDENCY,
        predecessor_invocation=predecessor['InvocationID'],maximum_new_physical_trials=1,
        source_actor='actual95_transfer_bc400',source_actor_is_not_the_new_followup_model=True,
        unchanged_model_body_lidar_actions_guard_and_140s_deadline=True,
        teacher_sources='current scan-derived guard tracks, odometry, velocity, legal masks and nominal actor only',
        truth_used_by_teacher=False,actor_training_inputs_unchanged=True,
        positive_source_requires_complete_safe_episode=True,
        no_physical_parallelism=True,new_ppo_steps=0,new_optimizer_updates=0,
        goal_achieved=False,deployment_approved=False))
    while True:
        current=state()
        if current['InvocationID'] not in ('',predecessor['InvocationID']):
            raise RuntimeError('Physical predecessor was restarted')
        if current['ActiveState'] in ('inactive','failed'):
            if current['ActiveState']!='inactive' or current['Result']!='success' or current['MainPID']!='0':
                write('RESULTS.json',dict(completed=True,skipped=True,reason='Physical regression did not finish successfully',
                    new_physical_steps=0,goal_achieved=False,deployment_approved=False))
                return
            break
        if current['ActiveState']!='active' or current['MainPID']=='0':
            raise RuntimeError('Unrecognized predecessor state; never start simultaneous physics')
        write('STATUS.json',dict(at=stamp(),status='waiting_for_physical_regression',physical_simulation_started=False,predecessor=current))
        time.sleep(5)
    for name,expected in inputs.items():assert digest(ROOT/name)==expected,name
    import numpy as np
    import torch
    from unittest.mock import patch
    from experiments.evaluate_docking_navigation import evaluate
    from experiments.experiment_provenance import record_provenance
    from neural_navigation.decision_feedback_runtime import StopFeedbackRuntime
    from neural_navigation.experiment_budget import ExperimentBudget
    from neural_navigation.masked_actions import ACTIONS,COMMANDS,masks_from_observation
    from neural_navigation.motion_risk import predicted_body_risk
    torch.set_num_threads(1)
    budget=ExperimentBudget(.35,memory_mib=3072,directory=OUT)
    record_provenance(OUT,BUNDLE,[Path(__file__)])
    created=[]
    class YieldTeacher:
        def __init__(self,actor,core,guard):
            self.actor=actor;self.core=core;self.guard=guard
            self.yielding=False;self.clear_frames=0;self.entries=[];self.events=[]
        def predict(self,observation):
            reference=np.asarray(self.actor.predict(observation))
            differences=np.max(abs(ACTIONS-reference),axis=1);nominal=int(differences.argmin())
            assert differences[nominal]<2e-6
            mask=masks_from_observation(np.asarray(observation)[:282]);assert mask[nominal]
            now=float(self.core.elapsed)
            valid=bool(observation[435]>.5 and observation[438]>.5)
            candidate=nominal if nominal else next((k for k in (1,2,3) if mask[k]),0)
            predictor=self.guard.predictive_constraint
            assert abs(float(predictor.timestamp)-now)<1e-8
            forecast=None;stop_contact=None;moving_contact=None;supported=0
            if valid and candidate:
                forecast=predicted_body_risk(predictor.tracks,np.asarray(self.core.odom),now,
                    COMMANDS[[0,candidate]],self.guard.half*2,horizon_s=5.,period_s=.2,
                    margin_m=.04,uncertainty_m_s=.025,
                    measured_velocity=np.asarray(self.core.measured_velocity),adaptive_track_uncertainty=True)
                stop_contact=forecast[0]['predicted_margin_contact_s']
                moving_contact=forecast[1]['predicted_margin_contact_s']
                supported=forecast[1]['supported_tracks']
            if (not self.yielding and nominal and valid and supported
                    and moving_contact is not None and stop_contact is None):
                self.yielding=True;self.clear_frames=0
                self.events.append(dict(type='begin_yield',time_s=now,nominal=nominal,
                    predicted_motion_contact_s=moving_contact))
            controlled=self.yielding
            if self.yielding:
                clear=valid and bool(candidate) and supported>0 and moving_contact is None and stop_contact is None
                self.clear_frames=self.clear_frames+1 if clear else 0
                if self.clear_frames>=3:
                    self.yielding=False;controlled=False
                    self.events.append(dict(type='release_to_model',time_s=now,nominal=nominal))
            choice=0 if controlled else nominal
            assert mask[choice]
            self.entries.append(dict(time_s=now,nominal=nominal,choice=choice,
                expert_controlled=controlled,prediction_valid=valid,supported_tracks=supported,
                forecast_candidate=candidate,stop_contact_s=stop_contact,motion_contact_s=moving_contact,
                consecutive_clear_frames=self.clear_frames,legal_choices=np.flatnonzero(mask).tolist()))
            return ACTIONS[choice].copy()
    class Runtime(StopFeedbackRuntime):
        def __init__(self,*args,**kwargs):
            super().__init__(*args,**kwargs)
            self.teacher=YieldTeacher(self.policy.policy,self.core,self.guard)
            self.policy.policy=self.teacher;created.append(self)
    write('STATUS.json',dict(at=stamp(),status='single_physical_early_yield_probe',physical_simulation_started=True))
    with patch('neural_navigation.goal_pose_runtime.resolve_runtime',return_value=Runtime):
        row,trace=evaluate(case,BUNDLE,budget)
    assert len(created)==1
    teacher=created[0].teacher;assert len(teacher.entries)==len(trace['choices'])
    active=np.asarray([r['expert_controlled'] for r in teacher.entries],bool)
    trace['expert_controlled']=active;trace['model_controlled']=~active
    trace['reference_choices']=np.asarray([r['nominal'] for r in teacher.entries],int)
    path=OUT/'early_yield_proactive_fresh_condition_01.npz'
    np.savez_compressed(path,**trace)
    for name,expected in inputs.items():assert digest(ROOT/name)==expected,name
    row.update(variant='offline_early_yield_teacher_with_actual95_actor',
        trace_path=str(path.relative_to(ROOT)),trace_sha256=digest(path),
        expert_controlled_frames=int(active.sum()),model_controlled_frames=int((~active).sum()),
        events=teacher.events,intervention=teacher.entries,
        all_teacher_overrides_are_legal_stops=True,
        not_a_learned_model_success=True,positive_source_not_automatically_admitted=True)
    write('RESULTS.json',dict(completed=True,at=stamp(),rows=[row],resource=budget.statistics(),
        full_safe_expert_trajectory_exists=bool(row['safe_success'] and active.any() and teacher.events),
        no_new_model_trained=True,new_ppo_steps=0,new_optimizer_updates=0,
        goal_achieved=False,deployment_approved=False))
    write('STATUS.json',dict(at=stamp(),status='complete',physical_simulation_started=True,safe_success=row['safe_success']))


if __name__=='__main__':
    os.chdir(ROOT)
    try:main()
    except Exception as error:
        write('FAILURE.json',dict(at=stamp(),error=repr(error),automatically_restarted=False))
        raise
