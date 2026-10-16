"""One sensor/odometry-based expert feasibility diagnostic after failed probes.

The completed causal audit shows left rotation legal while forward is vetoed.
Establish heading using pure rotation and zero fallback, rather than letting
the old actor creep closer during blocked forward requests. This is an offline
expert probe, never a deployed controller or a learned-model success.
"""
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
from zoneinfo import ZoneInfo


ROOT=Path(__file__).resolve().parents[3]
OUT=Path(__file__).resolve().parent
BASE=ROOT/'reports/stop_feedback_20261015'
BUNDLE=ROOT/'models/stop_feedback_20261015/actual95_transfer_bc400_s231016/bc_portable'


def stamp():return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def write(name,data):
    p=OUT/name;t=p.with_suffix(p.suffix+'.tmp')
    t.write_text(json.dumps(data,indent=2)+'\n');t.replace(p)


def main():
    if (OUT/'PLAN.json').exists():raise FileExistsError('Do not restart this single physical diagnostic')
    units=['rl-actual95-broad-regression-20261016.service',
           'rl-actual106-confirmed-free-probe-verified-20261016.service',
           'rl-actual106-safe-source-20261016.service',
           'rl-actual106-turn-bottleneck-20261016.service']
    states={}
    for unit in units:
        output=subprocess.check_output(['systemctl','--user','show',unit,
            '-p','MainPID','-p','ActiveState','-p','Result'],text=True)
        states[unit]=dict(line.split('=',1) for line in output.splitlines())
        assert states[unit]==dict(MainPID='0',Result='success',ActiveState='inactive'),states[unit]
    audit_path=BASE/'actual106_turn_bottleneck_20261016/RESULTS.json'
    audit=json.loads(audit_path.read_text());assert audit['completed']
    selected=next(r for r in audit['records'] if abs(r['time_s']-111.8)<1e-8)
    assert 4 in selected['legal_choices'] and 1 not in selected['legal_choices']
    previous_path=BASE/'actual106_confirmed_free_probe_verified_20261016/RESULTS.json'
    previous=json.loads(previous_path.read_text())
    assert previous['completed'] and len(previous['rows'])==3 and not any(r['safe_success'] for r in previous['rows'])
    prefix=json.loads((BASE/'actual_stop_validation_20261016/actual_prefix.json').read_text())
    prefix.update(id='actual106_sensor_turn_then_pass',release_s=106.)
    specification=dict(id=prefix['id'],kind='offline_sensor_odometry_expert',
        begin_s=110.,right_turn_rad=.60,left_heading_offset_rad=.80,
        forward_odometry_distance_m=.65,maximum_phase_s=12.,
        illegal_request_fallback='currently_legal_stop_only',
        no_translation_while_establishing_heading=True,
        learned_actor_after_expert_release=True)
    write('prefix.json',prefix);write('specification.json',specification)
    manifest=json.loads((BUNDLE/'manifest.json').read_text())
    paths=[Path(__file__),audit_path,previous_path,OUT/'prefix.json',OUT/'specification.json',
        ROOT/prefix['trace'],BUNDLE/'manifest.json']+[BUNDLE/name for name in manifest['files']]
    for folder in ['navrl','neural_navigation','experiments']:
        paths+=list((ROOT/'training'/folder).rglob('*.py'))
    paths+=list((ROOT/'training'/prefix['case']['scene']).glob('*'))
    inputs={str(p.relative_to(ROOT)):digest(p) for p in set(paths) if p.is_file()}
    write('PLAN.json',dict(at=stamp(),inputs=inputs,predecessors_verified_terminal=states,
        original_physical_prefix_frames=530,experimental_memory_activation_s=106.,
        unchanged_model_body_4cm_margin_commands_and_140s_timeout=True,
        same_memory_hypothesis=dict(free_evidence_margin_m=.03,free_evidence_frames=3,free_evidence_max_gap_s=.4),
        new_factor='Pure-turn odometry targets and legal zero fallback; no translational fallback while establishing heading.',
        expert_uses_only_current_observation_mask_odometry_and_its_own_phase=True,
        simulator_truth_or_future_obstacles_used_by_expert=False,
        maximum_physical_trials=1,no_parallel_physical_main=True,
        new_ppo_steps=0,new_optimizer_updates=0,goal_achieved=False,deployment_approved=False))
    import numpy as np
    import torch
    from unittest.mock import patch
    import experiments.evaluate_feedback_recovery as evaluator
    from experiments.experiment_provenance import record_provenance
    from navrl.geometry import wrap
    from neural_navigation.decision_feedback_runtime import StopFeedbackRuntime
    from neural_navigation.experiment_budget import ExperimentBudget
    from neural_navigation.masked_actions import ACTIONS,masks_from_observation
    torch.set_num_threads(1)
    budget=ExperimentBudget(.35,memory_mib=3072,directory=OUT)
    record_provenance(OUT,BUNDLE,[Path(__file__)])
    created=[]
    class Expert:
        def __init__(self,actor,core):
            self.actor=actor;self.core=core;self.phase='not_started';self.entries=[]
            self.origin=None;self.pass_origin=None;self.phase_since=None;self.release_s=None
        def predict(self,observation):
            reference=self.actor.predict(observation)
            now=float(self.core.elapsed);mask=masks_from_observation(np.asarray(observation)[:282])
            if now<specification['begin_s']-1e-8 or self.phase=='released':return reference
            odom=np.asarray(self.core.odom,float)
            if self.origin is None:
                self.origin=odom.copy();self.phase='right';self.phase_since=now
            yaw=float(wrap(odom[2]-self.origin[2]))
            if self.phase=='right' and yaw<=-specification['right_turn_rad']+.025:
                self.phase='left';self.phase_since=now
            if self.phase=='left' and yaw>=specification['left_heading_offset_rad']-.025:
                self.phase='pass';self.phase_since=now;self.pass_origin=odom[:2].copy()
            distance=0. if self.pass_origin is None else float(np.linalg.norm(odom[:2]-self.pass_origin))
            if self.phase=='pass' and distance>=specification['forward_odometry_distance_m']:
                self.phase='released';self.release_s=now;return reference
            if now-self.phase_since>specification['maximum_phase_s']+1e-8:self.phase='failed_hold'
            requested=5 if self.phase=='right' else 4 if self.phase=='left' else 0
            if self.phase=='pass':requested=next((k for k in [1,2,3] if mask[k]),0)
            choice=requested if mask[requested] else 0
            self.entries.append(dict(time_s=now,phase=self.phase,yaw_offset_rad=yaw,
                requested_choice=int(requested),choice=int(choice),request_legal=bool(mask[requested]),
                progress_m=distance,legal_choices=np.flatnonzero(mask).tolist()))
            return ACTIONS[choice].copy()
    class Runtime(StopFeedbackRuntime):
        def __init__(self,*args,**kwargs):
            super().__init__(*args,**kwargs)
            self.expert=Expert(self.policy.policy,self.core);self.policy.policy=self.expert
            self.memory_activated=False;created.append(self)
        def step(self,scan,odometry,velocity,timestamp,*args,**kwargs):
            if not self.memory_activated and timestamp>=106.-1e-8:
                self.guard.memory.config=dict(self.guard.memory.config,
                    free_evidence_margin_m=.03,free_evidence_frames=3,free_evidence_max_gap_s=.4)
                self.guard.memory.free_confirmations.clear();self.memory_activated=True
            return super().step(scan,odometry,velocity,timestamp,*args,**kwargs)
    write('STATUS.json',dict(at=stamp(),status='physical_sensor_turn_diagnostic',new_ppo_steps=0))
    with patch.object(evaluator,'StopFeedbackRuntime',Runtime):
        row,trace=evaluator.evaluate_recovery(prefix,BUNDLE,budget)
    assert len(created)==1 and created[0].memory_activated
    expert=created[0].expert
    times=trace['labels'][:,0];active=np.zeros(len(times),bool)
    for entry in expert.entries:
        found=np.flatnonzero(np.abs(times-entry['time_s'])<1e-8)
        assert len(found)==1;active[found[0]]=True
    trace['expert_controlled']=active
    trace['model_controlled'] &= ~active
    path=OUT/(specification['id']+'.npz');np.savez_compressed(path,**trace)
    for name,expected in inputs.items():assert digest(ROOT/name)==expected,name
    row.update(case_id=specification['id'],specification=specification,
        expert_controlled_frames=int(active.sum()),model_controlled_frames=int(trace['model_controlled'].sum()),
        expert_release_s=expert.release_s,final_expert_phase=expert.phase,
        intervention=expert.entries,trace_sha256=digest(path),
        no_runtime_teacher_admitted=True,not_a_learned_model_success=True,
        positive_training_source_not_automatically_admitted=True)
    write('RESULTS.json',dict(completed=True,at=stamp(),rows=[row],resource=budget.statistics(),
        new_ppo_steps=0,new_optimizer_updates=0,goal_achieved=False,deployment_approved=False,
        scope='One reused actual106 expert feasibility diagnostic with measured odometry phase targets.'))
    write('STATUS.json',dict(at=stamp(),status='complete',safe_expert_source_exists=bool(row['safe_success']),
        learned_actual106_recovery_demonstrated=False))
    print(json.dumps({k:v for k,v in row.items() if k!='intervention'}),flush=True)


if __name__=='__main__':
    os.chdir(ROOT)
    try:main()
    except Exception as error:
        write('FAILURE.json',dict(at=stamp(),error=repr(error),automatically_restarted=False,
            goal_achieved=False,deployment_approved=False))
        raise
