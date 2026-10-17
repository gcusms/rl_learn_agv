"""Physically create missing legal-wait states, then test model-only resumption."""
import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
BASE = Path('reports/stop_feedback_20261015')
DEPENDENCY = 'rl-w11-retention-pair-20261016.service'
INVOCATION = 'ccf6796d00f34b44add304cf192a1dae'


def stamp():
    return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(name,data):
    path=OUT/name
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
    temporary.replace(path)


def state():
    return dict(line.split('=',1) for line in subprocess.check_output([
        'systemctl','--user','show',DEPENDENCY,'-p','Id','-p','ActiveState',
        '-p','MainPID','-p','Result','-p','InvocationID'],text=True).splitlines())


def verify(plan):
    for name,expected in plan['inputs'].items():
        if digest(ROOT/name)!=expected:
            raise ValueError('Frozen pause probe input changed: '+name)


def prepare():
    if (OUT/'PLAN.json').exists():
        raise FileExistsError('Do not overwrite a prepared probe')
    current=state()
    if current['ActiveState']=='active':
        if current['InvocationID']!=INVOCATION or current['MainPID']=='0':
            raise ValueError('Unexpected main invocation')
    elif current['ActiveState']!='inactive' or current['Result']!='success' or current['MainPID']!='0':
        raise ValueError('The previous main job did not complete normally')
    source_folder=ROOT/BASE/'w11_recovery_bc800_navigation_20261016'
    report=json.loads((source_folder/'RESULTS.json').read_text())
    cases={c['id']:c for c in json.loads((source_folder/'cases.json').read_text())}
    rows={r['case_id']:r for r in report['rows']}
    if not report['completed']:
        raise ValueError('A completed source evaluation is required')
    specs=[]
    for case_id,pause_s in [('train_failure_w11_e7',2.6),('motion_profile_002',7.8)]:
        row=rows[case_id]
        if not row['safe_success'] or row['collisions'] or row['false_arrival']:
            raise ValueError('The pre-pause reference must have a complete safe outcome')
        source=source_folder/(case_id+'_'+row['variant']+'.npz')
        if digest(source)!=row['trace_sha256']:
            raise ValueError('The pre-pause source trace changed')
        specs.append(dict(id='legal_pause_'+case_id,case=cases[case_id],
            trace=str(source.relative_to(ROOT)),source_trace_sha256=row['trace_sha256'],
            pause_s=pause_s,hold_s=2.4,release_s=round(pause_s+2.4,8)))
    write('specifications.json',specs)
    old_path=ROOT/BASE/'reference_wait_t32_protocol_20261016/PLAN.json'
    inputs=dict(json.loads(old_path.read_text())['inputs'])
    extra=[Path(__file__),OUT/'specifications.json',old_path,
        source_folder/'RESULTS.json',source_folder/'cases.json',
        ROOT/BASE/'retention_followup_20261016/RESULTS.json',
        ROOT/BASE/'safe_wait_inventory_20261016/RESULTS.json',
        ROOT/'training/experiments/evaluate_feedback_recovery.py',
        ROOT/'training/experiments/evaluate_docking_navigation.py']
    extra += [ROOT/s['trace'] for s in specs]
    for path in extra:
        inputs[str(path.relative_to(ROOT))]=digest(path)
    plan=dict(created_at=stamp(),dependency_unit=DEPENDENCY,dependency_invocation=INVOCATION,
        dependency_at_prepare=current,inputs=inputs,
        bundle='models/stop_feedback_20261015/w11_recovery_bc800_s231015/bc_portable',
        output=str(BASE/'feasible_wait_probe_navigation_20261016'),
        predeclared_probes=2,hold_seconds=2.4,
        selection='Previously safe W11 and profile002; pause at their first audited decision-divergence sensor timestamps.',
        purpose='Create physically real long-legal-wait observations missing from124safe traces; admit no labels before complete safety verification.',
        initialization_is_scripted_not_learned=True,model_only_after_release=True,
        guard_action_table_footprint_lidar_dynamics_unchanged=True,
        source_main_slot_reused=True,third_physical_simulation=False,
        soft_cpu_cores=.35,hard_cpu_percent=50,memory_high_mib=3072,memory_max_mib=4096,
        no_automatic_training_expansion_or_promotion=True,goal_achieved=False,deployment_approved=False)
    verify(plan)
    write('PLAN.json',plan)
    print(json.dumps(dict(prepared=True,probes=2,pinned_inputs=len(inputs))))


def execute_probe(specification,bundle,budget):
    import numpy as np
    from unittest.mock import patch
    from experiments.evaluate_feedback_recovery import SensorPrefixActor,load_prefix
    from experiments.evaluate_docking_navigation import evaluate
    from neural_navigation.decision_feedback_runtime import StopFeedbackRuntime
    from neural_navigation.masked_actions import ACTIONS

    source_spec=dict(trace=specification['trace'],release_s=specification['pause_s'])
    prefix,expected_poses=load_prefix(source_spec)
    created=[]

    class PauseActor:
        def __init__(self,actor,core):
            self.source=SensorPrefixActor(actor,core,prefix,specification['pause_s'])
            self.core=core
            self.hold_frames=0
            self.model_frames=0
            self.first_model_observation=None

        def predict(self,observation):
            if self.source.cursor<len(prefix['times']):
                return self.source.predict(observation)
            if self.core.elapsed<specification['release_s']-1e-8:
                self.hold_frames+=1
                return ACTIONS[0].copy()
            if self.first_model_observation is None:
                self.first_model_observation=np.asarray(observation).copy()
            self.model_frames+=1
            return self.source.actor.predict(observation)

    class PauseRuntime(StopFeedbackRuntime):
        def __init__(self,*args,**kwargs):
            super().__init__(*args,**kwargs)
            actor=PauseActor(self.policy.policy,self.core)
            self.policy.policy=actor
            created.append(actor)

    with patch('neural_navigation.goal_pose_runtime.resolve_runtime',return_value=PauseRuntime):
        row,trace=evaluate(specification['case'],bundle,budget)
    if len(created)!=1:
        raise ValueError('Exactly one runtime expected')
    actor=created[0]
    n=len(prefix['times'])
    if actor.source.cursor!=n:
        raise ValueError('The original source prefix did not finish')
    np.testing.assert_array_equal(trace['choices'][:n],prefix['choices'])
    np.testing.assert_allclose(trace['labels'][:n+1,1:4],expected_poses,atol=2e-8,rtol=0)
    model_controlled=trace['labels'][:,0]>=specification['release_s']-1e-8
    trace['model_controlled']=model_controlled
    if actor.model_frames and actor.hold_frames!=12:
        raise ValueError('Expected exactly twelve actual hold commands')
    held=(trace['labels'][:,0]>=specification['pause_s']-1e-8)&~model_controlled
    if not np.all(trace['choices'][held]==0) or not np.all(abs(trace['commands'][held])<1e-8):
        raise ValueError('Pause must execute zero under the unchanged guard')
    obs=trace['observations']
    masks=trace['masks'].astype(bool)
    usable=masks[:,1:].any(-1)&(obs[:,435]>.5)&(obs[:,438]>.5)&(
        np.linalg.norm(obs[:,218:220]*12.,axis=-1)>.3)
    context=(obs[:,144:216].min(-1)*12.<.8)|(~masks[:,1:].all(-1))|(
        obs[:,419:431].max(-1)>0.)|(obs[:,436]>0.)
    prior_legal_wait=obs[:,434]*5.
    late=model_controlled&usable&(prior_legal_wait>=2.-1e-6)
    motion=late&(trace['choices']!=0)&(abs(trace['commands']).max(-1)>1e-8)
    first=actor.first_model_observation
    row.update(case_id=specification['id'],original_case_id=specification['case']['id'],
        specification=specification,recovery_test=True,not_a_full_mission_score=True,
        source_prefix_frames=n,actual_scripted_hold_frames=actor.hold_frames,
        source_sensor_and_physical_prefix_parity_verified=True,
        release_sensor_timestamp_s=specification['release_s'],
        first_model_prior_legal_wait_s=None if first is None else float(first[434]*5.),
        model_controlled_frames=int(model_controlled.sum()),
        post_release_valid_late_wait_frames=int(late.sum()),
        post_release_valid_late_motion_frames=int(motion.sum()),
        guaranteed_proactive_context_late_motion_frames=int((motion&context).sum()),
        initialization_reached_release=bool(actor.model_frames),
        may_supply_verified_late_motion=bool(row['safe_success'] and not row['collisions']
            and not row['false_arrival'] and not row['physical_guards'] and (motion&context).any()),
        not_admitted_to_training_yet=True)
    return row,trace


def run():
    import numpy as np
    import torch
    from neural_navigation.experiment_budget import ExperimentBudget
    plan=json.loads((OUT/'PLAN.json').read_text())
    if (ROOT/plan['output']).exists():
        raise FileExistsError('Do not restart a physical probe')
    verify(plan)
    while True:
        current=state()
        if current['InvocationID'] not in ('',plan['dependency_invocation']):
            raise ValueError('The preceding main job was restarted')
        if current['ActiveState'] in ('failed','inactive'):
            if current['ActiveState']!='inactive' or current['Result']!='success' or current['MainPID']!='0':
                raise ValueError('The preceding main job did not complete successfully')
            break
        write('STATUS.json',dict(status='waiting',at=stamp(),dependency=current,physical_simulation_started=False))
        time.sleep(5)
    previous=json.loads((ROOT/BASE/'retention_pair_20261016/RESULTS.json').read_text())
    if not previous['completed'] or len(previous['results'])!=8:
        raise ValueError('All eight preceding fixed evaluation groups must finish')
    verify(plan)
    output=ROOT/plan['output']
    output.mkdir()
    specifications=json.loads((OUT/'specifications.json').read_text())
    (output/'specifications.json').write_text(json.dumps(specifications,indent=2)+'\n')
    torch.set_num_threads(1)
    budget=ExperimentBudget(.35,memory_mib=3072,directory=output)
    write('ADMISSION.json',dict(admitted_two_physical_probes=True,at=stamp(),
        previous_main_complete=True,source_main_slot_reused=True,third_physical_simulation=False,
        original_protection_unchanged=True,model_promotion=False))
    report=dict(completed=False,rows=[],scope=plan['purpose'],
        scripted_initialization_only=True,model_only_after_release=True,
        new_ppo_steps=0,goal_achieved=False,deployment_approved=False)
    for specification in specifications:
        verify(plan)
        write('STATUS.json',dict(status='running',at=stamp(),case_id=specification['id']))
        row,trace=execute_probe(specification,ROOT/plan['bundle'],budget)
        path=output/(specification['id']+'.npz')
        np.savez_compressed(path,**trace)
        row.update(variant='reference_bc_after_actual_hold',trace_sha256=digest(path))
        report['rows'].append(row)
        report['resource']=budget.statistics()
        (output/'RESULTS.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
        print(json.dumps(row,ensure_ascii=False),flush=True)
    report['completed']=True
    (output/'RESULTS.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    write('RESULTS.json',dict(completed=True,at=stamp(),report=plan['output']+'/RESULTS.json',
        outcomes=[{k:r[k] for k in ['case_id','safe_success','collisions','timeout',
            'first_model_prior_legal_wait_s','post_release_valid_late_motion_frames',
            'guaranteed_proactive_context_late_motion_frames','may_supply_verified_late_motion']}
            for r in report['rows']],no_automatic_training_expansion_or_promotion=True,
        goal_achieved=False,deployment_approved=False))
    write('STATUS.json',dict(status='complete',at=stamp(),goal_achieved=False,deployment_approved=False))


if __name__=='__main__':
    os.chdir(ROOT)
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['prepare','run'])
    args=parser.parse_args()
    try:
        prepare() if args.action=='prepare' else run()
    except Exception as error:
        write('STATUS.json',dict(status='failed',at=stamp(),error=repr(error),automatically_restarted=False))
        raise
