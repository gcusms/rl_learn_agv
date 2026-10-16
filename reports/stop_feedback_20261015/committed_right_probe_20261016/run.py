"""Test a guarded, direction-consistent expert in the actual106s failure.

This is an offline physical feasibility experiment. The priority selector is
never installed in the navigation policy and adds no PPO updates. It waits for
the one current main physical slot, retaining the separate auxiliary slot.
"""
import argparse
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
BASE=Path('reports/stop_feedback_20261015')
DEPENDENCY='rl-actual-stop-validation-main-20261016.service'
INVOCATION='230b830f7df54d93bb366947fccd53b4'


def stamp():
    return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(name,value):
    path=OUT/name;temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2)+'\n');temporary.replace(path)


def state():
    return dict(line.split('=',1) for line in subprocess.check_output([
        'systemctl','--user','show',DEPENDENCY,'-p','ActiveState','-p','MainPID',
        '-p','Result','-p','InvocationID'],text=True).splitlines())


def verify(plan):
    for name,expected in plan['inputs'].items():
        if digest(ROOT/name)!=expected:raise ValueError('Fixed experiment input changed: '+name)


def prepare():
    if (OUT/'PLAN.json').exists():raise FileExistsError('Do not replace the experiment')
    dependency=state()
    if dependency['ActiveState']=='active':
        if dependency['InvocationID']!=INVOCATION or dependency['MainPID']=='0':raise ValueError('Unexpected main job')
    elif dependency['ActiveState']!='inactive' or dependency['Result']!='success' or dependency['MainPID']!='0':
        raise ValueError('Main dependency must be the known live or successfully completed job')
    original=ROOT/BASE/'actual_stop_validation_20261016/PLAN.json'
    plan=json.loads(original.read_text())
    inputs=dict(plan['inputs'])
    first_path=ROOT/BASE/'actual_t106_action9_probe_20261016/RESULTS.json'
    first=json.loads(first_path.read_text())['rows'][0]
    if (first['specification']['id']!='actual_t106_action9_4frames' or first['safe_success']
            or first['collisions'] or not first['timeout']):raise ValueError('The completed short right-turn negative is required')
    trace=ROOT/BASE/'actual_t106_action9_probe_20261016/actual_t106_action9_4frames.npz'
    if digest(trace)!=first['trace_sha256']:raise ValueError('The complete negative trace changed')
    write('first_negative_row.json',first)
    prefix=json.loads((ROOT/BASE/'actual_stop_validation_20261016/actual_prefix.json').read_text())
    write('prefix.json',prefix)
    specs=[dict(id='right_direction_'+str(seconds)+'s',start_s=106.,end_s=106.+seconds,
        ordered_legal_actions=[5,9],fallback_choice=0) for seconds in [3,6]]
    write('specifications.json',specs)
    for path in [Path(__file__),original,trace,OUT/'first_negative_row.json',OUT/'prefix.json',
            OUT/'specifications.json',ROOT/BASE/'supported_motion_gap_20261016/GRADIENTS.json']:
        inputs[str(path.relative_to(ROOT))]=digest(path)
    specification=dict(created_at=stamp(),inputs=inputs,bundle=plan['bundle'],
        dependency_unit=DEPENDENCY,dependency_invocation=INVOCATION,dependency_at_prepare=dependency,
        dependency_completed_report=str(BASE/'actual_stop_validation_20261016/RESULTS_main.json'),
        output=str(BASE/'committed_right_actual_t106_20261016'),predeclared_physical_probes=2,
        source_main_slot_reused=True,third_physical_simulation=False,
        soft_cpu_cores=.35,hard_cpu_percent=50,memory_high_mib=3072,memory_max_mib=4096,
        guard_action_table_footprint_lidar_and_timeout_unchanged=True,
        selection_uses_only_current_original_action_mask=True,priority_selector_is_not_a_runtime_policy=True,
        stop_remains_fallback=True,new_ppo_steps=0,no_automatic_labels_training_or_promotion=True,
        goal_achieved=False,deployment_approved=False)
    verify(specification);write('PLAN.json',specification)
    print(json.dumps(dict(prepared=True,probes=2,pinned_inputs=len(inputs))))


def execute(specification,prefix,bundle,budget):
    import numpy as np
    from unittest.mock import patch
    import experiments.evaluate_feedback_recovery as recovery
    from neural_navigation.decision_feedback_runtime import StopFeedbackRuntime
    from neural_navigation.masked_actions import ACTIONS,masks_from_observation
    created=[]
    class ExpertActor:
        def __init__(self,actor,core):self.actor,self.core,self.entries=actor,core,[]
        def predict(self,observation):
            now=float(self.core.elapsed)
            if specification['start_s']-1e-8<=now<specification['end_s']-1e-8:
                mask=masks_from_observation(np.asarray(observation)[:282])
                choice=next((a for a in specification['ordered_legal_actions'] if mask[a]),0)
                if not mask[choice]:raise ValueError('Expert fallback or selected motion is inadmissible')
                self.entries.append(dict(sensor_timestamp=now,selected_choice=choice,
                    legal_actions=np.flatnonzero(mask).tolist(),expert_command=ACTIONS[choice].tolist()))
                return ACTIONS[choice].copy()
            return self.actor.predict(observation)
    class Runtime(StopFeedbackRuntime):
        def __init__(self,*a,**kw):
            super().__init__(*a,**kw)
            actor=ExpertActor(self.policy.policy,self.core);self.policy.policy=actor;created.append(actor)
    with patch.object(recovery,'StopFeedbackRuntime',Runtime):
        row,trace=recovery.evaluate_recovery(prefix,bundle,budget)
    if len(created)!=1:raise ValueError('Exactly one instrumented runtime required')
    times=trace['labels'][:,0]
    expert=(times>=specification['start_s']-1e-8)&(times<specification['end_s']-1e-8)
    if int(expert.sum())!=len(created[0].entries):raise ValueError('Expert frame accounting mismatch')
    trace['expert_controlled']=expert
    trace['model_controlled']=trace['model_controlled']&~expert
    row.update(case_id=specification['id'],specification=specification,
        not_a_trained_policy=True,expert_entries=created[0].entries,
        model_controlled_frames=int(trace['model_controlled'].sum()),expert_controlled_frames=int(expert.sum()),
        runtime_rule_added=False,commands_masks_body_and_sensors_unchanged=True,
        no_training_or_label_admission=True)
    return row,trace


def run():
    plan=json.loads((OUT/'PLAN.json').read_text());verify(plan)
    if (OUT/'RESULTS.json').exists() or (ROOT/plan['output']).exists():raise FileExistsError('Do not restart the probe')
    while True:
        dependency=state()
        if dependency['InvocationID'] not in ('',plan['dependency_invocation']):raise ValueError('The main job was restarted')
        if dependency['ActiveState'] in ('inactive','failed'):
            if dependency['ActiveState']!='inactive' or dependency['Result']!='success' or dependency['MainPID']!='0':
                raise ValueError('The preceding main job did not finish successfully')
            break
        write('STATUS.json',dict(status='waiting_for_main_slot',at=stamp(),dependency=dependency,physical_simulation_started=False))
        time.sleep(5)
    verify(plan)
    main=json.loads((ROOT/plan['dependency_completed_report']).read_text())
    if not main['completed'] or not main['fixed35_groups_completed']:raise ValueError('All main groups must finish before reusing their slot')
    prior_path=ROOT/BASE/'actual_t106_action9_probe_20261016/RESULTS.json'
    prior=json.loads(prior_path.read_text())
    if prior['completed'] and any(r['safe_success'] for r in prior['rows']):
        write('RESULTS.json',dict(completed=True,skipped=True,at=stamp(),
            reason='An original guarded action9 probe already supplied a complete safe escape; avoid redundant trials.',
            physical_simulation_started=False,new_ppo_steps=0,goal_achieved=False,deployment_approved=False))
        write('STATUS.json',dict(status='skipped_after_verified_escape',at=stamp()));return
    import numpy as np
    import torch
    from neural_navigation.experiment_budget import ExperimentBudget
    torch.set_num_threads(1)
    output=ROOT/plan['output'];output.mkdir()
    specifications=json.loads((OUT/'specifications.json').read_text())
    prefix=json.loads((OUT/'prefix.json').read_text())
    (output/'prefix.json').write_bytes((OUT/'prefix.json').read_bytes())
    (output/'specifications.json').write_bytes((OUT/'specifications.json').read_bytes())
    write('ADMISSION.json',dict(at=stamp(),main_terminal_state=dependency,
        completed_main_report_sha256=digest(ROOT/plan['dependency_completed_report']),
        source_main_slot_reused=True,third_physical_simulation=False,not_a_trained_policy=True))
    budget=ExperimentBudget(.35,memory_mib=3072,directory=output)
    report=dict(completed=False,not_a_trained_policy=True,rows=[],deployment_approved=False,
        new_ppo_steps=0,priority_selector_not_installed_in_runtime=True)
    for specification in specifications:
        verify(plan);write('STATUS.json',dict(status='running',at=stamp(),case=specification['id']))
        row,trace=execute(specification,prefix,ROOT/plan['bundle'],budget)
        path=output/(specification['id']+'.npz');np.savez_compressed(path,**trace)
        row.update(trace_sha256=digest(path),variant='offline_direction_consistent_expert')
        report['rows'].append(row);report['resource']=budget.statistics()
        (output/'RESULTS.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps({k:v for k,v in row.items() if k!='expert_entries'}),flush=True)
    report['completed']=True
    (output/'RESULTS.json').write_text(json.dumps(report,indent=2)+'\n')
    write('RESULTS.json',dict(completed=True,at=stamp(),report=str((output/'RESULTS.json').relative_to(ROOT)),
        no_labels_or_training_admitted=True,new_ppo_steps=0,goal_achieved=False,deployment_approved=False))
    write('STATUS.json',dict(status='complete',at=stamp(),goal_achieved=False))


if __name__=='__main__':
    os.chdir(ROOT)
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=['prepare','run'])
    args=parser.parse_args()
    try:prepare() if args.action=='prepare' else run()
    except Exception as error:
        write('STATUS.json',dict(status='failed',at=stamp(),error=repr(error),automatically_restarted=False));raise
