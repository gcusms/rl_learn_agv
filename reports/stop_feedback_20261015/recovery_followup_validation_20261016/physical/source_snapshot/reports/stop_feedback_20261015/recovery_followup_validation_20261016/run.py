"""Serial physical validation of the separately saved bounded BC followup.

First check the regressed complete task, then preserve normal and earlier
learned recovery behavior. An unfitted checkpoint never starts physical work.
Actual106 is not declared fixed when it supplied no successful source.
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
FIT=BASE/'recovery_followup_fit_20261016'
MODEL=ROOT/'models/stop_feedback_20261015/recovery_followup_bc400_s231016'
DEPENDENCY='rl-recovery-followup-fit-20261016.service'


def stamp():return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def write(name,data):
    p=OUT/name;t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(data,indent=2)+'\n');t.replace(p)


def state():
    output=subprocess.check_output(['systemctl','--user','show',DEPENDENCY,
        '-p','MainPID','-p','ActiveState','-p','Result','-p','InvocationID'],text=True)
    return dict(line.split('=',1) for line in output.splitlines())


def main():
    if (OUT/'PLAN.json').exists():raise FileExistsError('Fresh bounded validation required')
    dependency=state()
    paths=[Path(__file__),FIT/'run.py',FIT/'PLAN.json',
        BASE/'actual95_broad_regression_20261016/RESULTS.json',
        BASE/'actual95_learned_validation_20261016/RESULTS.json',
        BASE/'actual95_learned_validation_20261016/prefixes.json',
        BASE/'proactive_augmented_protocol_20261016/fresh_cases.json',
        BASE/'w11_recovery_bc800_navigation_20261016/cases.json',
        BASE/'w11_recovery_bc800_w11_20261016/specifications.json',
        BASE/'actual_stop_validation_20261016/RESULTS_main.json']
    for folder in ['navrl','neural_navigation','experiments']:
        paths+=list((ROOT/'training'/folder).rglob('*.py'))
    inputs={str(p.relative_to(ROOT)):digest(p) for p in set(paths)}
    write('PLAN.json',dict(at=stamp(),inputs=inputs,predecessor=DEPENDENCY,
        predecessor_invocation=dependency['InvocationID'],maximum_new_physical_cases=8,
        ordered_groups=['regressed_fresh33','normal4','motion15','actual95','actual90'],
        first_task_must_safely_arrive_and_retain_parent_per_case_efficiency=True,
        original_normal_and_recovery_safety_and_efficiency_required=True,
        new_ppo_steps=0,new_optimizer_updates=0,no_parallel_physical_main=True,
        body_actions_masks_guard_and_timeout_unchanged=True,goal_achieved=False,deployment_approved=False))
    while True:
        current=state()
        if current['InvocationID'] not in ('',dependency['InvocationID']):raise RuntimeError('Fitting predecessor restarted')
        if current['ActiveState'] in ('inactive','failed'):
            if current['ActiveState']!='inactive' or current['Result']!='success' or current['MainPID']!='0':
                raise RuntimeError('Fitting process did not finish successfully')
            break
        write('STATUS.json',dict(at=stamp(),status='waiting_for_final_fitting_artifact',physical_simulation_started=False,predecessor=current))
        time.sleep(5)
    fit_path=FIT/'RESULTS.json';fit=json.loads(fit_path.read_text());assert fit['completed']
    if not fit['exported_for_bounded_evaluation']:
        write('RESULTS.json',dict(completed=True,skipped=True,reason='Fitting failed exact old-label/stop or new-data admission.',
            fitting_sha256=digest(fit_path),new_physical_steps=0,goal_achieved=False,deployment_approved=False))
        return
    assert digest(MODEL/'policy_bc.zip')==fit['checkpoint_sha256']
    bundle=MODEL/'bc_portable';manifest=json.loads((bundle/'manifest.json').read_text())
    model_inputs={str(p.relative_to(ROOT)):digest(p) for p in [fit_path,MODEL/'policy_bc.zip',
        bundle/'manifest.json',*[bundle/name for name in manifest['files']]]}
    write('MODEL_ADMISSION.json',dict(at=stamp(),inputs=model_inputs,physical_simulation_started=True,
        original2421_and_57stop_labels_retained=True,actual106_safe_source_admitted=fit['actual106_source']['admitted']))
    inputs.update(model_inputs)
    def verify():
        for name,expected in inputs.items():assert digest(ROOT/name)==expected,name
    verify()
    import numpy as np
    import torch
    from experiments.evaluate_docking_navigation import evaluate
    from experiments.evaluate_feedback_recovery import evaluate_recovery
    from experiments.experiment_provenance import record_provenance
    from experiments.paired_navigation_gate import compare
    from neural_navigation.experiment_budget import ExperimentBudget
    torch.set_num_threads(1)
    old=json.loads((BASE/'actual95_learned_validation_20261016/RESULTS.json').read_text())
    broad=json.loads((BASE/'actual95_broad_regression_20261016/RESULTS.json').read_text())
    historical=json.loads((BASE/'actual_stop_validation_20261016/RESULTS_main.json').read_text())
    fresh=next(c for c in json.loads((BASE/'proactive_augmented_protocol_20261016/fresh_cases.json').read_text()) if c['id']=='localization_nav_33')
    cases=json.loads((BASE/'w11_recovery_bc800_navigation_20261016/cases.json').read_text())
    normal_ids={r['case_id'] for r in old['normal_rows']}
    normal=[c for c in cases if c['id'] in normal_ids]
    motion=next(c for c in cases if c['id']=='motion_profile_015')
    prefix95=next(c for c in json.loads((BASE/'actual95_learned_validation_20261016/prefixes.json').read_text()) if c['release_s']==95.)
    prefix90=next(c for c in json.loads((BASE/'w11_recovery_bc800_w11_20261016/specifications.json').read_text()) if c['release_s']==90.)
    assert len(normal)==4
    physical=OUT/'physical';physical.mkdir();record_provenance(physical,bundle,[Path(__file__)])
    budget=ExperimentBudget(.35,memory_mib=3072,directory=physical)
    report=dict(completed=False,rows=[],gates={},new_ppo_steps=0,new_optimizer_updates=0,
        actual106_repair_not_claimed=True,original_extra12_collisions_not_claimed_fixed=True,
        full18_and_extra12_acceptance_pending=True,goal_achieved=False,deployment_approved=False)
    def run(case,group,recovery=False):
        verify()
        write('STATUS.json',dict(at=stamp(),status='physical_validation',group=group,case_id=case['id'],finished=len(report['rows'])))
        row,trace=(evaluate_recovery if recovery else evaluate)(case,bundle,budget)
        path=physical/(group+'_'+case['id']+'.npz');np.savez_compressed(path,**trace)
        row.update(group=group,variant='recovery_followup_bc400',trace_sha256=digest(path),
            trace_path=str(path.relative_to(ROOT)),model_only_after_initialization=True)
        report['rows'].append(row);report['resource']=budget.statistics();write('RESULTS.json',report)
        print(json.dumps(row),flush=True)
        return row
    def finish(reason):
        verify()
        report.update(completed=True,at=stamp(),reason=reason,resource=budget.statistics())
        write('RESULTS.json',report);write('STATUS.json',dict(at=stamp(),status='complete',reason=reason))
    row=run(fresh,'regressed_fresh33')
    parent=next(r for r in historical['groups']['fresh']['rows'] if r['case_id']==fresh['id'])
    repaired=bool(row['safe_success'] and row['seconds']<=parent['seconds']*1.2+1e-8
        and row['travel_m']<=parent['travel_m']*1.2+1e-8)
    report['gates']['fresh33_per_case_parent_efficiency']=repaired
    if not repaired:finish('The regressed task failed the retained per-case safety/efficiency gate.');return
    new_normal=[run(c,'normal4') for c in normal]
    paired=compare(old['normal_rows'],new_normal);report['gates']['normal4']=paired
    if not paired['all_regression_checks_passed']:finish('Normal4 regression gate failed.');return
    motion_row=run(motion,'motion15')
    old_motion=next(r for r in broad['full18_rows'] if r['case_id']==motion['id'])
    paired=compare(old['normal_rows']+[old_motion],new_normal+[motion_row]);report['gates']['normal5']=paired
    if not paired['all_regression_checks_passed']:finish('Normal5 regression gate failed.');return
    recovery=run(prefix95,'actual95',recovery=True)
    known=next(r for r in old['recovered_rows'] if r['release_sensor_timestamp_s']==95.)
    paired=compare([known],[recovery]);report['gates']['actual95']=paired
    if not paired['all_regression_checks_passed']:finish('Known learned actual95 recovery regressed.');return
    recovery=run(prefix90,'actual90',recovery=True)
    paired=compare(broad['groups']['w11_t90']['rows'],[recovery]);report['gates']['actual90']=paired
    report['bounded_gates_passed']=paired['all_regression_checks_passed']
    finish('Bounded gates completed; full development regression and unresolved actual106 remain required.')


if __name__=='__main__':
    os.chdir(ROOT)
    try:main()
    except Exception as error:
        write('FAILURE.json',dict(at=stamp(),error=repr(error),automatically_restarted=False,
            goal_achieved=False,deployment_approved=False))
        raise
