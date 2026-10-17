"""Gated learned-only validation of one separately saved early-yield candidate.

Maximum eight episodes. No teacher, controller patch, guard change or optimizer
is used. Stop expansion after the first failed declared admission gate.
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
MODEL=ROOT/'models/stop_feedback_20261015/early_yield_bc400_s231016'
FIT=BASE/'early_yield_fit_20261016'
DEPENDENCY='rl-early-yield-fit-20261016.service'


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def stamp():return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()


def write(name,value):
    p=OUT/name;t=p.with_suffix(p.suffix+'.tmp')
    t.write_text(json.dumps(value,indent=2)+'\n');t.replace(p)


def state():
    raw=subprocess.check_output(['systemctl','--user','show',DEPENDENCY,
        '-p','MainPID','-p','ActiveState','-p','Result','-p','InvocationID'],text=True)
    return dict(line.split('=',1) for line in raw.splitlines())


def main():
    if (OUT/'PLAN.json').exists():raise FileExistsError('Fresh bounded validation required')
    predecessor=state()
    paths=[Path(__file__),FIT/'run.py',FIT/'PLAN.json',
        BASE/'proactive_augmented_protocol_20261016/fresh_cases.json',
        BASE/'w11_recovery_bc800_navigation_20261016/cases.json',
        BASE/'actual95_learned_validation_20261016/prefixes.json',
        BASE/'w11_recovery_bc800_w11_20261016/specifications.json']
    for folder in ('neural_navigation','navrl','experiments'):
        paths+=list((ROOT/'training'/folder).rglob('*.py'))
    inputs={str(p.relative_to(ROOT)):digest(p) for p in set(paths)}
    write('PLAN.json',dict(at=stamp(),inputs=inputs,predecessor=DEPENDENCY,
        predecessor_invocation=predecessor['InvocationID'],maximum_new_physical_cases=8,
        ordered_groups=['source_condition_01','unfitted_conditions_02_04_05','retained_fresh33',
                        'retained_full_w11','retained_actual95','retained_actual90'],
        first_task_requires_safe_arrival_and_time_and_travel_within_teacher_times_1_2=True,
        extra_conditions_require_safe_arrival_and_no_collision_not_failure_time_comparison=True,
        normal_and_prior_recovery_safety_efficiency_required=True,
        learned_model_only=True,no_runtime_teacher=True,unchanged_body_actions_masks_guard_and_deadlines=True,
        reused_development_maps_not_independent_generalization=True,new_ppo_steps=0,
        new_optimizer_updates=0,goal_achieved=False,deployment_approved=False))
    while True:
        current=state()
        if current['InvocationID'] not in ('',predecessor['InvocationID']):raise RuntimeError('Fitting predecessor restarted')
        if current['ActiveState'] in ('inactive','failed'):
            if current['ActiveState']!='inactive' or current['Result']!='success' or current['MainPID']!='0':
                raise RuntimeError('Source-gated fitting failed')
            break
        write('STATUS.json',dict(at=stamp(),status='waiting_for_source_gated_fit',physical_simulation_started=False,predecessor=current))
        time.sleep(5)
    fit_path=FIT/'RESULTS.json';fit=json.loads(fit_path.read_text());assert fit['completed']
    if not fit['exported_for_bounded_evaluation']:
        write('RESULTS.json',dict(completed=True,skipped=True,reason='No safely sourced and exactly retained candidate admitted.',
            fitting_sha256=digest(fit_path),new_physical_steps=0,goal_achieved=False,deployment_approved=False))
        write('STATUS.json',dict(at=stamp(),status='complete_without_physics'));return
    import numpy as np
    import torch
    from experiments.evaluate_docking_navigation import evaluate
    from experiments.evaluate_feedback_recovery import evaluate_recovery
    from experiments.experiment_provenance import record_provenance
    from experiments.paired_navigation_gate import compare
    from experiments.stop_recovery_policy import StopRecoveryPPO
    from experiments.train_prefix_recovery import frozen_base_digest
    from neural_navigation.experiment_budget import ExperimentBudget
    from neural_navigation.masked_actions import ACTIONS,masks_from_observation
    torch.set_num_threads(1)
    bundle=MODEL/'bc_portable';manifest=json.loads((bundle/'manifest.json').read_text())
    assert manifest['observation_contract']['dimensions']==439
    source_path=BASE/'early_yield_physical_20261016/RESULTS.json'
    broad_path=BASE/'recovery_followup_broad_regression_20261016/RESULTS.json'
    known_path=BASE/'recovery_followup_validation_20261016/RESULTS.json'
    source=json.loads(source_path.read_text());broad=json.loads(broad_path.read_text());known=json.loads(known_path.read_text())
    assert source['completed'] and broad['completed'] and known['completed']
    final_paths=[fit_path,source_path,broad_path,known_path,MODEL/'policy_bc.zip',MODEL/'demonstrations.npz',
        MODEL/'continuation_state.pt',bundle/'manifest.json',*[bundle/n for n in manifest['files']]]
    inputs.update({str(p.relative_to(ROOT)):digest(p) for p in final_paths})
    assert digest(MODEL/'policy_bc.zip')==fit['checkpoint_sha256']
    def verify():
        for name,checksum in inputs.items():assert digest(ROOT/name)==checksum,name
    verify()
    budget=ExperimentBudget(.35,memory_mib=3072,directory=OUT)
    model=StopRecoveryPPO.load(MODEL/'policy_bc.zip',device='cpu');model.policy.set_training_mode(False)
    assert model.num_timesteps==0 and frozen_base_digest(model.policy)==fit['frozen_base_sha256']
    continuation=torch.load(MODEL/'continuation_state.pt',map_location='cpu',weights_only=False)
    model.policy.optimizer.load_state_dict(continuation['optimizer'])
    assert continuation['torch_rng'].dtype==torch.uint8 and continuation['numpy_rng'][0]=='MT19937'
    with np.load(MODEL/'demonstrations.npz',allow_pickle=False) as z:
        data={k:z[k].copy() for k in ('observations','masks','choices')}
    with np.load(ROOT/'models/stop_feedback_20261015/recovery_followup_bc400_s231016/demonstrations.npz',allow_pickle=False) as z:
        for key in data:np.testing.assert_array_equal(data[key][:2505],z[key])
    assert data['observations'].shape==(len(data['choices']),436)
    np.testing.assert_array_equal(masks_from_observation(data['observations'][:,:282]),data['masks'])
    assert data['masks'][:,0].all() and data['masks'][np.arange(len(data['choices'])),data['choices']].all()
    assert int((data['choices'][:2505]==0).sum())==57
    with torch.no_grad():choices=model.policy.get_distribution(torch.from_numpy(data['observations']),data['masks']).distribution.probs.argmax(-1).numpy()
    np.testing.assert_array_equal(choices,data['choices']);budget.tick()
    actor=torch.jit.load(str(bundle/'actor.pt'),map_location='cpu').eval()
    selected=np.unique(np.r_[np.flatnonzero(data['choices']==0),np.arange(2505,len(choices)),np.linspace(0,2504,128,dtype=int)])
    for i in selected:
        x=data['observations'][i];runtime=np.r_[x[:300],np.zeros(3,np.float32),x[300:]].astype(np.float32)
        with torch.no_grad():command=actor(torch.from_numpy(runtime[None])).numpy()[0]
        np.testing.assert_array_equal(command,ACTIONS[choices[i]]);budget.tick()
    write('MODEL_ADMISSION.json',dict(at=stamp(),inputs=inputs,checkpoint_sha256=fit['checkpoint_sha256'],
        optimizer_and_rng_loaded=True,all_demonstrations_reproduced=len(choices),
        exported_decisions_checked=len(selected),no_optimizer_updates=True))
    del actor,model
    physical=OUT/'physical';physical.mkdir();record_provenance(physical,bundle,[Path(__file__)])
    fresh=json.loads((BASE/'proactive_augmented_protocol_20261016/fresh_cases.json').read_text())
    cases=json.loads((BASE/'w11_recovery_bc800_navigation_20261016/cases.json').read_text())
    specs95=json.loads((BASE/'actual95_learned_validation_20261016/prefixes.json').read_text())
    specs90=json.loads((BASE/'w11_recovery_bc800_w11_20261016/specifications.json').read_text())
    report=dict(completed=False,rows=[],gates={},new_ppo_steps=0,new_optimizer_updates=0,
        full18_extra12_and_stopping_heading_regressions_still_required=True,
        actual106_not_claimed_fixed=True,goal_achieved=False,deployment_approved=False)
    def save():report['resource']=budget.statistics();write('RESULTS.json',report)
    def finish(reason):
        verify();report.update(completed=True,at=stamp(),reason=reason);save()
        write('STATUS.json',dict(at=stamp(),status='complete',reason=reason))
    def run(case,group,recovery=False):
        verify();write('STATUS.json',dict(at=stamp(),status='learned_physical_validation',group=group,case_id=case['id'],finished=len(report['rows'])))
        row,trace=(evaluate_recovery if recovery else evaluate)(case,bundle,budget)
        path=physical/(group+'_'+case['id']+'.npz');np.savez_compressed(path,**trace)
        row.update(group=group,variant='early_yield_bc400',trace_path=str(path.relative_to(ROOT)),
            trace_sha256=digest(path),learned_model_only_after_initialization=True)
        report['rows'].append(row);save();return row
    case=next(c for c in fresh if c['id']=='proactive_fresh_condition_01')
    row=run(case,'source_condition_01');teacher=source['rows'][0]
    first=bool(row['safe_success'] and row['seconds']<=teacher['seconds']*1.2+1e-8 and row['travel_m']<=teacher['travel_m']*1.2+1e-8)
    report['gates']['source_condition_safe_and_efficient']=first
    if not first:finish('Source condition did not transfer to the learned-only model.');return
    unseen=[]
    for key in ('02','04','05'):
        unseen.append(run(next(c for c in fresh if c['id']=='proactive_fresh_condition_'+key),'unfitted_conditions'))
    report['gates']['three_unfitted_conditions_safe']=all(r['safe_success'] for r in unseen)
    if not report['gates']['three_unfitted_conditions_safe']:
        finish('Unfitted previously failed conditions did not all improve safely; no further expansion.');return
    row=run(next(c for c in fresh if c['id']=='localization_nav_33'),'retained_fresh33')
    parent=next(r for r in broad['groups']['fresh12']['rows'] if r['case_id']==row['case_id'])
    gate=compare([parent],[row]);report['gates']['retained_fresh33']=gate
    if not gate['all_regression_checks_passed']:finish('Retained repaired efficiency regressed.');return
    row=run(next(c for c in cases if c['id']=='train_failure_w11_e7'),'retained_full_w11')
    parent=next(r for r in broad['full18_rows'] if r['case_id']==row['case_id'])
    gate=compare([parent],[row]);report['gates']['retained_full_w11']=gate
    if not gate['all_regression_checks_passed']:finish('Normal learned navigation regressed.');return
    for specs,release,group in ((specs95,95.,'retained_actual95'),(specs90,90.,'retained_actual90')):
        row=run(next(s for s in specs if s['release_s']==release),group,recovery=True)
        parent=next(r for r in known['rows'] if r.get('release_sensor_timestamp_s')==release)
        gate=compare([parent],[row]);report['gates'][group]=gate
        if not gate['all_regression_checks_passed']:finish('A retained learned recovery regressed.');return
    report['bounded_gates_passed']=True
    finish('Bounded learned-only gates passed; broader regressions and unresolved actual106 remain required.')


if __name__=='__main__':
    os.chdir(ROOT)
    try:main()
    except Exception as error:
        write('FAILURE.json',dict(at=stamp(),error=repr(error),automatically_restarted=False))
        raise
