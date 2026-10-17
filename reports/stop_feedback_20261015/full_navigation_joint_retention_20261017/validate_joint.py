"""Full original gates, with the known sensor01 safety failure checked first.

All known6 tasks are openly reused training/development cases after joint retention. Unexecuted frozen
seeds remain conditional. No training, hidden teacher or automatic promotion.
"""
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from zoneinfo import ZoneInfo


ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent
BASE=HERE.parent
OUT=BASE/'full_navigation_joint_retention_validation_20261017'
MODEL=ROOT/'models/stop_feedback_20261015/full_navigation_joint_retention_s231017'
BASELINE=ROOT/'models/stop_feedback_20261015/normal_suite_wait_pair_v2_s231016/wait_continuation/portable'


def read(path):return json.loads(path.read_text())
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def write(path,value):
    temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');temp.replace(path)


def main():
    assert not OUT.exists();OUT.mkdir(parents=True)
    prior_unit='rl-full-navigation-joint-retention-20261017.service'
    checked=subprocess.run(['systemctl','--user','show',prior_unit,'--property=MainPID,LoadState,ActiveState,ExecMainStatus'],capture_output=True,text=True,check=True)
    state=dict(line.split('=',1) for line in checked.stdout.splitlines() if '=' in line)
    assert state['MainPID']==str(os.getppid()),'Validation requires its actual serial coordinator'
    fit_path=HERE/'FIT_RESULTS.json';fit=read(fit_path)
    assert fit['completed'] and fit['fit_passed'] and fit['export_parity_passed'] and fit['strict_runtime_admission_passed']
    assert digest(MODEL/'checkpoint.pt')==fit['checkpoint_sha256']
    tasks_path=BASE/'context_feedback_longfit_protocol_20261016/tasks.json'
    parent_path=BASE/'normal_suite_wait_validation_v2_20261016/RESULTS.json'
    known_path=BASE/'native625_local_speedup_retention_20261017/RESULTS.json'
    cases_path=BASE/'native625_frozen_holdout_20261016/CASES.json'
    fresh_dir=BASE/'memory_joint_retention_protocol_20261017'
    paths=[Path(__file__),HERE/'worker.py',HERE/'run.py',fit_path,HERE/'ADMISSION.json',HERE/'training_data.npz',HERE/'actor.py',HERE/'FIT_PLAN.json',
        MODEL/'checkpoint.pt',tasks_path,parent_path,known_path,cases_path,fresh_dir/'CASES.json',fresh_dir/'FREEZE.json']
    for folder in ['neural_navigation','experiments','navrl']:paths+=list((ROOT/'training'/folder).rglob('*.py'))
    for bundle in [MODEL/'portable',BASELINE]:
        manifest=read(bundle/'manifest.json');paths+=[bundle/'manifest.json']+[bundle/name for name in manifest['files']]
    tasks,parent,known=[read(p) for p in [tasks_path,parent_path,known_path]]
    parent_rows={r['case_id']:r for g in parent['groups'].values() for r in g['rows']}
    known_rows={r['case_id']:r for r in known['frozen_rows'] if r['variant']=='parent'}
    known_cases={r['id']:r for r in read(cases_path)}
    assert len(known_rows)==6
    priority_id='pre_ppo_holdout_sensor_01';assert priority_id in known_rows
    recoveries=[*tasks['actual95_actual90'],tasks['delayed_wait']]
    hard_reference=parent['groups']['hard3']['rows']
    hard_specs=[dict(id=r['case_id'],case=tasks['actual106']['case'],release_s=r['release_sensor_timestamp_s'],trace=r['trace_path']) for r in hard_reference]
    for spec in recoveries+hard_specs:paths.append(ROOT/spec['trace'])
    scenes={c['scene'] for c in [*tasks['normal18'],*tasks['sensor5'],*tasks['heading2'],*known_cases.values(),*read(fresh_dir/'CASES.json')]}
    scenes|={s['case']['scene'] for s in recoveries+hard_specs}
    for scene in scenes:paths += [p for p in (ROOT/'training'/scene).iterdir() if p.is_file()]
    hashes={str(p.relative_to(ROOT)):digest(p) for p in set(paths)}
    write(OUT/'PLAN.json',dict(at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),inputs=hashes,
        completed_fit_source=fit['model'],serial_coordinator_state_after_fit_child_terminal=state,
        new_ppo_steps=0,source_prior_additional_ppo_steps=1993,source_parent_auxiliary_updates=119,additional_supervised_updates=fit['accepted_supervised_updates'],
        maximum_new_physical_episodes=53,first_case=priority_id,second_case_for_regression='pre_ppo_holdout_dynamic_02',
        original_fixed28_then_remaining_known6_hard3_and_conditional8_pairs=True,
        hard3_all_safe_required_before_fresh_expansion=True,late106_never_waived=True,
        known_dynamic00_dynamic01_and_sensor01_are_training_development_references=True,
        sensor00_success_sequence_influenced_teacher=True,all_known6_now_training_development=True,source_supervised_updates=2500,
        fresh_seeds_frozen_before_this_fit_and_excluded_from_training=True,
        original4cm_body_unknown_blind_actions_dynamics_and_deadlines=True,
        per_case_time_and_travel_increase_limit=.2,total_normal_time_limit=.05,total_normal_travel_limit=.02,
        fresh_worker_process_for_every_case=True,no_parallel_physical_jobs=True,
        soft_cpu_cores=.35,memory_hard_max_mib=3072,no_automatic_promotion=True,
        goal_achieved=False,deployment_approved=False))
    from experiments.paired_navigation_gate import compare
    (OUT/'physical').mkdir();(OUT/'jobs').mkdir()
    report=dict(completed=False,model=fit['model'],groups={},gates={},fresh_rows=[],new_physical_episodes=0,
        new_ppo_steps=0,new_optimizer_updates=0,source_prior_additional_ppo_steps=1993,source_parent_auxiliary_updates=119,
        evaluated_supervised_updates=fit['accepted_supervised_updates'],
        known_dynamic00_dynamic01_and_sensor01_are_training_references=True,all_known6_now_training_development=True,source_supervised_updates=2500,goal_achieved=False,deployment_approved=False)

    def save():
        report['at']=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat();write(OUT/'RESULTS.json',report)

    def finish(reason):
        for name,expected in hashes.items():assert digest(ROOT/name)==expected,name
        assert report['new_physical_episodes']<=53
        report.update(completed=True,reason=reason,verified_input_hashes=len(hashes));save()
        write(OUT/'STATUS.json',dict(status='complete',reason=reason,completed_episodes=report['new_physical_episodes']))

    def run(spec,group,recovery=False,variant='candidate',bundle=MODEL/'portable'):
        stem=group+'_'+spec['id']+'_'+variant
        path=OUT/'jobs'/(stem+'.json');row_path=OUT/'jobs'/(stem+'_RESULT.json')
        job=dict(spec=spec,recovery=recovery,variant=variant,bundle=str(bundle),
            trace=str(OUT/'physical'/(stem+'.npz')),output=str(row_path));write(path,job)
        write(OUT/'STATUS.json',dict(status='physical_case',group=group,case_id=spec['id'],variant=variant,
            completed_episodes=report['new_physical_episodes'],worker_job=str(path.relative_to(ROOT))))
        log=OUT/'jobs'/(stem+'.log')
        with log.open('w') as stream:
            worker=subprocess.Popen([sys.executable,'-u',str(HERE/'worker.py'),str(path)],stdout=stream,stderr=subprocess.STDOUT,cwd=ROOT)
            write(OUT/'ACTIVE_WORKER.json',dict(pid=worker.pid,job=str(path.relative_to(ROOT)),log=str(log.relative_to(ROOT))))
            status=worker.wait()
        if status or not row_path.exists():raise RuntimeError(f'Worker failed with status {status}; inspect {log.relative_to(ROOT)}')
        row=read(row_path)
        if group=='fresh8':report['fresh_rows'].append(row)
        else:report['groups'].setdefault(group,dict(rows=[]))['rows'].append(row)
        report['new_physical_episodes']+=1;save()
        print(json.dumps(dict(group=group,case_id=row['case_id'],safe_success=row['safe_success'],
            collisions=row['collisions'],seconds=row['seconds'],travel_m=row['travel_m'])),flush=True)
        return row

    def safe(row):return row['safe_success'] and row['protocol_valid'] and not any(row[k] for k in ['collisions','false_arrival','physical_guards'])
    def retained(row,reference):
        key='post_release_seconds' if row.get('recovery_test') else 'seconds'
        return safe(row) and all(row[k]<=reference[k]*1.2+1e-8 for k in [key,'travel_m'])

    priority_ids=[priority_id,'pre_ppo_holdout_dynamic_02']
    for early_id in priority_ids:
        row=run(known_cases[early_id],'observed6')
        reference=known_rows[early_id]
        passed=safe(row) and (not reference['safe_success'] or retained(row,reference))
        report['gates'][early_id]=dict(passed=passed,reference_was_unsafe=not reference['safe_success'])
        if not passed:
            finish('Priority repaired safety or known dynamic02 retention failed; stop expansion');return
    fixed={g:{r['case_id']:r for r in rows} for g,rows in [
        ('normal18',tasks['reference']['normal18']),('sensor5',tasks['reference']['sensor5']),
        ('source_recovery3',tasks['reference']['actual95_actual90']+tasks['reference']['delayed_wait']),
        ('heading2',tasks['reference']['heading2'])]}
    for group,specs,recovery in [('normal18',tasks['normal18'],False),('sensor5',tasks['sensor5'],False),
        ('source_recovery3',recoveries,True),('heading2',tasks['heading2'],False)]:
        for spec in specs:
            row=run(spec,group,recovery);passed=retained(row,fixed[group][row['case_id']]) and retained(row,parent_rows[row['case_id']])
            report['gates'][row['case_id']]=dict(passed=passed)
            if not passed:finish('Fixed prior success or efficiency lost; stop expansion');return
        if group=='normal18':
            rows=report['groups'][group]['rows'];report['normal18_pair']=compare([parent_rows[r['case_id']] for r in rows],rows)
            report['normal18_against_older_fixed']=compare(tasks['reference']['normal18'],rows)
            if not(report['normal18_pair']['all_regression_checks_passed'] and report['normal18_against_older_fixed']['all_regression_checks_passed']):
                finish('Normal18 aggregate original efficiency gate failed');return
    report['all_fixed28_passed']=True;gains=[case_id for case_id in priority_ids if not known_rows[case_id]['safe_success']]
    for case_id,reference in known_rows.items():
        if case_id in priority_ids:continue
        row=run(known_cases[case_id],'observed6')
        if any(row[k] for k in ['collisions','false_arrival','physical_guards']):
            finish('Known complete-mission case still unsafe; no fresh expansion');return
        if reference['safe_success'] and not retained(row,reference):
            finish('Known complete-mission success or efficiency lost');return
        if row['safe_success'] and not reference['safe_success']:gains.append(case_id)
    for spec,reference in zip(hard_specs,hard_reference):
        row=run(spec,'hard3',True)
        if not safe(row):finish('An original hard recovery remains unsafe or unsolved; no fresh expansion');return
        if reference['safe_success'] and not retained(row,reference):finish('Prior hard recovery efficiency lost');return
        if row['safe_success'] and not reference['safe_success']:gains.append(row['case_id'])
    report['hard3_all_safe']=True;report['gained_development_successes']=gains
    if not gains:finish('No new safe arrival; keep prospective seeds unexecuted');return
    for spec in read(fresh_dir/'CASES.json'):
        for variant,bundle in [('parent',BASELINE),('candidate',MODEL/'portable')]:
            row=run(spec,'fresh8',variant=variant,bundle=bundle)
            if variant=='candidate' and any(row[k] for k in ['collisions','false_arrival','physical_guards']):
                finish('Prospective candidate unsafe; remaining pairs stopped');return
    before=[r for r in report['fresh_rows'] if r['variant']=='parent'];after=[r for r in report['fresh_rows'] if r['variant']=='candidate']
    report['fresh_seed_pair_comparison']=compare(before,after)
    finish('Declared protocol complete; full goal and deployment require separate completion audit')


if __name__=='__main__':
    try:main()
    except Exception as error:
        if OUT.exists():write(OUT/'FAILURE.json',dict(error=repr(error),no_auto_restart=True))
        raise
