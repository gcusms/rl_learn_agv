"""Combine fixed tasks with explicitly identified cached baseline provenance."""
import copy
import hashlib
import json
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parents[2]
RUNS=ROOT/'training/runs'

def main():
    final=RUNS/'docking_full_evaluation_20261014'
    old=RUNS/'goal_pose_paired_evaluation_20261014'
    extra=RUNS/'docking_parent_new_tasks_20261014'
    output=RUNS/'docking_matched_comparison_20261014'
    all_reports=[json.loads((p/'RESULTS.json').read_text()) for p in (final,old,extra)]
    assert all(r['completed'] for r in all_reports)
    final_cases=json.loads((final/'cases.json').read_text())
    old_cases=json.loads((old/'cases.json').read_text())
    extra_cases=json.loads((extra/'cases.json').read_text())
    assert final_cases[:len(old_cases)]==old_cases
    assert final_cases[len(old_cases):]==extra_cases
    parent=ROOT/'models/goal_pose_pilot_20261014/policy_3000.zip'
    assert hashlib.sha256(parent.read_bytes()).hexdigest()=='d1a6b8b0d903e631f359038d161eb8f0ea971893490ee515c124c918fe727969'
    output.mkdir(exist_ok=False);rows=[]
    for source,report,selected in [(final,all_reports[0],None),(old,all_reports[1],'trained'),(extra,all_reports[2],'trained')]:
        for original in report['rows']:
            if selected and original['variant']!=selected:continue
            row=copy.deepcopy(original);f=source/f"{row['case_id']}_{row['variant']}.npz"
            assert hashlib.sha256(f.read_bytes()).hexdigest()==row['trace_sha256']
            if selected:row['variant']='previous_3k'
            row['provenance']=dict(source=str(source.relative_to(ROOT)),historical_cached=source==old)
            shutil.copy2(f,output/f"{row['case_id']}_{row['variant']}.npz");rows.append(row)
    assert len(rows)==84 and len({(r['case_id'],r['variant']) for r in rows})==84
    groups={}
    for category in sorted({r['category'] for r in rows}):
        groups[category]={}
        for variant in sorted({r['variant'] for r in rows}):
            selected=[r for r in rows if r['category']==category and r['variant']==variant]
            if selected:groups[category][variant]=dict(tasks=len(selected),success=sum(r['safe_success'] for r in selected),
                collisions=sum(r['collisions'] for r in selected),timeouts=sum(r['timeout'] for r in selected),false_arrivals=sum(r['false_arrival'] for r in selected))
    case_file=output/'cases.json';shutil.copy2(final/'cases.json',case_file)
    result=dict(completed=True,scope='28 fixed development tasks on training maps; 72 fresh runs and 12 reused parent baseline runs; not final acceptance',
        deployment_approved=False,baseline_checkpoint_sha256=hashlib.sha256(parent.read_bytes()).hexdigest(),
        cases_sha256=hashlib.sha256(case_file.read_bytes()).hexdigest(),rows=rows,groups=groups,
        baseline_provenance=dict(fresh_tasks=16,cached_tasks=12,case_parameters_and_trace_hashes_verified=True))
    (output/'RESULTS.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(groups,indent=2))

if __name__=='__main__':main()
