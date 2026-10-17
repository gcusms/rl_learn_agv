"""Add complete, safe development references with matched v2 input semantics.

This explicitly turns two already observed cases into training data. They are
no longer independent tests; the unexecuted frozen seed set stays excluded.
"""
from datetime import datetime
import hashlib
import json
from pathlib import Path
import runpy
from zoneinfo import ZoneInfo

import numpy as np

from neural_navigation.aligned_risk_observation import fuse_current_frame, contract
from neural_navigation.experiment_budget import ExperimentBudget


ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent
BASE=HERE.parent


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path,value):
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');tmp.replace(path)


def main():
    assert not (HERE/'DATA.json').exists() and not (HERE/'demonstrations.npz').exists()
    original_path=BASE/'guard_forecast_alignment_protocol_20261017/aligned_demonstrations.npz'
    original_report_path=BASE/'guard_forecast_alignment_protocol_20261017/ALIGNED_DEMONSTRATIONS.json'
    old_report_path=BASE/'native625_local_speedup_retention_20261017/RESULTS.json'
    current_report_path=BASE/'guard_aligned_mixed_ppo_resume_validation_20261017/RESULTS.json'
    original_report=json.loads(original_report_path.read_text())
    assert original_report['data_preflight_passed'] and digest(original_path)==original_report['archive_sha256']
    old,current=[json.loads(p.read_text()) for p in [old_report_path,current_report_path]]
    assert current['completed'] and current['new_physical_episodes']==30
    first=next(r for r in old['frozen_rows'] if r['case_id']=='pre_ppo_holdout_dynamic_01' and r['variant']=='parent')
    second=next(r for r in current['groups']['observed6']['rows'] if r['case_id']=='pre_ppo_holdout_dynamic_00')
    history_path=BASE/'guard_forecast_alignment_protocol_20261017/causal_history.py'
    CausalHistory=runpy.run_path(str(history_path))['CausalHistory']
    paths=[Path(__file__),original_path,original_report_path,old_report_path,current_report_path,history_path,
        ROOT/'training/neural_navigation/aligned_risk_observation.py']
    budget=ExperimentBudget(.35,600,HERE)
    with np.load(original_path,allow_pickle=False) as z:
        parts=[dict(observations=z['observations'].copy(),actions=z['actions'].copy(),masks=z['masks'].copy(),
            source_group=np.zeros(len(z['actions']),np.int32))]
    assert len(parts[0]['actions'])==3799
    source_rows=[]
    for number,(row,old_semantics) in enumerate([(first,True),(second,False)],1):
        assert row['safe_success'] and row['physical_arrival'] and row['protocol_valid']
        assert not any(row[k] for k in ['collisions','false_arrival','physical_guards','timeout'])
        assert not row.get('recovery_test')
        path=ROOT/row['trace_path'];assert digest(path)==row['trace_sha256'];paths.append(path)
        with np.load(path,allow_pickle=False) as z:
            x=z['observations'].copy();actions=z['choices'].copy();masks=z['masks'].copy()
            times=z['labels'][:,0];diagnostics=z['constraint_diagnostics']
        original=x.copy();raw_history,fused_history=CausalHistory(),CausalHistory()
        for i,timestamp in enumerate(times):
            np.testing.assert_array_equal(raw_history.append(original[i,467:481],float(timestamp)),original[i,439:481])
            if old_semantics:
                valid=bool(original[i,435]>.5 and original[i,438]>.5)
                fused=fuse_current_frame(original[i,467:481],diagnostics[i,3:],int(diagnostics[i,0]),
                    float(timestamp),float(timestamp),input_valid=valid,scan_valid=valid)
                x[i,439:481]=fused_history.append(fused,float(timestamp))
            budget.tick()
        np.testing.assert_array_equal(x[:,:439],original[:,:439])
        np.testing.assert_array_equal(x[:,481:],original[:,481:])
        assert masks[np.arange(len(actions)),actions].all()
        parts.append(dict(observations=x,actions=actions,masks=masks,source_group=np.full(len(actions),number,np.int32)))
        source_rows.append(dict(group=number,case_id=row['case_id'],source=row['trace_path'],source_sha256=digest(path),
            full_mission_success_reference=True,old_v1_history_causally_converted=old_semantics,
            admitted_frames=len(actions),stop_labels=int((actions==0).sum()),
            changed_history_frames=int(np.any(x!=original,axis=1).sum()),
            now_training_development_case_not_independent_evaluation=True))
    arrays={k:np.concatenate([p[k] for p in parts]) for k in parts[0]}
    seen={};conflicts=[]
    for i,(observation,action) in enumerate(zip(arrays['observations'],arrays['actions'])):
        key=observation.tobytes()
        if key in seen and seen[key][1]!=int(action):conflicts.append([seen[key][0],i])
        else:seen[key]=(i,int(action))
    assert not conflicts,'Inspect conflicting references before fitting'
    assert len(arrays['actions'])==4282
    np.testing.assert_array_equal(arrays['observations'][:3799],parts[0]['observations'])
    archive=HERE/'demonstrations.npz';np.savez_compressed(archive,**arrays);budget.tick()
    write(HERE/'DATA.json',dict(completed=True,at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
        admitted_rows=len(arrays['actions']),original_reference_rows=3799,
        original_reference_stop_labels=int((parts[0]['actions']==0).sum()),
        total_stop_labels=int((arrays['actions']==0).sum()),source_rows=source_rows,
        exact_input_conflicts=conflicts,illegal_labels=0,old_original_rows_byte_identical=True,
        input_risk_semantics=contract(),same_current_scan_for_all_components=True,
        initialization_and_failed_episodes_excluded=True,fresh_seed_set_excluded=True,
        archive_sha256=digest(archive),inputs={str(p.relative_to(ROOT)):digest(p) for p in paths},
        resource=budget.statistics(),new_ppo_steps=0,new_optimizer_updates=0,new_physical_episodes=0,
        goal_achieved=False,deployment_approved=False))
    print(json.dumps(dict(rows=len(arrays['actions']),stops=int((arrays['actions']==0).sum()),sources=source_rows),ensure_ascii=False))


if __name__=='__main__':main()
