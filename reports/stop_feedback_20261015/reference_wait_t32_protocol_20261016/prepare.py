"""Pin one useful-action counterfactual at the first observed BC disagreement."""
from datetime import datetime
import hashlib
import json
from pathlib import Path
import subprocess
from zoneinfo import ZoneInfo

import numpy as np


ROOT=Path(__file__).resolve().parents[3]
OUT=Path(__file__).resolve().parent
BASE=OUT.parent


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    if (OUT/'PLAN.json').exists():
        raise ValueError('Do not overwrite a predeclared counterfactual')
    protocol=BASE/'post_aux_diagnostics_fixed_20261016'
    predecessor=json.loads((protocol/'PLAN.json').read_text())
    for name,value in predecessor['inputs'].items():
        if digest(ROOT/name)!=value:
            raise ValueError('Earlier diagnostic input changed')
    service='rl-post-aux-fixed-20261016.service'
    live=dict(v.split('=',1) for v in subprocess.check_output([
        'systemctl','--user','show',service,'-p','MainPID','-p','ActiveState','-p','InvocationID'],text=True).splitlines())
    if (live['MainPID']=='0' or live['ActiveState']!='active'
            or live['InvocationID']!='c60bf2973f184188ac26018fd1539489'):
        raise ValueError('The exact predecessor must be confirmed live before queueing')
    folder=BASE/'w11_control2000_navigation_20261016'
    report=json.loads((folder/'RESULTS.json').read_text())
    row,=[r for r in report['rows'] if r['case_id']=='train_failure_w11_e7']
    if not report['completed'] or not row['timeout'] or row['collisions']:
        raise ValueError('Expected complete non-colliding control timeout')
    trace=folder/(row['case_id']+'_'+row['variant']+'.npz')
    if digest(trace)!=row['trace_sha256']:
        raise ValueError('Source physical trace changed')
    audit=BASE/'stop_support_diagnosis_20261016'
    diagnosis=json.loads((audit/'RESULTS.json').read_text())
    if not diagnosis['completed']:
        raise ValueError('Completed action/support diagnosis required')
    with np.load(trace,allow_pickle=False) as z,np.load(audit/'predictions.npz',allow_pickle=False) as predictions:
        obs=z['observations'];masks=z['masks'].astype(bool)
        selected=(np.max(abs(z['commands']),axis=-1)<1e-8)&masks[:,1:].any(-1)&(np.linalg.norm(obs[:,218:220]*12.,axis=-1)>.3)
        indices=np.flatnonzero(selected)
        bc=predictions['reference_bc__control_full_w11__probabilities']
        control=predictions['control2k__control_full_w11__probabilities']
        if len(bc)!=len(indices):
            raise ValueError('Prediction and actual stopped-state order differ')
        eligible=(bc.argmax(-1)!=0)&(control.argmax(-1)==0)&(obs[indices,434]*5.>=2.-1e-6)
        k=int(np.flatnonzero(eligible)[0]);i=int(indices[k])
        release=round(float(z['labels'][i,0]),6)
        if release!=32. or i!=160:
            raise ValueError('The independently diagnosed first disagreement changed')
        selection=dict(source_frame=i,sensor_timestamp_s=release,
            previous_idle_with_motion_s=float(obs[i,434]*5.),
            legal_actions=np.flatnonzero(masks[i]).tolist(),
            reference_choice=int(bc[k].argmax()),reference_choice_probability=float(bc[k].max()),
            control_choice=0,control_stop_probability=float(control[k,0]),
            selection_rule='First chronological actual legal idle>=2s with referenceBC greedy motion and actual control greedy stop.',
            no_future_navigation_success_used_for_selection=True)
    case,=[c for c in json.loads((folder/'cases.json').read_text()) if c['id']==row['case_id']]
    specification=dict(id='control_w11_wait_t32',case=case,release_s=release,
        trace=str(trace.relative_to(ROOT)))
    (OUT/'prefix.json').write_text(json.dumps([specification],indent=2)+'\n')
    files=[OUT/'prepare.py',OUT/'after_aux.py',OUT/'prefix.json',folder/'RESULTS.json',
        folder/'cases.json',trace,audit/'RESULTS.json',audit/'predictions.npz',protocol/'PLAN.json']
    inputs=dict(predecessor['inputs'])
    for p in files:inputs[str(p.relative_to(ROOT))]=digest(p)
    plan=dict(created_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        dependency_unit=service,dependency_invocation=live['InvocationID'],
        dependency_report=str((protocol/'RESULTS.json').relative_to(ROOT)),
        dependency_live_at_prepare=live,selection=selection,
        hypothesis='A complete original-guard continuation can determine whether the BC-preferred legal turn is useful in an actual PPO-induced long wait.',
        bundle='models/stop_feedback_20261015/w11_recovery_bc800_s231015/bc_portable',
        output='reports/stop_feedback_20261015/reference_control_wait_t32_20261016',
        source_baseline_complete_timeout=row,
        new_physical_continuations=1,forced_post_release_actions=0,
        paired_control_rerun=False,source_baseline_reused_not_counted_again=True,
        original_masks_guard_rewards_hardware_unchanged=True,
        training_history_rehearsal_not_unseen_mission=True,
        resource=dict(soft_cpu_cores=.15,hard_cpu_percent=20,nice=19,
            memory_high_gib=2.5,memory_max_gib=3,no_third_physical_simulation=True),
        inputs=inputs,no_automatic_expansion_training_or_promotion=True,
        goal_achieved=False,deployment_approved=False)
    (OUT/'PLAN.json').write_text(json.dumps(plan,indent=2)+'\n')
    print(json.dumps(dict(prepared=True,source_time_s=release,source_frame=i,
        new_physical_continuations=1,pinned_inputs=len(inputs),selection=selection)))


if __name__=='__main__':
    main()
