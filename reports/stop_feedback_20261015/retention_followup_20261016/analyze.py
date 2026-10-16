"""Compare completed outcomes, actual waiting, and the first changed decisions."""
from datetime import datetime
import hashlib
import json
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import torch

from experiments.audit_decision_waits import measure
from experiments.paired_navigation_gate import compare
from experiments.stop_recovery_policy import StopRecoveryPPO
from neural_navigation.goal_pose import contract as goal_contract


ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
BASE = ROOT / 'reports/stop_feedback_20261015'
INPUTS = {}


def pin(path):
    path = path.resolve()
    data = path.read_bytes()
    INPUTS[str(path.relative_to(ROOT))] = hashlib.sha256(data).hexdigest()
    return data


def read_report(folder):
    path = BASE / folder / 'RESULTS.json'
    report = json.loads(pin(path))
    if not report['completed']:
        raise ValueError('Use complete evaluations only: ' + folder)
    cases = None
    if (path.parent / 'cases.json').exists():
        cases = {r['id']:r for r in json.loads(pin(path.parent / 'cases.json'))}
    return report, cases


def summary(rows):
    return dict(rows=len(rows), safe=sum(bool(r['safe_success']) for r in rows),
        collisions=sum(int(r['collisions']) for r in rows),
        timeouts=sum(bool(r['timeout']) for r in rows),
        failed_case_ids=[r['case_id'] for r in rows if not r['safe_success']])


def trace(folder, row):
    path = BASE / folder / (row['case_id'] + '_' + row['variant'] + '.npz')
    if hashlib.sha256(pin(path)).hexdigest() != row['trace_sha256']:
        raise ValueError('Trace mismatch: ' + str(path))
    with np.load(path, allow_pickle=False) as data:
        return {name:data[name].copy() for name in data.files}


def first_change(left, right):
    n = min(len(left['choices']), len(right['choices']))
    changed = np.flatnonzero(left['choices'][:n] != right['choices'][:n])
    if not len(changed):
        return dict(first_changed_frame=None, compared_frames=n), None
    i = int(changed[0])
    parity = {}
    for key in ['observations','masks','scans','odometry','measured_velocity']:
        parity[key] = bool(np.allclose(left[key][:i+1],right[key][:i+1],
            atol=2e-8,rtol=0,equal_nan=True))
    parity['physical_pose'] = bool(np.allclose(left['labels'][:i+1,1:4],
        right['labels'][:i+1,1:4],atol=2e-8,rtol=0))
    if not all(parity.values()):
        raise ValueError('A changed policy choice has a different causal history')
    return dict(first_changed_frame=i, sensor_time_s=float(left['labels'][i,0]),
        left_choice=int(left['choices'][i]), right_choice=int(right['choices'][i]),
        legal_choices=np.flatnonzero(left['masks'][i]).tolist(), history_parity=parity), (
        left['observations'][i].copy(),left['masks'][i].copy())


def main():
    if (OUT / 'RESULTS.json').exists():
        raise FileExistsError('Analysis is immutable')
    torch.set_num_threads(1)
    pin(Path(__file__))
    for name in ['training/experiments/audit_decision_waits.py',
                 'training/experiments/paired_navigation_gate.py',
                 'training/experiments/stop_recovery_policy.py',
                 'training/experiments/proactive_recovery.py']:
        pin(ROOT / name)
    folders = dict(reference='w11_recovery_bc800_navigation_20261016',
        control='w11_control2000_navigation_20261016',
        retained='w11_retained2000_navigation_20261016')
    reports, cases = {}, {}
    for name,folder in folders.items():
        reports[name],cases[name] = read_report(folder)
    if not (cases['reference'] == cases['control'] == cases['retained']):
        raise ValueError('Identical physical cases required')
    rows = {name:{r['case_id']:r for r in report['rows']} for name,report in reports.items()}
    ids = list(rows['reference'])
    if len(ids) != 18 or any(set(group)!=set(ids) for group in rows.values()):
        raise ValueError('All original 18 task identities required')
    original = [name for name in ids if not name.startswith('recovery_holdout_')]
    directional = [name for name in ids if len(cases['reference'][name]['goal'])==3]
    if len(original)!=14 or len(directional)!=6:
        raise ValueError('Original-success and direction suites changed')
    gates = {}
    waits = {}
    traces = {}
    for name,folder in folders.items():
        if name!='reference':
            gates[name] = dict(full=compare(list(rows['reference'].values()),list(rows[name].values())),
                original14=compare([rows['reference'][i] for i in original],[rows[name][i] for i in original]),
                directional6=summary([rows[name][i] for i in directional]))
        waits[name] = {}
        for case_id in ids:
            row = rows[name][case_id]
            data = trace(folder,row)
            goal = cases[name][case_id]['goal']
            tolerance = goal_contract()['position_tolerance_m'] if len(goal)==3 else .2
            yaw = goal_contract()['yaw_tolerance_rad'] if len(goal)==3 else None
            waits[name][case_id] = measure(data,row['seconds'],tolerance,yaw_tolerance=yaw)
            if case_id in ['motion_profile_002','train_failure_w11_e7']:
                traces[name,case_id] = data

    changes = {}
    replay = {}
    for label,left,right,case_id in [
        ('w11_control_to_retained','control','retained','train_failure_w11_e7'),
        ('profile002_reference_to_retained','reference','retained','motion_profile_002')]:
        changes[label],state = first_change(traces[left,case_id],traces[right,case_id])
        changes[label].update(left=left,right=right,case_id=case_id)
        if state is not None:
            observation,mask = state
            if observation.shape != (439,):
                raise ValueError('Expected directional feedback runtime observation')
            observation = np.r_[observation[:300],observation[303:]].astype(np.float32)
            replay[label] = observation,mask
    probabilities = {}
    for name,folder,checkpoint in [
        ('reference','w11_recovery_bc800_s231015','policy_bc.zip'),
        ('control','w11_control2000_s231015','policy.zip'),
        ('retained','w11_retained2000_s231015','policy.zip')]:
        path = ROOT/'models/stop_feedback_20261015'/folder/checkpoint
        pin(path)
        model = StopRecoveryPPO.load(path,device='cpu')
        probabilities[name] = {}
        for label,(observation,mask) in replay.items():
            with torch.no_grad():
                p = model.policy.get_distribution(torch.from_numpy(observation[None]),
                    mask[None]).distribution.probs.cpu().numpy()[0]
            if not np.isfinite(p).all() or np.any(p[~mask.astype(bool)]>1e-7):
                raise ValueError('Invalid masked probabilities')
            choice = int(p.argmax())
            probabilities[name][label] = dict(choice=choice,probabilities=p.tolist())
            change = changes[label]
            if name == change['left'] and choice != change['left_choice']:
                raise ValueError('Left physical decision did not replay')
            if name == change['right'] and choice != change['right_choice']:
                raise ValueError('Right physical decision did not replay')
        del model

    auxiliary = json.loads(pin(BASE/'post_aux_diagnostics_fixed_20261016/RESULTS.json'))
    if not auxiliary['completed'] or len(auxiliary['candidate_rows'])!=6:
        raise ValueError('Six completed diagnostic cases required')
    for name in ['early_transfer_diagnostic_navigation_20261016',
                 'early_transfer_diagnostic_reference_20261016']:
        report,_ = read_report(name)
        for row in report['rows']:
            trace(name,row)
    auxiliary_gate = compare(auxiliary['reference_rows'],auxiliary['candidate_rows'])
    if auxiliary_gate != auxiliary['paired_navigation']:
        raise ValueError('Auxiliary paired outcome did not reproduce')

    result = dict(completed=True,at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
        summaries={name:summary(list(group.values())) for name,group in rows.items()},
        gates=gates,actual_waits=waits,first_changed_decisions=changes,
        first_decision_model_replay=probabilities,
        diagnostic_six=dict(candidate=summary(auxiliary['candidate_rows']),
            reference=summary(auxiliary['reference_rows']),paired=auxiliary_gate,
            original_quick_admission_remains_failed=True,
            not_the_original14_or_direction_acceptance=True),
        inputs=INPUTS,new_environment_steps=0,new_ppo_steps=0,
        full35_retained_comparison_complete=False,goal_achieved=False,deployment_approved=False,
        limitations=['Full-start outcome improvement is not proof of late-wait recovery.',
            'First changed decisions share causal history, but all later actions may differ.',
            'Six diagnostic cases are reused development/training cases, not unseen-map generalization.',
            'Both PPO models lose an originally successful task; neither is admitted as a deployment upgrade.'])
    (OUT/'RESULTS.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(completed=True,summaries=result['summaries'],
        original14_lost={name:g['original14']['lost_successes'] for name,g in gates.items()},
        first_changed_decisions=changes,diagnostic_six=result['diagnostic_six'],inputs=len(INPUTS)),
        ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
