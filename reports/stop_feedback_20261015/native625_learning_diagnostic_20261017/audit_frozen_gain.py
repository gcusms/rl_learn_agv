"""Audit a completed fresh-seed full-mission gain without replaying physics."""
from datetime import datetime
import hashlib
import json
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT/'reports/stop_feedback_20261015'
HERE = Path(__file__).resolve().parent


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    output = HERE/'WHOLE_MISSION_FROZEN_GAIN_AUDIT.json'
    assert not output.exists()
    study = BASE/'native625_local_speedup_retention_20261017'
    report_path = study/'RESULTS.json'
    report_bytes = report_path.read_bytes()
    report = json.loads(report_bytes)
    case_id = 'pre_ppo_holdout_dynamic_00'
    rows = {row['variant']:row for row in report['frozen_rows'] if row['case_id']==case_id}
    assert set(rows)=={'parent','candidate'}
    assert rows['parent']['timeout'] and not rows['parent']['safe_success']
    assert rows['candidate']['safe_success'] and not rows['candidate']['timeout']
    assert all(not any(row[key] for key in ['collisions','false_arrival','physical_guards']) for row in rows.values())
    traces = {}
    files = [Path(__file__),study/'PLAN.json',study/'DEPENDENCIES.json',
        BASE/'native625_frozen_holdout_20261016/CASES.json',
        BASE/'native625_frozen_holdout_20261016/PLAN.json',
        BASE/'native625_batch_credit_pilot_20261017/PLAN.json',
        BASE/'native625_batch_credit_pilot_20261017/RESULTS.json']
    for variant,row in rows.items():
        path = ROOT/row['trace_path']
        assert digest(path)==row['trace_sha256']
        with np.load(path,allow_pickle=False) as archive:
            traces[variant] = {key:archive[key].copy() for key in archive.files}
        files.append(path)
    old,new = traces['parent'],traces['candidate']
    common = min(len(old['choices']),len(new['choices']))
    changed = np.flatnonzero(old['choices'][:common]!=new['choices'][:common])
    assert len(changed)
    frame = int(changed[0])
    exact = {}
    for key in ['observations','masks','scans','odometry','measured_velocity','labels','human_positions']:
        exact[key] = bool(np.array_equal(old[key][:frame+1],new[key][:frame+1],equal_nan=True))
    exact['previous_choices'] = bool(np.array_equal(old['choices'][:frame],new['choices'][:frame]))
    exact['previous_commands'] = bool(np.array_equal(old['commands'][:frame],new['commands'][:frame]))
    assert all(exact.values()),exact
    old_choice,new_choice = int(old['choices'][frame]),int(new['choices'][frame])
    assert old['masks'][frame,old_choice] and new['masks'][frame,new_choice]
    for variant in rows:
        path = ROOT/rows[variant]['bundle']
        files.append(path/'manifest.json')
    source = ROOT/rows['parent']['bundle']
    candidate = ROOT/rows['candidate']['bundle']
    unchanged = ['hardware.json','localizer.pt','router.pt','parent_actor.pt','previous_risk_actor.pt',
        'localization_kernel.so','localization_contract.json','uncertainty_contract.json',
        'docking_contract.json','decision_feedback.json','release_observation.json',
        'long_risk_observation.json','goal_pose_observation.json','quality_observation.json']
    for name in unchanged:
        assert digest(source/name)==digest(candidate/name),name
        files += [source/name,candidate/name]
    files += [source/'actor.pt',candidate/'actor.pt']
    freeze = json.loads((BASE/'native625_frozen_holdout_20261016/PLAN.json').read_text())
    training_plan = json.loads((BASE/'native625_batch_credit_pilot_20261017/PLAN.json').read_text())
    cases_relative = 'reports/stop_feedback_20261015/native625_frozen_holdout_20261016/CASES.json'
    assert freeze['frozen_before_ppo'] and cases_relative not in training_plan['inputs']
    assert freeze['cases_sha256']==digest(ROOT/cases_relative)
    result = dict(completed=True,at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),case_id=case_id,
        completed_paired_rows=rows,current_study_not_finished_at_capture=not report['completed'],
        mutable_study_report_sha256_at_capture=hashlib.sha256(report_bytes).hexdigest(),
        first_choice_difference=dict(frame=frame,sensor_time_s=float(old['labels'][frame,0]),
            parent_choice=old_choice,candidate_choice=new_choice,
            parent_executed_command=old['commands'][frame].tolist(),
            candidate_executed_command=new['commands'][frame].tolist(),
            legal_choices=np.flatnonzero(old['masks'][frame]).tolist(),exact_common_inputs=exact),
        all_frames_model_controlled_from_initialization=True,no_fixed_physical_prefix=True,
        original12actions_625observation_hardware_4cm_masks_and_frozen_components_unchanged=True,
        actor_sha256={variant:digest(ROOT/row['bundle']/'actor.pt') for variant,row in rows.items()},
        frozen_before_ppo=True,same_known_map_new_seed=True,not_unseen_map_or_real_car=True,
        only_one_of_eight_pairs_reported_here=True,no_global_success_rate_or_optimality_claim=True,
        no_new_physics=True,no_new_optimizer_updates=True,new_ppo_steps=0,
        original106s_recovery_unresolved=True,goal_achieved=False,deployment_approved=False,
        inputs={str(path.relative_to(ROOT)):digest(path) for path in set(files)})
    output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({key:value for key,value in result.items() if key in
        ('case_id','first_choice_difference','actor_sha256')},ensure_ascii=False))


if __name__=='__main__':main()
