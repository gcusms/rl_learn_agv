"""Count actual training exposure to short-horizon STOP-contact decisions.

An absent predicted contact is only absence within the predictor's horizon,
not proof that an alternative is safe. These are correlated sampled frames,
never an independent success rate or new positive demonstration labels.
"""
from collections import Counter
from datetime import datetime
import hashlib
import json
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    output = HERE/'TRAINING_RISK_COVERAGE.json'
    assert not output.exists()
    report_path = HERE.parent/'guard_aligned_mixed_ppo_resume_20261017/RESULTS.json'
    credit_path = HERE/'reconstructed_credit.npz'
    credit_result_path = HERE/'RESULTS.json'
    hardware_path = ROOT/'models/stop_feedback_20261015/guard_aligned_migration_s231017/portable/hardware.json'
    report = json.loads(report_path.read_text())
    credit_result = json.loads(credit_result_path.read_text())
    assert report['completed'] and report['new_ppo_steps'] == 1993
    assert digest(credit_path) == credit_result['output_credit_sha256']
    config = json.loads(hardware_path.read_text())['near_field_memory']['predictive_action_constraint']
    threshold,increment = config['minimum_check_s'],config['reaction_s']
    inputs = [Path(__file__),report_path,credit_path,credit_result_path,hardware_path]
    total,rows = Counter(),[]
    offset = 0
    with np.load(credit_path,allow_pickle=False) as credit:
        for row in report['episodes']:
            path = ROOT/row['rollout_path']
            assert digest(path) == row['rollout_sha256']
            inputs.append(path)
            with np.load(path,allow_pickle=False) as z:
                x,actions,legal = z['observations'],z['actions'],z['masks']
                ttc = z['short_guard_contacts']
                n = len(actions)
                np.testing.assert_array_equal(actions,credit['actions'][offset:offset+n])
                eligible = credit['eligible'][offset:offset+n]
                advantage = credit['normalized_advantages'][offset:offset+n]
                offset += n
                valid = (x[:,435]>.5)&(x[:,438]>.5)
                finite = np.isfinite(ttc)
                hazard = valid&finite[:,0]
                imminent = hazard&(ttc[:,0]<=threshold+1e-8)
                motion = legal[:,1:].any(1)
                decision = imminent&motion
                later_or_none = legal[:,1:]&(~finite[:,1:]|(ttc[:,1:]>ttc[:,0,None]+increment))
                has_rank_alternative = decision&later_or_none.any(1)
                chosen = ttc[np.arange(n),actions]
                sampled_rank_alternative = has_rank_alternative&(actions!=0)&(
                    ~np.isfinite(chosen)|(chosen>ttc[:,0]+increment))
                stats = dict(valid_short_contact_frames=int((valid&finite.any(1)).sum()),
                    stop_contact_frames=int(hazard.sum()),
                    stop_contact_with_motion=int((hazard&motion).sum()),
                    imminent_stop_contact_frames=int(imminent.sum()),
                    imminent_stop_contact_only_stop=int((imminent&~motion).sum()),
                    imminent_stop_contact_with_motion=int(decision.sum()),
                    imminent_stop_contact_with_trainable_choices=int((decision&eligible).sum()),
                    at_least_one_legal_later_or_no_contact_forecast=int(has_rank_alternative.sum()),
                    sampled_later_or_no_contact_motion=int(sampled_rank_alternative.sum()),
                    sampled_stop_with_imminent_contact_and_choices=int((decision&(actions==0)).sum()),
                    negative_advantage_for_those_stops=int((decision&(actions==0)&eligible&(advantage<0)).sum()),
                    positive_advantage_for_those_stops=int((decision&(actions==0)&eligible&(advantage>0)).sum()))
                total.update(stats)
                rows.append(dict(batch=row['batch'],release_frame=row['release_frame'],
                    full_navigation_case=row['full_navigation_case'],source=row['rollout_path'],
                    source_training_success=row['safe_success'],counts=stats))
    assert offset == 1993
    result = dict(completed=True,at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
        sampled_decisions=offset,imminent_horizon_s=threshold,forecast_difference_s=increment,
        counts=dict(total),episodes=rows,
        only_correlated_training_frames_not_independent_tasks=True,
        forecast_difference_is_not_a_safety_or_ground_truth_label=True,
        no_contact_does_not_certify_clear_space=True,
        no_new_rewards_or_demonstrations_or_runtime_changes=True,
        new_ppo_steps=0,new_optimizer_updates=0,new_physical_episodes=0,
        inputs={str(p.relative_to(ROOT)):digest(p) for p in inputs},
        goal_achieved=False,deployment_approved=False)
    output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(result['counts'],ensure_ascii=False))


if __name__=='__main__':
    main()
