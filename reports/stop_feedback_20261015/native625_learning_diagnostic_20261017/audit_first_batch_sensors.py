"""Audit actual newly logged sensors against policy input and candidate masks."""
import hashlib
import json
from pathlib import Path

import numpy as np

from neural_navigation.decision_feedback import GROUPS,reason_group
from neural_navigation.experiment_budget import ExperimentBudget
from neural_navigation.masked_actions import COMMANDS
from neural_navigation.release_observation import scan_quality
from neural_navigation.sensors import LidarSpec


ROOT=Path(__file__).resolve().parents[3]
BASE=ROOT/'reports/stop_feedback_20261015'
HERE=Path(__file__).resolve().parent


def main():
    output=HERE/'FIRST_BATCH_SENSOR_AUDIT.json'
    assert not output.exists()
    budget=ExperimentBudget(.1,memory_mib=600,directory=HERE)
    report=json.loads((BASE/'native625_batch_credit_pilot_20261017/RESULTS.json').read_text())
    row=report['episodes'][0]
    path=ROOT/row['rollout_path']
    assert hashlib.sha256(path.read_bytes()).hexdigest()==row['rollout_sha256']
    with np.load(path,allow_pickle=False) as saved:
        data={key:saved[key].copy() for key in saved.files}
    hardware=ROOT/'models/stop_feedback_20261015/normal_suite_wait_pair_v2_s231016/wait_continuation/portable/hardware.json'
    lidar=LidarSpec(json.loads(hardware.read_text())['lidar'])
    n=len(data['actions'])
    assert data['raw_scans'].shape==(n,lidar.beams)
    assert data['input_odometry'].shape==(n,3) and np.isfinite(data['input_odometry']).all()
    assert data['input_measured_velocity'].shape==(n,2) and np.isfinite(data['input_measured_velocity']).all()
    quality=np.asarray([scan_quality(scan,lidar) for scan in data['raw_scans']])
    np.testing.assert_array_equal(quality,data['observations'][:,481:625])
    accepted=np.isclose(data['per_action_guard_executed'],COMMANDS[None],rtol=0,atol=1e-9).all(2)
    expected_groups=np.zeros((n,12),np.int64)
    for i in range(n):
        for action in range(12):
            reason=data['per_action_guard_reasons'][i,action]
            group=reason_group(None if reason=='None' else reason)
            expected_groups[i,action]=0 if accepted[i,action] else (group or len(GROUPS)-1)
    np.testing.assert_array_equal(expected_groups,data['observations'][:,303:399].reshape(n,12,8).argmax(2))
    accepted[:,0]=True
    np.testing.assert_array_equal(accepted,data['masks'])
    first_native=BASE/'native625_suite_ppo_20261016/episode_01.npz'
    with np.load(first_native,allow_pickle=False) as old:
        for key in ['observations','masks','actions','executed_commands','old_log_probability','old_values','labels','rewards']:
            np.testing.assert_array_equal(data[key],old[key])
    snapshot=HERE/'FIRST_BATCH_ROW.json'
    assert not snapshot.exists()
    snapshot.write_text(json.dumps(row,indent=2)+'\n')
    files=[Path(__file__),snapshot,path,hardware,first_native]
    budget.tick()
    result=dict(completed=True,frames=n,raw_lidar_beams=lidar.beams,
        scan_quality_matches_actual_actor_input=True,
        per_action_raw_reasons_match_current_reason_features=True,
        raw_guard_execution_matches_policy_mask=True,
        first_episode_matches_original_preupdate_baseline_byte_exact=True,
        training_safe_success=row['safe_success'],
        live_optimizer_updates_when_observed=report['new_actor_optimizer_updates'],
        independent_geometry_accuracy_not_proven=True,training_not_independent_evaluation=True,
        inputs={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
        resource=budget.statistics(),no_new_physics=True,no_optimizer_updates=True,
        new_ppo_steps=0,goal_achieved=False,deployment_approved=False)
    output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='inputs'},ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
