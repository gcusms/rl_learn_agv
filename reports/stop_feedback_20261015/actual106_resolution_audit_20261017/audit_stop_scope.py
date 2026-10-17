"""Check whether this PPO update was directly locked to the failed STOP rows."""
from datetime import datetime
import hashlib
import json
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT/'reports/stop_feedback_20261015'
OUT = Path(__file__).resolve().parent/'STOP_RETENTION_SCOPE_AUDIT.json'


def main():
    assert not OUT.exists()
    anchors_path = BASE/'normal_suite_wait_protocol_v2_20261016/demonstrations.npz'
    failure_path = BASE/'context_broad_diagnostic_20261016/physical/hard3_w11_training_stall_t106.npz'
    training_path = BASE/'native625_batch_credit_pilot_20261017/RESULTS.json'
    admission_path = BASE/'normal_suite_wait_protocol_v2_20261016/SOURCE_ADMISSION.json'
    paths = [Path(__file__),anchors_path,failure_path,training_path,admission_path]
    inputs = {str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    with np.load(anchors_path,allow_pickle=False) as data:
        x,labels,masks = [data[key].copy() for key in ['observations','choices','masks']]
    with np.load(failure_path,allow_pickle=False) as data:
        failed = data['observations'][530:].copy()
    stop = labels==0
    anchor_rows = {row.tobytes() for row in x[stop]}
    overlaps = sum(row.tobytes() in anchor_rows for row in failed)
    training = json.loads(training_path.read_text())
    updates = training['updates']
    fractions = [float(row.get('accepted_actor_step_fraction',0.)) for row in updates]
    assert len(x)==7378 and int(stop.sum())==1201 and len(failed)==170
    result = dict(completed=True,at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),inputs=inputs,
        reference_decisions=len(x),reference_stop_labels=int(stop.sum()),
        reference_stops_with_nonzero_legal_choices=int((stop&masks[:,1:].any(1)).sum()),
        failed106_suffix_frames=len(failed),exact_failed_observation_stop_anchor_overlaps=overlaps,
        proposed_updates=len(updates),attempted_actor_updates=sum(bool(row['actor_update_attempted']) for row in updates),
        accepted_actor_updates=sum(bool(row['actor_updated']) for row in updates),
        full_parameter_steps=sum(fraction==1. for fraction in fractions),
        shortened_parameter_steps=sum(0.<fraction<1. for fraction in fractions),
        mean_accepted_fraction_over_all_updates=float(np.mean(fractions)),
        interpretation='The observed updates do not support a wholesale hard-STOP-lock explanation; retain safety anchors.',
        limitations=['Byte-exact overlap is not a learned-feature or generalization independence proof.',
            'Successful trajectories can require deliberate waiting despite legal moving choices.',
            'Update counts and accepted fractions are not a physical outcome or sufficient learning proof.'],
        no_anchor_or_reward_or_training_change=True,new_physical_steps=0,new_ppo_steps=0,
        no_optimizer_updates=True,goal_achieved=False,deployment_approved=False)
    OUT.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({key:value for key,value in result.items() if key!='inputs'},ensure_ascii=False))


if __name__=='__main__':main()
