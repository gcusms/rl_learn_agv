"""Audit the latest actor's actual recorded causal inputs, with no model.

Teacher initialization and learned continuation are counted separately.
This checks feature consistency and availability, not neural understanding or
whether a currently legal action would actually reach the goal safely.
"""
from collections import Counter
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
from zoneinfo import ZoneInfo

import numpy as np

from neural_navigation.decision_feedback import FEEDBACK_REASONS, GROUPS
from neural_navigation.experiment_budget import ExperimentBudget
from neural_navigation.masked_actions import COMMANDS
from neural_navigation.safety_context import REASONS


ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
BASE = HERE.parent
OUTPUT = HERE/'RECORDED_FEEDBACK_AUDIT.json'


def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    assert not OUTPUT.exists(), 'Keep this completed snapshot audit immutable'
    report_path = BASE/'full_navigation_margin_resume_validation_20261017/RESULTS.json'
    raw = report_path.read_bytes()
    report = json.loads(raw)
    assert report['model']=='models/stop_feedback_20261015/full_navigation_margin_resume_s231017'
    sources = [(name,row) for name,block in report['groups'].items() for row in block['rows']]
    assert sources
    budget = ExperimentBudget(.1,550,HERE)
    rows = []
    hashes = {str(Path(__file__).relative_to(ROOT)):digest(Path(__file__))}
    for name in ['decision_feedback.py','aligned_risk_observation.py','safety_context.py','masked_actions.py']:
        path = ROOT/'training/neural_navigation'/name
        hashes[str(path.relative_to(ROOT))] = digest(path)
    for group,row in sources:
        path = ROOT/row['trace_path']
        assert digest(path)==row['trace_sha256'],row['case_id']
        hashes[str(path.relative_to(ROOT))] = row['trace_sha256']
        with np.load(path,allow_pickle=False) as z:
            x,mask,choice,command = [z[key] for key in ['observations','masks','choices','commands']]
            labels,contact = z['labels'],z['constraint_diagnostics'][:,3:15]
            velocity = z['measured_velocity']
            controlled = z['model_controlled'] if 'model_controlled' in z else np.ones(len(choice),bool)
        n = len(choice)
        assert n==row['frames'] and x.shape==(n,625) and np.isfinite(x).all()
        assert controlled.shape==(n,) and controlled.dtype==bool
        np.testing.assert_array_equal(mask,np.c_[np.ones(n,bool),x[:,235:244]>.5,x[:,267:269]>.5])
        assert mask[np.arange(n),choice].all()
        groups = x[:,303:399].reshape(n,12,8)
        assert np.isin(groups,[0.,1.]).all()
        np.testing.assert_array_equal(groups.sum(2),np.ones((n,12)))
        np.testing.assert_array_equal(groups[:,1:,0]>.5,mask[:,1:])
        np.testing.assert_allclose(x[:,431],mask[:,1:].sum(1)/11.,rtol=0,atol=3e-8)
        np.testing.assert_array_equal(x[1:,222:224],(command[:-1]/[.2,.5]).astype(np.float32))
        np.testing.assert_allclose(x[:,220:222],np.clip(velocity/[.2,.5],-1.,1.),rtol=0,atol=3e-7)
        changed = np.max(abs(command-COMMANDS[choice]),axis=1)>=2e-8
        if changed.any():
            assert changed.sum()==1 and changed[-1] and row['safe_success']
            assert np.max(abs(command[-1]))<1e-8, 'Only a terminal arrival stop may change execution'
        short = np.zeros_like(contact,np.float32)
        finite = np.isfinite(contact)
        short[finite] = np.clip(1.-contact[finite]/1.3,0.,1.)
        np.testing.assert_allclose(x[:,419:431],short,rtol=0,atol=1e-7)
        valid = (x[:,435]>.5)&(x[:,438]>.5)
        floor = np.zeros_like(contact,np.float32)
        floor[finite] = np.clip(1.-contact[finite]/5.2,0.,1.)
        admitted = finite&valid[:,None]
        assert (x[:,467:479][admitted]>=floor[admitted]-1e-7).all()
        previous = x[:,399:419]
        assert np.isin(previous,[0.,1.]).all()
        np.testing.assert_array_equal(previous.sum(1),np.ones(n))
        np.testing.assert_array_equal(previous[0],np.eye(20,dtype=np.float32)[FEEDBACK_REASONS.index('none')])
        expected_previous = []
        for i in range(n-1):
            # The legacy context retains the previous executed guard reason.
            # An accepted zero command gains the explicit policy/only-stop
            # label in the new feedback, instead of being confused with motion.
            legacy = REASONS[int(x[i+1,250:266].argmax())]
            idle = np.max(abs(command[i]))<1e-8
            reason = legacy
            if legacy=='none' and idle:
                reason = 'policy_wait' if mask[i,1:].any() else 'only_stop_available'
            expected_previous.append(FEEDBACK_REASONS.index(reason))
        np.testing.assert_array_equal(previous[1:].argmax(1),expected_previous)
        times = labels[:,0]
        checked_history = 0
        for i,stamp in enumerate(times):
            if stamp-times[0]<.4-1e-8:continue
            for part,age in enumerate([.4,.2,0.]):
                indices = np.flatnonzero(times[:i+1]<=stamp-age+1e-8)
                j = int(indices[-1])
                assert stamp-age-times[j]<=.05+1e-8
                np.testing.assert_array_equal(x[i,439+14*part:453+14*part],x[j,467:481])
            checked_history += 1
        only_stop = ~mask[:,1:].any(1)
        stop = choice==0
        blocked = Counter()
        for k,name in enumerate(GROUPS):
            blocked[name] = int(((groups[:,1:,k]>.5)&~mask[:,1:]&controlled[:,None]).sum())
        previous_counts = Counter(FEEDBACK_REASONS[int(k)] for k in previous[controlled].argmax(1))
        rows.append(dict(case_id=row['case_id'],group=group,recovery_test=bool(row.get('recovery_test')),
            recorded_frames=n,scripted_initialization_frames=int((~controlled).sum()),learned_frames=int(controlled.sum()),
            learned_stop_with_motion_frames=int((controlled&stop&~only_stop).sum()),
            learned_only_stop_frames=int((controlled&only_stop).sum()),
            learned_motion_frames=int((controlled&~stop).sum()),
            learned_previous_reason_counts=dict(previous_counts),
            learned_nonzero_rejected_action_group_counts=dict(blocked),
            finite_short_contact_frames=int((admitted.any(1)&controlled).sum()),
            history_frames_checked=checked_history,previous_reason_frames_checked=n-1,
            trace_path=row['trace_path'],safe_success=row['safe_success']))
        budget.tick()
    assert not any(name in sys.modules for name in ['torch','mujoco'])
    snapshot = HERE/'RECORDED_FEEDBACK_VALIDATION_SNAPSHOT.json'
    snapshot.write_bytes(raw)
    output = dict(completed=True,at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),rows=rows,
        recorded_cases=len(rows),recorded_frames=sum(r['recorded_frames'] for r in rows),
        scripted_initialization_frames=sum(r['scripted_initialization_frames'] for r in rows),
        learned_frames=sum(r['learned_frames'] for r in rows),
        current_reason_groups_match_separate_action_masks=True,
        previous_execution_actions_velocities_and_reason_frames_match=True,
        guard_contact_scores_and_v2_risk_floor_match=True,physical_history_frame_ages_match=True,
        per_action_exact_reason_strings_not_separately_saved_in_physical_trace=True,
        categorical_reasons_checked_not_independent_full_guard_recomputation=True,
        feature_presence_does_not_prove_correct_neural_use_or_useful_navigation=True,
        no_neural_model_or_simulator_loaded=True,new_physical_episodes=0,new_ppo_steps=0,new_optimizer_updates=0,
        source_validation_completed_at_snapshot=report['completed'],
        source_validation_snapshot=str(snapshot.relative_to(ROOT)),source_validation_sha256=hashlib.sha256(raw).hexdigest(),
        inputs=hashes,resource=budget.statistics(),goal_achieved=False,deployment_approved=False)
    temporary = OUTPUT.with_suffix('.tmp')
    temporary.write_text(json.dumps(output,ensure_ascii=False,indent=2)+'\n')
    temporary.replace(OUTPUT)
    print(json.dumps({k:output[k] for k in ['completed','recorded_cases','recorded_frames','scripted_initialization_frames','learned_frames','resource']},ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
