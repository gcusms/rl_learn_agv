"""Check the new causal feedback on a frozen failure, without a steering teacher."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from navrl.geometry import ROOT
from neural_navigation.goal_pose_runtime import DockingRuntime
from neural_navigation.decision_feedback_runtime import StopFeedbackRuntime
from neural_navigation.decision_feedback import GROUP_DIM, RISK_SLICE, REASON_SLICE, FEEDBACK_REASONS
from neural_navigation.masked_actions import COMMANDS,masks_from_observation
from neural_navigation.experiment_budget import ExperimentBudget
from experiments.stop_feedback_navigation import FeedbackInterface
from experiments.stop_feedback_ppo import warm_start
from experiments.train_stop_feedback import export_candidate,digest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('baseline','checkpoint','trace','cases','output'):parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1);budget=ExperimentBudget(.35,memory_mib=1536,directory=args.output)
    model=warm_start(args.checkpoint,FeedbackInterface())
    candidate=args.output/'initial_portable';export_candidate(model,args.baseline,candidate)
    case=next(c for c in json.loads(args.cases.read_text()) if c['id']=='motion_profile_015')
    scene=ROOT/case['scene'];prior=np.asarray(case['start'])+np.asarray(case.get('initial_prior_offset',[0.,0.,0.]))
    before=DockingRuntime(args.baseline,scene/'decision_map.yaml',prior,case['goal'],'shortcut')
    after=StopFeedbackRuntime(candidate,scene/'decision_map.yaml',prior,case['goal'],'shortcut')
    rows=[];previous='none';max_risk_error=0.;prefix_difference=0.;changed_commands=0
    with np.load(args.trace) as data:
        for i in range(len(data['labels'])):
            t=float(data['labels'][i,0]);inputs=(data['scans'][i],data['odometry'][i],data['measured_velocity'][i],t)
            old=before.step(*inputs,capture_observation=True);new=after.step(*inputs,capture_observation=True)
            x=np.asarray(new['policy_observation']);prefix=np.asarray(old['policy_observation'])
            prefix_difference=max(prefix_difference,float(abs(x[:303]-prefix).max()))
            changed_commands+=int(not np.array_equal(old['command'],new['command']))
            constraint=after.guard.predictive_constraint
            risk=np.zeros(12);finite=np.isfinite(constraint.contacts)
            risk[finite]=1-constraint.contacts[finite]/(constraint.config['horizon_s']+constraint.config['period_s'])
            max_risk_error=max(max_risk_error,float(abs(x[303+RISK_SLICE.start:303+RISK_SLICE.stop]-risk).max()))
            feedback=after.policy.feedback
            reason=FEEDBACK_REASONS[int(x[303+REASON_SLICE.start:303+REASON_SLICE.stop].argmax())]
            assert reason==previous,(i,reason,previous)
            previous=feedback.previous_reason
            mask=masks_from_observation(x[:282]);assert np.array_equal(mask,feedback.frame_mask)
            rows.append(dict(timestamp=t,legal_moving=int(mask[1:].sum()),
                previous_reason=reason,executed_reason=previous,command=new['command'],
                legacy_stop_risk=float(x[282]),shared_stop_risk=float(x[303+RISK_SLICE.start]),
                rejection_reasons=new['current_action_rejection_reasons']))
            budget.tick()
    assert prefix_difference==0. and changed_commands==0
    assert max_risk_error<1e-7
    result=dict(completed=True,scope='Fixed sensor explanation/parity check; no new navigation outcome.',
        frames=len(rows),legacy_prefix_max_difference=prefix_difference,changed_initial_commands=changed_commands,
        shared_forecast_max_difference=max_risk_error,causal_previous_reasons_correct=True,
        trace_sha256=digest(args.trace),rows=rows,resource=budget.statistics(),deployment_approved=False)
    (args.output/'RESULTS.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='rows'}),flush=True)


if __name__=='__main__':main()
