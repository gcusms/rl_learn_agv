"""Native parity between PPO curriculum and exported sensor-stream execution."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from experiments.recovery_sampling import ScheduledRecoveryCurriculum
from experiments.quality_masked_navigation import QualityMaskedWrapper
from experiments.time_consistent_tracking import TimeConsistentCore,TimeConsistentFreshRuntime
from neural_navigation.masked_actions import ACTIONS


def main():
    parser=argparse.ArgumentParser()
    for name in ('bundle','training-trace','output'):parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():parser.error('Fresh output required')
    case=json.loads(args.training_trace.read_text())['case'];torch.set_num_threads(1)
    base=ScheduledRecoveryCurriculum(args.bundle,stage='robust',route_mode='shortcut',motion_profiles=True,
        footprint_clearance_cost=1.,recorded_cases=[case],core_factory=TimeConsistentCore)
    env=QualityMaskedWrapper(base,reward_scale=1/60);reports=[]
    try:
        # First reset exercises recorded reset; second exercises general reset.
        for episode in range(2):
            obs,info=env.reset(seed=92601 if episode==0 else None)
            actual=json.loads(base.training_case_json);options=actual['options'];e=base.active
            runtime=TimeConsistentFreshRuntime(args.bundle,base.scene/'decision_map.yaml',
                np.asarray(options['start'])+np.asarray(options.get('initial_prior_offset',[0,0,0])),options['goal'],'shortcut')
            assert isinstance(base.core,TimeConsistentCore)
            for frame in range(30):
                result=runtime.step(e.scan,e.observed_odometry,e.measured_velocity,e.elapsed,capture_observation=True)
                np.testing.assert_allclose(obs,result['policy_observation'],atol=2e-6,rtol=0)
                proposal=np.asarray(result['proposed_action']);choice=int(np.max(abs(ACTIONS-proposal),axis=1).argmin())
                assert env.action_masks()[choice]
                obs,_,done,truncated,transition=env.step(choice)
                np.testing.assert_allclose(result['command'],transition['commanded_velocity'],atol=2e-8,rtol=0)
                assert result['stop_reason']==transition['execution_guard_reason']
                assert not transition['nonzero_choice_changed_by_guard']
                if done or truncated:raise ValueError('Short parity trajectory ended early')
            reports.append(dict(recorded=info['focused_training_replay'],frames=30))
        assert [x['recorded'] for x in reports]==[True,False]
    finally:env.close()
    args.output.write_text(json.dumps(dict(verified=True,episodes=reports,
        limitation='Two native reset modes and sixty frames; execution consistency, not navigation acceptance.'),indent=2))
    print('Verified recorded/general reset and sixty-frame actor/command/feedback parity',flush=True)


if __name__=='__main__':main()
