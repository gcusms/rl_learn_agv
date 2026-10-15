"""Validate new support on actual guard-checked simulated sensor observations."""
import json
from pathlib import Path
import numpy as np
import torch
from navrl.geometry import ROOT
from neural_navigation.stress_env import StressNavigationEnv
from neural_navigation.masked_actions import ACTIONS, masks_from_observation
from experiments.time_consistent_tracking import TimeConsistentFreshRuntime
from experiments.probe_fine_yaw_recovery import FineYawDiagnostic
from experiments.focused_recovery_policy import recovery_support
from experiments.soft_prior_recovery_policy import warm_start_soft_prior
from experiments.masked_navigation import PortableMaskedActor
from prototypes.continued_recovery_policy import warm_start_continued, continued_support
from prototypes.check_continued_migration import InspectionInterface
from sb3_contrib import MaskablePPO


def main():
    torch.set_num_threads(1)
    project=Path(__file__).resolve().parents[1]
    base=project/'training/runs/safety_context_20261011'
    folder=base/'confirmed_free_initial_five';bundle=folder/'portable'
    trace=json.loads((folder/'evaluation/independent_dynamic_023.json').read_text())
    duration=json.loads((base/'RECOVERY_TURN_DURATION_DIAGNOSTIC.json').read_text())
    expected=next(r for r in duration['results'] if r['manual_turn_frames_requested']==12)['rows']
    case=trace['case'];scene=ROOT/case['scene']
    env=StressNavigationEnv(case=case,components=bundle,scene_dir=scene,route_mode='shortcut',max_steps=700)
    observations=[]
    try:
        env.reset(seed=case['seed'],options={k:case[k] for k in ('start','goal','initial_prior_offset') if k in case})
        runtime=TimeConsistentFreshRuntime(bundle,scene/'decision_map.yaml',
            np.asarray(case['start'])+np.asarray(case.get('initial_prior_offset',[0,0,0])),case['goal'],'shortcut')
        for index in range(48):
            if index==35:
                runtime.policy=FineYawDiagnostic(runtime.policy,runtime,.5,12)
            result=runtime.step(env.scan,env.observed_odometry,env.measured_velocity,env.elapsed,capture_observation=True)
            if index<35:
                np.testing.assert_allclose(result['policy_observation'],trace['samples'][index]['policy_observation'],atol=2e-6,rtol=0)
                np.testing.assert_allclose(result['command'],trace['samples'][index]['command'],atol=2e-8,rtol=0)
                assert result['stop_reason']==trace['samples'][index]['stop_reason']
            else:
                np.testing.assert_allclose(result['command'],expected[index-35]['command'],atol=2e-8,rtol=0)
                assert result['stop_reason']==expected[index-35]['stop_reason']
            observations.append(result['policy_observation'])
            v,w=result['command'];env.step(np.asarray([v/(.2 if v>=0 else .08),w/.5]))
        assert runtime.policy.accepted==12 and runtime.policy.rejected==0
        assert not env.collision_events and not env.hard_guard_activations
    finally:
        env.close()
    obs=torch.as_tensor(np.asarray(observations,np.float32))
    masks=masks_from_observation(obs.numpy()[:,:282])
    extra=continued_support(obs)&~recovery_support(obs)
    assert bool(extra.any()),'Actual expanded-support states required'
    checkpoint=base/'quality_recovery_4000/candidate/policy_2000.zip'
    legacy=MaskablePPO.load(checkpoint,device='cpu')
    scale=float(getattr(legacy,'training_reward_scale',1.))
    reference=warm_start_soft_prior(checkpoint,InspectionInterface(scale),92611,1/6)
    candidate=warm_start_continued(checkpoint,InspectionInterface(scale),92611,1/6,.7)
    with torch.no_grad():
        before=reference.policy.get_distribution(obs,action_masks=masks)
        after=candidate.policy.get_distribution(obs,action_masks=masks)
        torch.testing.assert_close(before.mode(),after.mode(),atol=0,rtol=0)
        torch.testing.assert_close(before.distribution.probs[~extra],after.distribution.probs[~extra],atol=0,rtol=0)
        assert bool(((before.distribution.probs-after.distribution.probs).abs().sum(-1)[extra]>0).all())
        actor=torch.jit.trace(PortableMaskedActor(candidate.policy),obs)
        np.testing.assert_array_equal(actor(obs).numpy(),ACTIONS[after.mode().numpy()])
    rows=[]
    for index in torch.nonzero(extra).flatten().tolist():
        rows.append(dict(frame=index,heading_cos=float(obs[index,232]),
            last_executed_normalized=obs[index,247:249].tolist(),
            native_allowed_choices=np.flatnonzero(masks[index]).tolist()))
    report=dict(actual_sensor_frames=48,prefix_and_guard_checked_manual_branch_reproduced=True,
        extra_support_frames=int(extra.sum()),extra_support_observations=rows,
        initialization_greedy_commands_exact=True,sampling_intentionally_changed_only_in_extra_support=True,
        portable_command_parity=True,manual_commands_accepted=12,manual_guard_rejections=0,
        prototype_commands_executed=False,prototype_trained=False,
        limitation='Manual simulator branch supplies inspection inputs only. No labels or new prototype commands execute; parity and broadened probabilities do not prove learned sustained recovery.')
    (base/'CONTINUED_ACTUAL_SENSOR_SUPPORT_CHECK.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
