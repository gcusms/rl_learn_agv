"""Replay the first real PPO episode that broke reference STOP retention.

Both arms start at its identical pre-update checkpoint and use the same
on-policy observations, original likelihoods, masks, returns, minibatches and
anchors. There is no new physics and neither arm replaces the live learner.
"""
import copy
import gc
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from experiments.guard_reason_release_actor import GuardReasonReleaseActor
from experiments.native625_onpolicy import Native625OnPolicy
from neural_navigation.experiment_budget import ExperimentBudget
from retained_stop_ppo import RetainedStopOnPolicy


ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / 'reports/stop_feedback_20261015'
HERE = Path(__file__).resolve().parent
MODELS = ROOT / 'models/stop_feedback_20261015/native625_suite_ppo_s231017'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    output = HERE / 'STOP_GUARD_REAL_UPDATE_REPLAY.json'
    assert not output.exists()
    torch.set_num_threads(1)
    budget = ExperimentBudget(.1, memory_mib=1200, directory=HERE)
    plan_path = BASE / 'native625_suite_ppo_20261016/PLAN.json'
    plan = json.loads(plan_path.read_text())
    rows_path = HERE / 'STAGE_ONE_ROWS.json'
    rows = json.loads(rows_path.read_text())
    parent = ROOT / 'models/stop_feedback_20261015/normal_suite_wait_pair_v2_s231016/wait_continuation'
    source_path = BASE / 'normal_suite_wait_protocol_v2_20261016/demonstrations.npz'
    assert digest(source_path) == plan['inputs'][str(source_path.relative_to(ROOT))]
    with np.load(source_path,allow_pickle=False) as archive:
        anchors, labels = archive['observations'].copy(),archive['choices'].copy()
    stops = anchors[labels == 0].copy()
    pools = [np.flatnonzero(labels == 0), np.flatnonzero(labels != 0)]
    torch.manual_seed(231017)
    actor = GuardReasonReleaseActor(torch.jit.load(str(parent / 'parent_actor.pt'),map_location='cpu')).eval()
    actor.load_state_dict(torch.load(parent/'checkpoint.pt',map_location='cpu',weights_only=False)['actor'])
    control = Native625OnPolicy(actor,learning_rate=.0003,temperature=4.)
    initial_critic = {k:v.clone() for k,v in control.critic.state_dict().items()}
    files = [Path(__file__),HERE/'retained_stop_ppo.py',plan_path,rows_path,source_path,
        parent/'checkpoint.pt',parent/'parent_actor.pt']

    def mismatches(model):
        count=0
        with torch.no_grad():
            for start in range(0,len(stops),128):
                count+=int((model(torch.from_numpy(stops[start:start+128])).abs().amax(1)>0.).sum())
                budget.tick()
        return count

    assert mismatches(actor)==0
    first_bad = None
    for row in rows:
        checkpoint=MODELS/f"episode_{row['episode']:02d}.pt"
        saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
        actor.load_state_dict(saved['actor'])
        files.append(checkpoint)
        wrong=mismatches(actor)
        if wrong:
            first_bad=(row,checkpoint,wrong)
            break
        del saved
    assert first_bad is not None,'No STOP mismatch in the fixed stage-one checkpoints'
    row,expected_checkpoint,expected_wrong=first_bad
    before_path=parent/'checkpoint.pt' if row['episode']==1 else MODELS/f"episode_{row['episode']-1:02d}.pt"
    before=torch.load(before_path,map_location='cpu',weights_only=False)
    actor.load_state_dict(before['actor'])
    control.critic.load_state_dict(initial_critic if row['episode']==1 else before['critic'])
    if row['episode']!=1:
        control.actor_optimizer.load_state_dict(before['actor_optimizer'])
        control.critic_optimizer.load_state_dict(before['critic_optimizer'])
        control.reference_head.load_state_dict(before['frozen_reference_head'])
    assert mismatches(actor)==0
    guarded_actor=GuardReasonReleaseActor(torch.jit.load(str(parent/'parent_actor.pt'),map_location='cpu')).eval()
    guarded_actor.load_state_dict(before['actor'])
    guarded=RetainedStopOnPolicy(guarded_actor,stops,learning_rate=.0003,temperature=4.)
    guarded.critic.load_state_dict(control.critic.state_dict())
    guarded.reference_head.load_state_dict(control.reference_head.state_dict())
    guarded.actor_optimizer.load_state_dict(copy.deepcopy(control.actor_optimizer.state_dict()))
    guarded.critic_optimizer.load_state_dict(copy.deepcopy(control.critic_optimizer.state_dict()))
    generator=torch.Generator().manual_seed(231018)
    if row['episode']!=1:
        generator.set_state(before['minibatch_rng'])
    trace=ROOT/row['rollout_path']
    assert digest(trace)==row['rollout_sha256']
    files += [trace,before_path]
    with np.load(trace,allow_pickle=False) as archive:
        rollout={k:archive[k].copy() for k in archive.files}
    statistics={'control':[],'guarded':[]}
    for _epoch in range(4):
        order=torch.randperm(len(rollout['actions']),generator=generator).numpy()
        for start in range(0,len(order),64):
            selected=order[start:start+64]
            anchor_indices=np.r_[pools[0][torch.randint(len(pools[0]),(64,),generator=generator).numpy()],
                pools[1][torch.randint(len(pools[1]),(64,),generator=generator).numpy()]]
            args=[rollout[k][selected] for k in ['observations','masks','actions',
                'old_log_probability','advantages','returns']]
            for name,learner in [('control',control),('guarded',guarded)]:
                statistics[name].append(learner.update(*args,sampled_temperature=4.,anchors=anchors[anchor_indices]))
                budget.tick()
    expected=torch.load(expected_checkpoint,map_location='cpu',weights_only=False)
    maximum_parameter_error=0.
    for key,value in actor.state_dict().items():
        difference=float((value.to(torch.float64)-expected['actor'][key].to(torch.float64)).abs().max()) if value.numel() else 0.
        maximum_parameter_error=max(maximum_parameter_error,difference)
        torch.testing.assert_close(value,expected['actor'][key],rtol=0,atol=2e-7)
    assert torch.equal(generator.get_state(),expected['minibatch_rng'])
    control_wrong,guarded_wrong=mismatches(actor),guarded.stop_mismatches()
    assert control_wrong==expected_wrong and guarded_wrong==0
    gc.collect();budget.tick()
    result=dict(completed=True,source_episode=row['episode'],sampled_decisions=len(rollout['actions']),
        identical_pre_update_actor_critic_optimizers_and_minibatches=True,
        control_matches_original_checkpoint=True,maximum_control_parameter_error=maximum_parameter_error,
        minibatch_rng_end_matches_original_checkpoint=True,reference_stop_labels=len(stops),
        control_stop_mismatches=control_wrong,guarded_stop_mismatches=guarded_wrong,
        guarded_accepted_actor_updates=sum(bool(s['actor_updated']) for s in statistics['guarded']),
        guarded_rejected_actor_updates=sum(bool(s['actor_update_attempted'] and not s['actor_updated']) for s in statistics['guarded']),
        guarded_step_fractions=[s['accepted_actor_step_fraction'] for s in statistics['guarded']],
        update_statistics=statistics,
        offline_replay_optimizer_calls=sum(len(v) for v in statistics.values()),
        no_live_optimizer_or_physics_modified=True,no_new_physics=True,new_ppo_steps=0,
        no_new_candidate_saved=True,physical_recovery_improvement_unverified=True,
        inputs={str(p.relative_to(ROOT)):digest(p) for p in set(files)},resource=budget.statistics(),
        goal_achieved=False,deployment_approved=False)
    output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['inputs','update_statistics']},ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
