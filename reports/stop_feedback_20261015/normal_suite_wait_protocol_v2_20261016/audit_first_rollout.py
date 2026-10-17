"""Audit the first actual native625 rollout and its updates, without training."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from experiments.guard_reason_release_actor import GuardReasonReleaseActor
from experiments.native625_onpolicy import Native625OnPolicy
from navrl.map import StaticMap
from neural_navigation.experiment_budget import ExperimentBudget
from neural_navigation.masked_actions import COMMANDS
from neural_navigation.training_distance import TrainingDistanceGraph


ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / 'reports/stop_feedback_20261015'
HERE = Path(__file__).resolve().parent


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    output = HERE / 'FIRST_NATIVE_PPO_ROLLOUT_AUDIT.json'
    assert not output.exists(), 'The first-rollout audit is immutable'
    report_path = BASE / 'native625_suite_ppo_20261016/RESULTS.json'
    report = json.loads(report_path.read_text())
    row = report['episodes'][0]
    assert row['episode'] == 1 and row['release_frames'] == 450
    path = ROOT / row['rollout_path']
    assert digest(path) == row['rollout_sha256']
    with np.load(path, allow_pickle=False) as saved:
        rollout = {key:saved[key].copy() for key in saved.files}
    parent_path = BASE / 'normal_suite_wait_validation_v2_20261016/RESULTS.json'
    parent = json.loads(parent_path.read_text())
    assert parent['completed'] and parent['normal18_pair']['all_regression_checks_passed']
    model = ROOT / parent['selected_model']
    plan_path = BASE / 'native625_suite_ppo_20261016/PLAN.json'
    plan = json.loads(plan_path.read_text())
    assert digest(model / 'checkpoint.pt') == plan['inputs'][str((model/'checkpoint.pt').relative_to(ROOT))]
    torch.set_num_threads(1)
    budget = ExperimentBudget(.1, memory_mib=1100, directory=HERE)
    actor = GuardReasonReleaseActor(torch.jit.load(str(model / 'parent_actor.pt'), map_location='cpu')).eval()
    actor.load_state_dict(torch.load(model / 'checkpoint.pt', map_location='cpu', weights_only=False)['actor'])
    learner = Native625OnPolicy(actor, temperature=4.)
    assert float(rollout['temperature']) == 4.
    x, masks, actions = [rollout[k] for k in ['observations','masks','actions']]
    assert len(x) == row['sampled_decisions'] <= 250
    assert masks[np.arange(len(x)), actions].all()
    with torch.no_grad():
        terms = learner.evaluate(x, masks, actions)
    likelihood_difference = abs(terms['log_probability'].numpy()-rollout['old_log_probability'])
    np.testing.assert_allclose(terms['log_probability'].numpy(), rollout['old_log_probability'], rtol=0, atol=2e-5)
    np.testing.assert_array_equal(rollout['old_values'], np.zeros(len(x)))
    assert int(terms['eligible'].sum()) == row['eligible_actor_decisions']
    expected_changes = (abs(rollout['executed_commands']-COMMANDS[actions]).max(1)>2e-8)
    np.testing.assert_array_equal(expected_changes, rollout['execution_changed'])
    for index in np.flatnonzero(expected_changes):
        assert index==len(x)-1 and rollout['stop_reasons'][index]=='estimated_goal_reached'
    expected_ends = np.zeros(len(x),bool)
    expected_ends[-1] = True
    np.testing.assert_array_equal(rollout['terminated'], expected_ends)
    np.testing.assert_array_equal(rollout['episode_ends'], expected_ends)
    assert abs(rollout['labels'][0,0]-90.) < 2e-8
    if len(x)>1:
        np.testing.assert_allclose(np.diff(rollout['labels'][:,0]), .2, rtol=0, atol=2e-8)
    # Independently reconstruct reward and return semantics below.
    tasks_path = BASE / 'context_feedback_longfit_protocol_20261016/tasks.json'
    case = json.loads(tasks_path.read_text())['actual106']['case']
    scene = ROOT / 'training' / case['scene']
    field = TrainingDistanceGraph(StaticMap(scene/'decision_map.yaml',radius=.3)).field(case['goal'])
    source = ROOT / row['source_trace']
    with np.load(source,allow_pickle=False) as saved:
        previous_command = saved['commands'][449].copy()
    expected_rewards, components = [], []
    for index, label in enumerate(rollout['labels']):
        reward = dict(geodesic_progress=0.,time=0.,travel=0.,action_change=0.,terminal=0.)
        if rollout['stop_reasons'][index]=='estimated_goal_reached':
            assert index == len(x)-1
            reward['terminal'] = 10. if row['physical_arrival'] else -20.
        else:
            reward.update(geodesic_progress=5.*(field.value(label[1:3])-field.value(label[4:6])),
                time=-.01*.2, travel=-.02*float(np.linalg.norm(label[4:6]-label[1:3])),
                action_change=-.001*float(abs(rollout['executed_commands'][index]-previous_command).sum()))
            if index==len(x)-1:
                if row['collisions'] or row['physical_guards'] or label[7]<.04-1e-4:
                    reward['terminal'] = -20.
                else:
                    assert len(x)==250, 'Unexpected nonterminal rollout cut'
                    reward['terminal'] = -5.
        expected_rewards.append(sum(reward.values()))
        components.append(reward)
        previous_command = rollout['executed_commands'][index]
    np.testing.assert_allclose(rollout['rewards'],expected_rewards,rtol=0,atol=1e-6)
    for key, total in row['reward_terms'].items():
        assert abs(total-sum(component[key] for component in components))<1e-5,key
    assert abs(row['training_reward']-float(rollout['rewards'].sum()))<1e-6
    values, following = rollout['old_values'], rollout['next_values']
    expected_advantages = np.zeros(len(x),dtype=np.float64)
    carry = 0.
    for index in range(len(x)-1,-1,-1):
        delta = rollout['rewards'][index]+.99*following[index]*(not expected_ends[index])-values[index]
        carry = delta+.99*.95*(not expected_ends[index])*carry
        expected_advantages[index] = carry
    np.testing.assert_allclose(rollout['advantages'],expected_advantages,rtol=0,atol=1e-4)
    np.testing.assert_allclose(rollout['returns'],expected_advantages+values,rtol=0,atol=1e-4)
    assert len(row['updates'])==4*int(np.ceil(len(x)/64))
    actor_updates = sum(bool(update['actor_updated']) for update in row['updates'])
    assert actor_updates>0, 'The first actual rollout did not update the actor'
    assert all(np.isfinite(update[key]) for update in row['updates']
        for key in ['approx_kl','clip_fraction','entropy','value_loss','actor_loss','anchor_kl'])
    snapshot = HERE / 'FIRST_NATIVE_PPO_ROW.json'
    assert not snapshot.exists()
    snapshot.write_text(json.dumps(row,indent=2)+'\n')
    files = [Path(__file__),snapshot,path,source,parent_path,plan_path,tasks_path,
        model/'checkpoint.pt',model/'parent_actor.pt',
        ROOT/'training/experiments/native625_onpolicy.py',
        ROOT/'training/experiments/train_native625_recovery.py',
        ROOT/'training/neural_navigation/training_distance.py']
    files += [p for p in scene.iterdir() if p.is_file()]
    budget.tick()
    result = dict(completed=True,audited_training_decisions=len(x),
        initialization450_steps_excluded=True,sampling_temperature=4.,
        sampled_actions_masks_old_likelihoods_and_execution_checked=True,
        eligible_actor_decisions=int(terms['eligible'].sum()),
        changed_execution_indices=np.flatnonzero(expected_changes).tolist(),
        original_action_likelihood_retained_even_at_terminal=True,
        maximum_old_log_probability_error=float(likelihood_difference.max()),
        maximum_reward_reconstruction_error=float(abs(rollout['rewards']-expected_rewards).max()),
        maximum_advantage_reconstruction_error=float(abs(rollout['advantages']-expected_advantages).max()),
        actual_actor_optimizer_updates=actor_updates,
        actual_critic_optimizer_updates=len(row['updates']),
        training_episode_safe_success=row['safe_success'],
        training_episode_not_independent_evaluation=True,
        inputs={str(p.relative_to(ROOT)):digest(p) for p in set(files)},
        resource=budget.statistics(),no_new_physics=True,no_optimizer_updates=True,
        new_ppo_steps=0,goal_achieved=False,deployment_approved=False)
    temporary = output.with_suffix('.tmp')
    temporary.write_text(json.dumps(result,indent=2)+'\n')
    temporary.replace(output)
    print(json.dumps({k:v for k,v in result.items() if k!='inputs'}),flush=True)


if __name__=='__main__':
    main()
