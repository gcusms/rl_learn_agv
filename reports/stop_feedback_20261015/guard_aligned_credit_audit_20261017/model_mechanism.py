"""Serial post-validation neural audit of the actual completed PPO data.

The waiting process imports only the standard library. It waits for the exact
physical validation process to finish before loading any neural model. Feature
ablations diagnose sensitivity only; they are never commands or reward labels.
"""
from datetime import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import time
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
BASE = HERE.parent
UNIT = 'rl-guard-aligned-ppo-resume-validate-20261017.service'
INVOCATION = 'db6a98a2b8cf45c3a331892b430ff4f0'


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')
    temporary.replace(path)


def main():
    output = HERE / 'MODEL_MECHANISM.json'
    assert not output.exists()
    training_path = BASE/'guard_aligned_mixed_ppo_resume_20261017/RESULTS.json'
    report = read(training_path)
    assert report['completed'] and report['new_ppo_steps'] == 1993
    parent = ROOT/'models/stop_feedback_20261015/guard_aligned_migration_s231017'
    batch01 = ROOT/'models/stop_feedback_20261015/guard_aligned_mixed_ppo_s231017/batch_01.pt'
    model = ROOT/report['model_path']
    paths = [Path(__file__), training_path, HERE/'RESULTS.json', parent/'checkpoint.pt',
        parent/'parent_actor.pt', parent/'portable/actor.pt', batch01,
        model/'checkpoint.pt', model/'parent_actor.pt', model/'portable/actor.pt',
        model/'portable/manifest.json']
    paths += [ROOT/r['rollout_path'] for r in report['episodes']]
    for folder in ['experiments', 'neural_navigation', 'navrl']:
        paths += list((ROOT/'training'/folder).rglob('*.py'))
    inputs = {str(p.relative_to(ROOT)):digest(p) for p in set(paths)}
    write(HERE/'MODEL_MECHANISM_PLAN.json', dict(inputs=inputs, exact_dependency=UNIT,
        expected_invocation=INVOCATION, new_physical_episodes=0, new_ppo_steps=0,
        soft_cpu_cores=.35, memory_hard_max_mib=1536, neural_imports_only_after_dependency_terminal=True,
        scope='log-probability/value replay, export parity and diagnostic feature sensitivity; not navigation'))
    while True:
        checked = subprocess.run(['systemctl','--user','show',UNIT,
            '--property=LoadState,ActiveState,SubState,MainPID,InvocationID,ExecMainStatus'],
            capture_output=True,text=True,check=True)
        state = dict(line.split('=',1) for line in checked.stdout.splitlines() if '=' in line)
        assert state['InvocationID'] in ('', INVOCATION), 'Validation invocation changed'
        if state['MainPID'] == '0':
            validation_path = BASE/'guard_aligned_mixed_ppo_resume_validation_20261017/RESULTS.json'
            validation = read(validation_path) if validation_path.exists() else {}
            terminal = dict(observed_unit_state=state, validation_completed=validation.get('completed'),
                validation_reason=validation.get('reason'),
                validation_physical_episodes=validation.get('new_physical_episodes'),
                dependency_failure_file_exists=(validation_path.parent/'FAILURE.json').exists())
            break
        write(HERE/'MODEL_MECHANISM_STATUS.json', dict(status='waiting_for_exact_physical_validation',
            checked_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),dependency=state,
            neural_libraries_not_loaded=True))
        time.sleep(5.)
    for name, expected in inputs.items():
        assert digest(ROOT/name) == expected, name
    import gc
    import numpy as np
    import torch
    from experiments.guard_reason_release_actor import GuardReasonReleaseActor
    from experiments.native625_onpolicy import Native625OnPolicy
    from neural_navigation.experiment_budget import ExperimentBudget
    from neural_navigation.masked_actions import COMMANDS

    torch.set_num_threads(1)
    budget = ExperimentBudget(.35, memory_mib=1200, directory=HERE)
    write(HERE/'MODEL_MECHANISM_STATUS.json', dict(status='actual_checkpoint_replay', dependency_terminal=terminal))
    episode_data = []
    for row in report['episodes']:
        path = ROOT/row['rollout_path']
        assert digest(path) == row['rollout_sha256']
        with np.load(path,allow_pickle=False) as z:
            episode_data.append((row,{k:z[k].copy() for k in (
                'observations','masks','actions','old_log_probability','old_values')}))
    replay = []
    parent_reason_weight = None
    with torch.no_grad():
        for number, checkpoint_path in [(1,parent/'checkpoint.pt'),(2,batch01)]:
            actor = GuardReasonReleaseActor(torch.jit.load(str(parent/'parent_actor.pt'),map_location='cpu')).eval()
            checkpoint = torch.load(checkpoint_path,map_location='cpu',weights_only=False)
            actor.load_state_dict(checkpoint['actor'])
            learner = Native625OnPolicy(actor,temperature=4.)
            if number == 2:
                learner.critic.load_state_dict(checkpoint['critic'])
            else:
                parent_reason_weight = actor.head[0].current_reason.weight.detach().clone()
            largest_log_error = largest_value_error = 0.
            decisions = eligible = 0
            for row,z in episode_data:
                if row['batch'] != number:
                    continue
                for start in range(0,len(z['actions']),128):
                    end = start+128
                    terms = learner.evaluate(z['observations'][start:end],z['masks'][start:end],z['actions'][start:end])
                    logp,values = terms['log_probability'].numpy(),terms['values'].numpy()
                    expected_logp,expected_value = z['old_log_probability'][start:end],z['old_values'][start:end]
                    np.testing.assert_allclose(logp,expected_logp,rtol=0,atol=2e-5)
                    np.testing.assert_allclose(values,expected_value,rtol=0,atol=2e-5)
                    largest_log_error = max(largest_log_error,float(np.max(abs(logp-expected_logp))))
                    largest_value_error = max(largest_value_error,float(np.max(abs(values-expected_value))))
                    decisions += len(logp)
                    eligible += int(terms['eligible'].sum())
                    budget.tick()
            replay.append(dict(batch=number,decisions=decisions,actor_eligible=eligible,
                maximum_log_probability_error=largest_log_error,maximum_old_value_error=largest_value_error,
                behavior_checkpoint=str(checkpoint_path.relative_to(ROOT)),passed=True,
                first_batch_critic_zero_output_constructor_used=number==1))
            del actor,learner,checkpoint
            gc.collect()
        actor = GuardReasonReleaseActor(torch.jit.load(str(model/'parent_actor.pt'),map_location='cpu')).eval()
        checkpoint = torch.load(model/'checkpoint.pt',map_location='cpu',weights_only=False)
        actor.load_state_dict(checkpoint['actor'])
        exported = torch.jit.load(str(model/'portable/actor.pt'),map_location='cpu').eval()
        all_x = np.concatenate([z['observations'] for _,z in episode_data])
        sensitivity = {name:dict(checked_supported_states=0,changed_greedy_actions=0,
            maximum_legal_relative_logit_change=0.) for name in ('current_reason_block','risk_history_block')}
        action_counts = {}
        parity_rows = 0
        for start in range(0,len(all_x),128):
            x = torch.from_numpy(all_x[start:start+128])
            full = actor(x)
            np.testing.assert_array_equal(full.numpy(),exported(x).numpy())
            features,_ = actor.features(x)
            legal,supported = actor.masks(x),actor.support(x)
            logits = actor.head(features)+2.*features[:,76:88]
            choices = logits.masked_fill(~legal,-1e8).argmax(1)
            for value in full.numpy():
                key = str(int(np.max(abs(COMMANDS-value[None]),axis=1).argmin()))
                action_counts[key] = action_counts.get(key,0)+1
            for name,block in [('current_reason_block',slice(312,408)),('risk_history_block',slice(0,42))]:
                changed = features.clone()
                changed[:,block] = 0.
                altered = actor.head(changed)+2.*changed[:,76:88]
                alternate_choice = altered.masked_fill(~legal,-1e8).argmax(1)
                item = sensitivity[name]
                item['checked_supported_states'] += int(supported.sum())
                item['changed_greedy_actions'] += int(((choices!=alternate_choice)&supported).sum())
                # Relative to STOP removes a uniform, action-irrelevant logit shift.
                delta = (logits-logits[:,:1])-(altered-altered[:,:1])
                active = legal & supported[:,None]
                if active.any():
                    item['maximum_legal_relative_logit_change'] = max(
                        item['maximum_legal_relative_logit_change'],float(delta[active].abs().max()))
            parity_rows += len(x)
            budget.tick()
        reason_weight = actor.head[0].current_reason.weight
        result = dict(completed=True,at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
            dependency_terminal=terminal,independent_behavior_replay=replay,
            total_replayed_decisions=sum(r['decisions'] for r in replay),
            final_checkpoint_export_parity_rows=parity_rows,final_greedy_action_counts=action_counts,
            current_reason_projection_nonzero_weights=int(torch.count_nonzero(reason_weight)),
            current_reason_projection_norm=float(reason_weight.norm()),
            current_reason_weight_delta_from_v2_migration=float((reason_weight-parent_reason_weight).norm()),
            isolated_feature_block_ablations=sensitivity,
            ablated_features_are_diagnostic_interventions_not_valid_navigation_observations=True,
            sensitivity_does_not_prove_correct_reason_interpretation_or_navigation_success=True,
            no_runtime_policy_or_input_modification=True,new_optimizer_updates=0,new_ppo_steps=0,
            new_physical_episodes=0,inputs=inputs,resource=budget.statistics(),
            goal_achieved=False,deployment_approved=False)
    for name,expected in inputs.items():
        assert digest(ROOT/name)==expected,name
    write(output,result)
    write(HERE/'MODEL_MECHANISM_STATUS.json',dict(status='complete',goal_achieved=False,
        independent_replay_decisions=result['total_replayed_decisions'],no_new_physics=True))
    print(json.dumps({k:result[k] for k in ('total_replayed_decisions','final_checkpoint_export_parity_rows',
        'current_reason_projection_nonzero_weights','isolated_feature_block_ablations')},ensure_ascii=False),flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        write(HERE/'MODEL_MECHANISM_FAILURE.json',dict(error=repr(error),no_auto_restart=True))
        raise
