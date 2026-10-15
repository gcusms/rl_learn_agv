"""Measure actual recovery learning without mistaking it for navigation success.

Feature substitutions hold physical inputs and masks fixed. They are offline
sensitivity diagnostics, potentially outside the training distribution, and
never issue commands or establish causal task performance.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from experiments.stop_recovery_policy import StopRecoveryPPO
from experiments.train_prefix_recovery import frozen_base_digest
from neural_navigation.decision_feedback import GROUP_DIM, REASON_SLICE, RISK_SLICE, STAT_SLICE
from neural_navigation.experiment_budget import ExperimentBudget


def probabilities(model, observations, masks, budget):
    batches = []
    with torch.no_grad():
        for start in range(0, len(observations), 250):
            x = torch.from_numpy(observations[start:start+250])
            batches.append(model.policy.get_distribution(x, masks[start:start+250]).distribution.probs.numpy())
            budget.tick()
    return np.concatenate(batches)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cpu-cores', type=float, default=.05)
    parser.add_argument('--through-steps', type=int,
        help='Inspect a published pre-update policy_step_N checkpoint; not a final training result')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    budget = ExperimentBudget(args.cpu_cores, directory=args.output)
    status = json.loads((args.model_run/'status.json').read_text())
    interim = args.through_steps is not None
    if interim and (args.through_steps < 1000 or args.through_steps % 1000):
        raise ValueError('Published pre-update checkpoints exist at positive 1000-step boundaries')
    if not interim and status['status'] != 'complete':
        raise ValueError('A completed training run is required')
    expected_steps = args.through_steps if interim else status['additional_ppo_steps']
    checkpoint = args.model_run/(f'policy_step_{expected_steps}.zip' if interim else 'policy.zip')
    last_metadata = None
    observations, masks, labels = [], [], []
    for path in sorted((args.model_run/'rollouts').glob('*.npz')):
        with np.load(path, allow_pickle=False) as data:
            metadata = json.loads(str(data['metadata_json']))
            if metadata['collected_steps'] > expected_steps:
                continue
            last_metadata = metadata
            observations.append(data['observations'][:, 0].copy())
            masks.append(data['action_masks'][:, 0].astype(bool))
            labels.extend(json.loads(str(s)) for s in data['execution_diagnostics_json'][:, 0])
    observations, masks = np.concatenate(observations), np.concatenate(masks)
    if len(observations) != expected_steps:
        raise ValueError('All actual PPO samples must be present')
    bc = StopRecoveryPPO.load(args.model_run/'policy_bc.zip', device='cpu')
    model = StopRecoveryPPO.load(checkpoint, device='cpu')
    if interim and (last_metadata['collected_steps'] != expected_steps
            or model.num_timesteps != expected_steps
            or model._n_updates != last_metadata['optimizer_epochs_before_update']):
        raise ValueError('The checkpoint must be the actual published pre-update policy')
    budget.tick()
    if frozen_base_digest(bc.policy) != frozen_base_digest(model.policy):
        raise ValueError('Base actor changed during PPO')
    with torch.no_grad():
        supported = model.policy.recovery_support(torch.from_numpy(observations)).numpy()
    if not np.array_equal(supported, [row['recovery_actor_supported'] for row in labels]):
        raise ValueError('Recorded training support differs from the saved policy contract')
    before = probabilities(bc, observations, masks, budget)
    after = probabilities(model, observations, masks, budget)
    before_choice, after_choice = before.argmax(-1), after.argmax(-1)
    changed = supported & (before_choice != after_choice)
    wait_to_motion = changed & (before_choice == 0) & (after_choice != 0)
    motion_to_wait = changed & (before_choice != 0) & (after_choice == 0)
    different_motion = changed & (before_choice != 0) & (after_choice != 0)
    long_wait = supported & (observations[:, 300+STAT_SLICE.start+3]*5. >= 1.-1e-6)
    substitutions = {}
    for name in ['previous_reason_none', 'without_contact_scores', 'without_wait_timers', 'without_explanations']:
        x = observations.copy()
        if name in ['previous_reason_none', 'without_explanations']:
            x[:, 300+REASON_SLICE.start:300+REASON_SLICE.stop] = 0.
            x[:, 300+REASON_SLICE.start] = 1.
        if name in ['without_contact_scores', 'without_explanations']:
            x[:, 300+RISK_SLICE.start:300+RISK_SLICE.stop] = 0.
        if name == 'without_wait_timers':
            x[:, 300+STAT_SLICE.start+2:300+STAT_SLICE.start+4] = 0.
        if name == 'without_explanations':
            x[:, 300:300+GROUP_DIM] = 0.
        with torch.no_grad():
            substituted_support = model.policy.recovery_support(torch.from_numpy(x)).numpy()
        # Proactive support itself may use contact context. Removing that
        # context can change the gate; report this instead of assuming the
        # intervention preserves support as the older stop-only policy did.
        common_support = supported & substituted_support
        prediction = probabilities(model, x, masks, budget)
        substitutions[name] = dict(
            learnable_support_changes=int(np.count_nonzero(supported != substituted_support)),
            common_supported_states=int(common_support.sum()),
            changed_argmax_in_common_supported_states=int(np.count_nonzero(
                (after.argmax(-1) != prediction.argmax(-1)) & common_support)),
            changed_argmax_in_supported_states=int(np.count_nonzero(
                (after.argmax(-1) != prediction.argmax(-1)) & supported)),
            max_probability_change_in_supported_states=float(abs(after[supported]-prediction[supported]).max()),
            actual_motion_samples_changed=int(sum(
                bool(supported[i] and row['sampled_choice'] and after[i].argmax() != prediction[i].argmax())
                for i, row in enumerate(labels))))
    old_parameters = dict(bc.policy.named_parameters())
    delta = {name: float(torch.linalg.vector_norm((p.detach()-old_parameters[name].detach()).flatten()))
        for name, p in model.policy.named_parameters() if '.residual.' in name}
    sources = Counter()
    frozen_sampling = {}
    decision_changes = {}
    for i, row in enumerate(labels):
        case = json.loads(row['case_json'])
        name = case.get('source_training_case_id') or case.get('scene', 'routine')
        decisions = decision_changes.setdefault(name, dict(supported_frames=0,
            bc_wait_to_ppo_motion=0, bc_motion_to_ppo_wait=0, different_motion_choice=0,
            long_prior_wait_with_choices_frames=0, long_wait_bc_wait_to_ppo_motion=0))
        for key, condition in [('supported_frames', supported),
                ('bc_wait_to_ppo_motion', wait_to_motion), ('bc_motion_to_ppo_wait', motion_to_wait),
                ('different_motion_choice', different_motion),
                ('long_prior_wait_with_choices_frames', long_wait),
                ('long_wait_bc_wait_to_ppo_motion', long_wait & wait_to_motion)]:
            decisions[key] += int(condition[i])
        counts = frozen_sampling.setdefault(name, dict(frames=0, frozen_frames=0,
            sampled_choice_differs_from_frozen_greedy=0, recovery_frames=0))
        counts['frames'] += 1
        counts['recovery_frames'] += int(supported[i])
        counts['frozen_frames'] += int(not supported[i])
        counts['sampled_choice_differs_from_frozen_greedy'] += int(
            not supported[i] and row['sampled_choice'] != before[i].argmax())
        if not supported[i]:
            continue
        sources[name] += 1
    result = dict(completed=True, actual_sampled_frames=len(labels), supported_frames=int(supported.sum()),
        training_run_completed=status['status']=='complete', analysis_covers_final_training=not interim,
        checkpoint_is_pre_update=interim,
        source_run_status_at_read=status, policy_checkpoint=str(checkpoint),
        checkpoint_collected_timesteps=model.num_timesteps,
        optimizer_epochs_completed=model._n_updates,
        last_rollout_included_in_checkpoint_updates=not interim,
        checkpoint_updates_through_steps=expected_steps-250 if interim else expected_steps,
        bc_to_ppo=dict(frozen_base_unchanged=True, residual_parameter_delta_l2=delta,
            supported_argmax_changes=int(np.count_nonzero(
                (before.argmax(-1) != after.argmax(-1)) & supported)),
            maximum_supported_probability_change=float(abs(before[supported]-after[supported]).max())),
        offline_choice_transitions=dict(bc_wait_to_ppo_motion=int(wait_to_motion.sum()),
            bc_motion_to_ppo_wait=int(motion_to_wait.sum()),
            different_motion_choice=int(different_motion.sum()),
            long_wait_threshold_s=1., long_prior_wait_with_choices_frames=int(long_wait.sum()),
            long_wait_bc_wait_to_ppo_motion=int((long_wait & wait_to_motion).sum()),
            physical_improvement_proven=False),
        choice_transitions_by_source=decision_changes,
        supported_training_sources=dict(sources), offline_sensitivity=substitutions,
        frozen_base_sampling_by_case=frozen_sampling,
        frozen_base_sampling_summary=dict(frames=int((~supported).sum()),
            sampled_choice_differs_from_frozen_greedy=sum(
                r['sampled_choice_differs_from_frozen_greedy'] for r in frozen_sampling.values()),
            interpretation='Actual stochastic samples outside trainable support; different from deterministic deployment, not evidence that every deviation is harmful.'),
        policy_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        resource=budget.statistics(), deployment_approved=False,
        limitations=['Reused actual training states, not new closed-loop tasks.',
            'A pre-update checkpoint has observed its last rollout but has not optimized on it; an interim analysis is not a complete training or navigation result.',
            'Substituted explanations can be outside the training distribution.',
            'Proactive contact ablations can also change the learnable-support gate; common-support and gate changes are reported separately.',
            'Sensitivity does not prove that feedback caused safe recovery or that PPO improved navigation.'])
    (args.output/'RESULTS.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
