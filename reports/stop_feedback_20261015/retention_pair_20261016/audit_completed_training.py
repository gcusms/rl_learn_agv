"""Read-only retention audit of both completed, independently trained 2k runs.

Uses recorded legal demonstrations only; never creates a physical environment,
samples commands, changes an optimizer, or promotes a model. Closed-loop
navigation remains the separately queued, predeclared paired evaluation.
"""
from datetime import datetime
import hashlib
import json
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import torch

from experiments.stop_recovery_policy import StopRecoveryPPO
from experiments.train_prefix_recovery import frozen_base_digest
from neural_navigation.experiment_budget import ExperimentBudget
from neural_navigation.masked_actions import masks_from_observation


ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    destination = OUT / 'COMPLETED_TRAINING_AUDIT.json'
    if destination.exists():
        raise ValueError('A completed audit must not be overwritten')
    plan = json.loads((OUT / 'PLAN.json').read_text())
    torch.set_num_threads(1)
    budget = ExperimentBudget(.05, memory_mib=2048, directory=OUT)
    reference = ROOT / Path(plan['reference_bc']).parent
    with np.load(reference / 'demonstrations.npz', allow_pickle=False) as data:
        obs = data['observations'].copy()
        masks = data['masks'].astype(bool)
        labels = data['choices'].copy()
    if len(labels) != 2089 or (labels == 0).sum() != 37:
        raise ValueError('The original 2089-label reference is required')
    np.testing.assert_array_equal(masks_from_observation(obs[:, :282]), masks)
    if not masks[np.arange(len(labels)), labels].all():
        raise ValueError('A demonstration target is not allowed by its recorded mask')
    variants = dict(reference=reference)
    variants.update({key: ROOT / value for key, value in plan['models'].items()})
    outcomes = {}
    base_digests = []
    inputs = {'PLAN.json': digest(OUT / 'PLAN.json'),
              str((reference / 'demonstrations.npz').relative_to(ROOT)):
                  digest(reference / 'demonstrations.npz')}
    for name, folder in variants.items():
        is_reference = name == 'reference'
        checkpoint = folder / ('policy_bc.zip' if is_reference else 'policy.zip')
        status = json.loads((folder / 'status.json').read_text())
        if not is_reference and (status['status'] != 'complete'
                                  or status['additional_ppo_steps'] != 2000):
            raise ValueError('Both independent physical PPO runs must be complete')
        model = StopRecoveryPPO.load(checkpoint, device='cpu')
        if model.num_timesteps != (0 if is_reference else 2000):
            raise ValueError('Saved model physical step count disagrees with status')
        if not model.policy.optimizer.state_dict()['state']:
            raise ValueError('Saved optimizer state is absent')
        base_digests.append(frozen_base_digest(model.policy))
        budget.tick()
        batches = []
        with torch.no_grad():
            for first in range(0, len(labels), 125):
                stop = first + 125
                probability = model.policy.get_distribution(
                    torch.from_numpy(obs[first:stop]), masks[first:stop]
                ).distribution.probs.numpy()
                if not np.isfinite(probability).all() or (probability[~masks[first:stop]] != 0).any():
                    raise ValueError('Invalid masked policy probability')
                batches.append(probability)
                budget.tick()
        probabilities = np.concatenate(batches)
        target_probabilities = probabilities[np.arange(len(labels)), labels]
        choices = probabilities.argmax(-1)
        groups = {}
        for group, selection in [
                ('all2089', np.ones(len(labels), dtype=bool)),
                ('original2030', np.arange(len(labels)) < 2030),
                ('additional_w11_59', np.arange(len(labels)) >= 2030),
                ('reasonable_wait37', labels == 0),
                ('movement2052', labels != 0)]:
            wrong = selection & (choices != labels)
            groups[group] = dict(examples=int(selection.sum()),
                correct_examples=int((selection & ~wrong).sum()),
                mismatches=int(wrong.sum()),
                wait_to_motion=int((wrong & (labels == 0)).sum()),
                motion_to_wait=int((wrong & (labels != 0) & (choices == 0)).sum()),
                different_motion=int((wrong & (labels != 0) & (choices != 0)).sum()),
                mean_target_probability=float(target_probabilities[selection].mean()),
                minimum_target_probability=float(target_probabilities[selection].min()))
        item = dict(policy_sha256=digest(checkpoint), groups=groups,
                    physical_ppo_steps=model.num_timesteps,
                    optimizer_state_present=True, frozen_base_sha256=base_digests[-1])
        inputs[str(checkpoint.relative_to(ROOT))] = digest(checkpoint)
        if not is_reference:
            audit_path = ROOT / 'reports/stop_feedback_20261015' / (
                'w11_' + name + '2000_audit_20261016') / 'RESULTS.json'
            audit = json.loads(audit_path.read_text())
            if not audit['completed'] or audit['causally_verified_sampled_frames'] != 2000:
                raise ValueError('Actual causal physical samples must pass the separate audit')
            learning_path = audit_path.parent.parent / (
                'w11_' + name + '2000_learning_20261016') / 'RESULTS.json'
            learning = json.loads(learning_path.read_text())
            if not learning['completed'] or not learning['analysis_covers_final_training']:
                raise ValueError('Final checkpoint learning analysis must be complete')
            item.update(training_counts=status['counts'], training_resource=status['resource'],
                demonstration_retention=status['demonstration_retention'],
                initialization_frames_excluded=status['initialization_frames'],
                actual_sampling_audit=str(audit_path.relative_to(ROOT)),
                final_training_state_changes=learning['offline_choice_transitions'])
            for path in [folder / 'status.json', audit_path, learning_path]:
                inputs[str(path.relative_to(ROOT))] = digest(path)
        outcomes[name] = item
        del model
    if len(set(base_digests)) != 1:
        raise ValueError('Frozen base navigation policy changed between the runs')
    result = dict(completed=True, at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
        scope='Offline original-label retention after two complete independent 2k PPO runs; not navigation acceptance.',
        independent_physical_steps_per_trained_model=2000,
        retention=outcomes, inputs_sha256=inputs,
        source_sha256=digest(Path(__file__)),
        additional_environment_steps=0, physical_environment_created=False,
        current_model_or_runtime_changed=False, all_frozen_bases_equal=True,
        resource=budget.statistics(),
        limitations=[
            'Different action labels can still describe a useful maneuver; a mismatch alone is not a collision.',
            'Training episode outcomes are on-policy exposure, not matched independent navigation scores.',
            'The retained run has an actual training collision; lower wait counts cannot establish safety.',
            'Independent closed-loop paired results are required before preferring either PPO variant.',
            'Neither run is a continuation of the old 8k simulator state.'
        ], goal_achieved=False, deployment_approved=False)
    destination.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(completed=True, original_label_accuracy={
        name: outcome['groups']['all2089']['correct_examples']
        for name, outcome in outcomes.items()}, total_examples=len(labels),
        additional_environment_steps=0, resource=result['resource']), indent=2), flush=True)


if __name__ == '__main__':
    main()
