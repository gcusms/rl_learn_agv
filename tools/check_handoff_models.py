"""Load the handoff checkpoints and exports without training or navigation."""
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ[name] = '1'
    os.chdir(ROOT)
    sys.path[:0] = [str(ROOT / 'training'), str(ROOT)]
    import numpy as np
    import torch
    from experiments.stop_recovery_policy import StopRecoveryPPO
    from experiments.train_prefix_recovery import frozen_base_digest
    from neural_navigation.experiment_budget import ExperimentBudget
    from neural_navigation.masked_actions import ACTIONS, masks_from_observation

    torch.set_num_threads(1)
    budget = ExperimentBudget(.15, memory_mib=2048, directory=ROOT)
    base = ROOT / 'models/stop_feedback_20261015'
    with np.load(base / 'early_transfer_bc800_s231015/demonstrations.npz', allow_pickle=False) as z:
        original = {key:z[key].copy() for key in ('observations', 'masks', 'choices')}
    rows = []
    for folder, count, correct_expected, parent in (
        ('actual95_transfer_bc400_s231016', 2421, 2421, None),
        ('actual95_control_bc400_s231016', 2355, 2354, None),
        ('recovery_followup_bc400_s231016', 2505, 2505, 'actual95_transfer_bc400_s231016'),
        ('early_yield_bc400_s231016', 2638, 2638, 'recovery_followup_bc400_s231016'),
        ('early_yield_remaining_bc600_s231016', 2915, 2915, 'early_yield_bc400_s231016'),
    ):
        path = base / folder
        bundle = path / 'bc_portable'
        manifest = json.loads((bundle / 'manifest.json').read_text())
        if manifest['observation_contract']['dimensions'] != 439:
            raise ValueError('Runtime contract changed: ' + folder)
        missing_native = []
        for name, expected in manifest['files'].items():
            file = bundle / name
            if file.suffix == '.so' and not file.exists():
                missing_native.append(name)
            elif digest(file) != expected:
                raise ValueError('Export checksum changed: ' + str(file))
        model = StopRecoveryPPO.load(path / 'policy_bc.zip', device='cpu')
        if model.num_timesteps != 0 or frozen_base_digest(model.policy) != 'e5502406c4ee8e42c4da09dd6affa022cb76e7f9b01316b1eaf0d2551431938e':
            raise ValueError('Step count or frozen base changed')
        continuation = torch.load(path / 'continuation_state.pt', map_location='cpu', weights_only=False)
        model.policy.optimizer.load_state_dict(continuation['optimizer'])
        if not model.policy.optimizer.state_dict()['state']:
            raise ValueError('Optimizer is missing')
        if continuation['torch_rng'].dtype != torch.uint8 or continuation['numpy_rng'][0] != 'MT19937':
            raise ValueError('Random state is missing')
        with np.load(path / 'demonstrations.npz', allow_pickle=False) as z:
            obs, masks, labels = [z[key].copy() for key in ('observations', 'masks', 'choices')]
        if obs.shape != (count, 436) or masks.shape != (count, 12):
            raise ValueError('Dataset dimensions changed')
        expected_data = original
        if parent is not None:
            with np.load(base / parent / 'demonstrations.npz', allow_pickle=False) as z:
                expected_data = {key: z[key].copy() for key in ('observations', 'masks', 'choices')}
        retained_count = len(expected_data['choices'])
        for key, value in zip(('observations', 'masks', 'choices'), (obs, masks, labels)):
            np.testing.assert_array_equal(value[:retained_count], expected_data[key])
        np.testing.assert_array_equal(masks_from_observation(obs[:, :282]), masks)
        if not masks[:, 0].all() or not masks[np.arange(count), labels].all():
            raise ValueError('Demonstration action legality changed')
        model.policy.set_training_mode(False)
        with torch.no_grad():
            predicted = model.policy.get_distribution(torch.from_numpy(obs), masks).distribution.probs.argmax(-1).numpy()
        correct = int((predicted == labels).sum())
        stop_indexes = np.flatnonzero(labels == 0)
        if correct != correct_expected or not np.array_equal(predicted[stop_indexes], labels[stop_indexes]):
            raise ValueError('Saved decisions or necessary stopping changed: ' + folder)
        budget.tick()
        indexes = np.unique(np.r_[stop_indexes, np.arange(retained_count, count),
                                  np.linspace(0, retained_count - 1, 128, dtype=int)])
        actor = torch.jit.load(str(bundle / 'actor.pt'), map_location='cpu').eval()
        for index in indexes:
            runtime = np.r_[obs[index, :300], np.zeros(3, np.float32), obs[index, 300:]].astype(np.float32)
            with torch.no_grad():
                command = actor(torch.from_numpy(runtime[None])).numpy()[0]
            np.testing.assert_array_equal(command, ACTIONS[predicted[index]])
            budget.tick()
        rows.append(dict(folder=folder, checkpoint_sha256=digest(path / 'policy_bc.zip'),
            examples=count, correct=correct, previous_examples_retained=retained_count,
            all_stop_labels_retained=len(stop_indexes), exported_decisions_checked=len(indexes),
            optimizer_and_rng_loaded=True, frozen_base_unchanged=True,
            native_libraries_to_rebuild=missing_native))
    print(json.dumps(dict(completed=True, models=rows, resource=budget.statistics(),
        new_physical_steps=0, new_optimizer_updates=0, new_ppo_steps=0,
        deployment_approved=False, scope='Model/data/export portability checks; no navigation acceptance.'), indent=2))


if __name__ == '__main__':
    main()
