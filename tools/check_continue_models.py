"""Check the two new continuation checkpoints without training or physics."""
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
    from neural_navigation.masked_actions import ACTIONS, masks_from_observation

    torch.set_num_threads(1)
    parent = ROOT / 'models/stop_feedback_20261015/early_transfer_bc800_s231015'
    with np.load(parent / 'demonstrations.npz', allow_pickle=False) as old:
        original = {key: old[key].copy() for key in ('observations', 'masks', 'choices')}
    frozen = 'e5502406c4ee8e42c4da09dd6affa022cb76e7f9b01316b1eaf0d2551431938e'
    rows = []
    for folder, count, expected_correct in (
        ('actual95_transfer_bc400_s231016', 2421, 2421),
        ('actual95_control_bc400_s231016', 2355, 2354),
    ):
        path = parent.parent / folder
        bundle = path / 'bc_portable'
        manifest = json.loads((bundle / 'manifest.json').read_text())
        if manifest['observation_contract']['dimensions'] != 439:
            raise ValueError('Runtime observation contract changed')
        missing_native = []
        for name, expected in manifest['files'].items():
            file = bundle / name
            if file.suffix == '.so' and not file.exists():
                missing_native.append(name)
            elif digest(file) != expected:
                raise ValueError('Export checksum mismatch: ' + folder + '/' + name)
        model = StopRecoveryPPO.load(path / 'policy_bc.zip', device='cpu')
        if model.num_timesteps != 0 or frozen_base_digest(model.policy) != frozen:
            raise ValueError('PPO step count or frozen base changed')
        if not model.policy.optimizer.state_dict()['state']:
            raise ValueError('Optimizer continuation state is missing')
        if not (path / 'continuation_state.pt').is_file():
            raise ValueError('Explicit RNG/optimizer state is missing')
        with np.load(path / 'demonstrations.npz', allow_pickle=False) as data:
            obs, masks, labels = [data[key].copy() for key in ('observations', 'masks', 'choices')]
        if obs.shape != (count, 436) or masks.shape != (count, 12):
            raise ValueError('Training input dimensions changed')
        for key, value in zip(('observations', 'masks', 'choices'), (obs, masks, labels)):
            if not np.array_equal(value[:2355], original[key]):
                raise ValueError('The original dataset changed: ' + key)
        if not np.array_equal(masks_from_observation(obs[:, :282]), masks):
            raise ValueError('Mask semantics changed')
        if not masks[np.arange(count), labels].all() or not masks[:, 0].all():
            raise ValueError('Demonstration legality or necessary stopping changed')
        model.policy.set_training_mode(False)
        with torch.no_grad():
            probabilities = model.policy.get_distribution(torch.from_numpy(obs), masks).distribution.probs
            choices = probabilities.argmax(-1).cpu().numpy()
        correct = int((choices == labels).sum())
        stops = int((choices[labels == 0] == 0).sum())
        old_stops = int((choices[:2355][labels[:2355] == 0] == 0).sum())
        if correct != expected_correct or old_stops != 55 or stops != int((labels == 0).sum()):
            raise ValueError('Saved model no longer reproduces its reported supervised result: '
                             + repr(dict(correct=correct, old_stops=old_stops, all_stops=stops)))
        # Heading fields are zero only in this interface check. All selected
        # examples belong to the far-goal recovery support, outside docking.
        indexes = np.unique(np.r_[np.linspace(0, count - 1, 64, dtype=int), np.arange(2355, count)])
        runtime_obs = np.concatenate((obs[indexes, :300], np.zeros((len(indexes), 3), np.float32),
                                      obs[indexes, 300:]), axis=1)
        actor = torch.jit.load(str(bundle / 'actor.pt'), map_location='cpu').eval()
        with torch.no_grad():
            commands = actor(torch.from_numpy(runtime_obs)).cpu().numpy()
        if not np.array_equal(commands, ACTIONS[choices[indexes]]):
            raise ValueError('Checkpoint and exported actor choose different actions')
        rows.append(dict(folder=folder, checkpoint_sha256=digest(path / 'policy_bc.zip'),
                         examples=count, correct=correct, old_necessary_stops_retained=old_stops,
                         all_stop_labels_retained=stops,
                         exported_decisions_checked=len(indexes), optimizer_loaded=True,
                         new_ppo_steps=0, native_libraries_to_rebuild=missing_native))
    print(json.dumps(dict(completed=True, models=rows, new_physical_steps=0,
                         new_optimizer_updates=0, deployment_approved=False,
                         scope='Checkpoint, data, mask, frozen-base and export parity only.'), indent=2))


if __name__ == '__main__':
    main()
