"""Verify an archived, unfinished PPO run without resuming or issuing commands."""
import argparse
import hashlib
import json
from pathlib import Path


def verify(root, relative_run):
    import numpy as np
    import torch
    from experiments.stop_recovery_policy import StopRecoveryPPO
    from experiments.train_prefix_recovery import frozen_base_digest

    torch.set_num_threads(1)
    root = Path(root).resolve()
    run = (root / relative_run).resolve()
    run.relative_to(root)
    snapshot = json.loads((run / 'SNAPSHOT.json').read_text())
    if snapshot['training_completed'] or snapshot['deployment_approved']:
        raise ValueError('This checker expects an unfinished experimental snapshot')
    for name, expected in snapshot['files'].items():
        path = (run / name).resolve()
        path.relative_to(run)
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError('Training snapshot file mismatch: ' + name)
    model = StopRecoveryPPO.load(run / snapshot['latest_checkpoint'], device='cpu')
    bc = StopRecoveryPPO.load(run / 'policy_bc.zip', device='cpu')
    steps = snapshot['checkpoint_collected_steps']
    if (model.num_timesteps != steps or model.observation_space.shape != (436,)
            or bc.num_timesteps != 0
            or frozen_base_digest(model.policy) != frozen_base_digest(bc.policy)):
        raise ValueError('Partial checkpoint contract mismatch')
    total = 0
    for path in sorted((run / 'rollouts').glob('*.npz')):
        with np.load(path, allow_pickle=False) as data:
            metadata = json.loads(str(data['metadata_json']))
            total += len(data['observations'])
            if metadata['collected_steps'] == steps:
                if model._n_updates != metadata['optimizer_epochs_before_update']:
                    raise ValueError('Pre-update checkpoint optimizer state mismatch')
                observations = data['observations'][:, 0].copy()
                masks = data['action_masks'][:, 0].astype(bool)
    if total != steps or steps - 250 != snapshot['checkpoint_optimized_through_steps']:
        raise ValueError('Incomplete actual rollout history')
    if not model.policy.optimizer.state_dict()['state']:
        raise ValueError('Missing trained optimizer state')
    with torch.no_grad():
        probabilities = model.policy.get_distribution(
            torch.from_numpy(observations), masks).distribution.probs.numpy()
    if (not np.isfinite(probabilities).all()
            or np.any(probabilities[~masks] > 1e-7)
            or not np.allclose(probabilities.sum(axis=1), 1., atol=1e-6)):
        raise ValueError('Masked policy inference failed')
    return dict(completed=True, snapshot_files_checked=len(snapshot['files']),
        source_collected_steps=snapshot['source_collected_steps'],
        checkpoint_collected_steps=steps,
        checkpoint_optimized_through_steps=steps-250,
        optimizer_epochs_completed=model._n_updates,
        optimizer_loaded=True, original_base_unchanged=True,
        actual_samples_archived=total, masked_inference_frames=len(observations),
        checkpoint_sha256=hashlib.sha256((run/snapshot['latest_checkpoint']).read_bytes()).hexdigest(),
        training_completed=False, exact_rollout_resume_supported=False,
        closed_loop_acceptance=False, deployment_approved=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-run', required=True)
    args = parser.parse_args()
    print(json.dumps(verify(Path(__file__).resolve().parents[1], args.model_run), indent=2))


if __name__ == '__main__':
    main()
