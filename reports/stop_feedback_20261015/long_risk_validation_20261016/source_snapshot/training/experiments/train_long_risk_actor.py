"""Bounded supervised risk-head experiment using only complete-safe replays."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import shutil

import numpy as np
import torch
import torch.nn.functional as F

from experiments.long_risk_actor import LongRiskActor, digest, export
from neural_navigation.experiment_budget import ExperimentBudget
from neural_navigation.long_risk_observation import contract
from neural_navigation.masked_actions import ACTIONS, masks_from_observation


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'reports/stop_feedback_20261015'


def write(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def state_digest(module):
    value = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        value.update(name.encode())
        value.update(tensor.detach().cpu().numpy().tobytes())
    return value.hexdigest()


def main(args):
    args.dataset, args.parent, args.output = args.dataset.resolve(), args.parent.resolve(), args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    budget = ExperimentBudget(args.cpu_cores, memory_mib=3072, directory=args.output)
    result_path = args.dataset.parent / 'RESULTS.json'
    sources = json.loads(result_path.read_text())
    assert sources['completed'] and sources['frames'] == 550
    assert digest(args.dataset) == sources['dataset_sha256']
    protocol = BASE / 'long_risk_protocol_20261016/PLAN.json'
    assert json.loads(protocol.read_text())['new_seed_cases_must_not_enter_fitting']
    paths = [Path(__file__), args.dataset, result_path, args.parent / 'actor.pt',
        args.parent / 'manifest.json', protocol, BASE / 'long_risk_protocol_20261016/new_sensor_cases.json',
        ROOT / 'training/neural_navigation/long_risk_observation.py',
        ROOT / 'training/neural_navigation/long_risk_runtime.py',
        ROOT / 'training/experiments/long_risk_actor.py']
    inputs = {str(path.relative_to(ROOT)): digest(path) for path in paths}
    write(args.output / 'PLAN.json', dict(inputs=inputs, seed=args.seed,
        contract=contract(), maximum_supervised_updates=args.maximum_updates,
        new_ppo_steps=0, parent_actor_frozen=True, failed_or_partial_sources_excluded=True,
        no_runtime_teacher=True, active_features=88, head_width=32,
        unknown_risk_requires_learned_wait_context=True,
        physical_validation_required_after_fit=True))
    parent = torch.jit.load(str(args.parent / 'actor.pt'), map_location='cpu').eval()
    actor = LongRiskActor(parent).eval()
    frozen = state_digest(actor.parent)
    with np.load(args.dataset, allow_pickle=False) as archive:
        arrays = {key: archive[key].copy() for key in archive.files}
    observations = arrays['observations']
    masks = arrays['masks']
    labels = arrays['choices']
    assert observations.shape == (550, 481) and masks.shape == (550, 12)
    np.testing.assert_array_equal(masks_from_observation(observations[:, :282]), masks)
    assert masks[:, 0].all() and masks[np.arange(len(labels)), labels].all()
    x = torch.from_numpy(observations)
    with torch.no_grad():
        features, baseline = actor.features(x)
        active = actor.support(x)
        nominal = np.abs(baseline.numpy()[:, None, :] - ACTIONS[None, :, :]).max(axis=2).argmin(axis=1)
    np.testing.assert_array_equal(nominal, arrays['parent_choices'])
    mismatches = np.flatnonzero(~active.numpy() & (nominal != labels))
    write(args.output / 'SOURCE_ADMISSION.json', dict(completed=True,
        full_safe_replay_source_frames=len(labels), active_frames=int(active.sum()),
        active_stop_labels=int(((labels == 0) & active.numpy()).sum()),
        active_motion_labels=int(((labels != 0) & active.numpy()).sum()),
        inactive_target_disagreements=mismatches.tolist(),
        admitted=not len(mismatches), no_failed_or_partial_positive_sources=True,
        original_sources=sources['sources']))
    if len(mismatches):
        write(args.output / 'RESULTS.json', dict(completed=True, skipped=True,
            reason='Target behavior lies outside trainable support; do not silently discard it',
            new_optimizer_updates=0, new_ppo_steps=0, exported_for_physical_validation=False,
            goal_achieved=False, deployment_approved=False))
        return
    mask_tensor = torch.from_numpy(masks)[active]
    targets = torch.from_numpy(labels)[active]
    features = features[active].detach()
    stop, move = targets == 0, targets != 0
    assert stop.any() and move.any()
    conflicts = {}
    for index, (value, target) in enumerate(zip(features.numpy(), targets.numpy())):
        key = value.tobytes()
        previous = conflicts.setdefault(key, (int(target), index))
        if previous[0] != int(target):
            raise ValueError('Conflicting identical head inputs: ' + repr((previous[1], index)))
    optimizer = torch.optim.Adam(actor.head.parameters(), lr=.002)
    records = []
    for update in range(1, args.maximum_updates + 1):
        logits = actor.head(features) + 2. * features[:, -12:]
        logits = logits.masked_fill(~mask_tensor, -1e8)
        losses = F.cross_entropy(logits, targets, reduction='none')
        loss = .5 * losses[stop].mean() + .5 * losses[move].mean()
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(actor.head.parameters(), 1.)
        optimizer.step()
        budget.tick()
        if update % 25 == 0 or update == args.maximum_updates:
            with torch.no_grad():
                probability = (actor.head(features) + 2. * features[:, -12:]).masked_fill(
                    ~mask_tensor, -1e8).softmax(dim=1)
                correct = int((probability.argmax(-1) == targets).sum())
                p = probability[torch.arange(len(targets)), targets]
            record = dict(update=update, loss=float(loss.detach()), active_correct=correct,
                active_examples=len(targets), minimum_stop_target_probability=float(p[stop].min()),
                minimum_motion_target_probability=float(p[move].min()))
            records.append(record)
            write(args.output / 'STATUS.json', dict(status='bounded_head_fit', **record))
            if min(record['minimum_stop_target_probability'], record['minimum_motion_target_probability']) >= .9:
                break
    with torch.no_grad():
        commands = actor(x).numpy()
    predicted = np.abs(commands[:, None, :] - ACTIONS[None, :, :]).max(axis=2).argmin(axis=1)
    np.testing.assert_array_equal(commands, ACTIONS[predicted])
    assert state_digest(actor.parent) == frozen
    assert not any(parameter.requires_grad for parameter in actor.parent.parameters())
    correct = int((predicted == labels).sum())
    shutil.copy2(args.parent / 'actor.pt', args.output / 'parent_actor.pt')
    shutil.copy2(args.dataset, args.output / 'demonstrations.npz')
    torch.save(dict(actor=actor.state_dict(), optimizer=optimizer.state_dict(),
        torch_rng=torch.get_rng_state(), numpy_rng=np.random.get_state(), python_rng=random.getstate(),
        parent_actor_sha256=digest(args.output / 'parent_actor.pt'), contract=contract(),
        supervised_updates=update, new_ppo_steps=0, exact_physical_rollout_resume=False),
        args.output / 'checkpoint.pt')
    accepted = correct == len(labels)
    if accepted:
        # Include sensing failures, stop-only masks and terminal headings in export parity.
        disabled = observations[:16].copy()
        disabled[:, 435] = 0.
        near = observations[:16].copy()
        near[:, 218:220] = 0.
        near[:, 302] = 1.
        stopped = observations[:16].copy()
        stopped[:, 235:244] = 0.
        stopped[:, 267:269] = 0.
        checks = np.r_[observations, disabled, near, stopped]
        export(actor, args.parent, args.output / 'portable', checks)
        restored = LongRiskActor(torch.jit.load(str(args.output / 'parent_actor.pt'), map_location='cpu')).eval()
        state = torch.load(args.output / 'checkpoint.pt', map_location='cpu', weights_only=False)
        assert state['contract'] == contract()
        restored.load_state_dict(state['actor'])
        restored_optimizer = torch.optim.Adam(restored.head.parameters(), lr=.002)
        restored_optimizer.load_state_dict(state['optimizer'])
        assert restored_optimizer.state_dict()['state']
        with torch.no_grad():
            np.testing.assert_array_equal(restored(torch.from_numpy(checks)).numpy(), actor(torch.from_numpy(checks)).numpy())
    for name, expected in inputs.items():
        assert digest(ROOT / name) == expected, name
    write(args.output / 'RESULTS.json', dict(completed=True, at=datetime.now(timezone.utc).isoformat(),
        correct_source_labels=correct, total_source_labels=len(labels), updates=update,
        fit_records=records, frozen_parent_tensor_sha256=frozen,
        checkpoint_sha256=digest(args.output / 'checkpoint.pt'),
        exported_for_physical_validation=accepted, optimizer_and_rng_saved=True,
        new_ppo_steps=0, new_physical_steps=0, resource=budget.statistics(),
        no_claim_of_learned_navigation_acceptance=True, goal_achieved=False, deployment_approved=False))
    write(args.output / 'STATUS.json', dict(status='complete', exported_for_physical_validation=accepted))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--parent', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=231016)
    parser.add_argument('--maximum-updates', type=int, default=1000)
    parser.add_argument('--cpu-cores', type=float, default=.35)
    args = parser.parse_args()
    if not 1 <= args.maximum_updates <= 1000:
        parser.error('One bounded experiment: 1..1000 updates')
    try:
        main(args)
    except Exception as error:
        if args.output.exists():
            write(args.output / 'FAILURE.json', dict(error=repr(error), automatically_restarted=False))
        raise
