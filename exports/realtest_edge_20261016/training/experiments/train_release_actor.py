"""Bounded measured-context fit; no PPO update and no unsafe positive labels."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import shutil

import numpy as np
import torch
import torch.nn.functional as F

from experiments.long_risk_actor import LongRiskActor, digest
from experiments.release_actor import ReleaseActor, export
from experiments.train_long_risk_actor import state_digest
from neural_navigation.experiment_budget import ExperimentBudget
from neural_navigation.masked_actions import ACTIONS, masks_from_observation
from neural_navigation.release_observation import contract


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'reports/stop_feedback_20261015'


def write(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def main(args):
    args.dataset, args.risk_model, args.output = args.dataset.resolve(), args.risk_model.resolve(), args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    budget = ExperimentBudget(.35, memory_mib=3072, directory=args.output)
    source_result = args.dataset.parent / 'RESULTS.json'
    sources = json.loads(source_result.read_text())
    assert sources['completed'] and sources['frames'] == 1790 and sources['failed_latest_trace_excluded']
    assert digest(args.dataset) == sources['dataset_sha256']
    paths = [Path(__file__), args.dataset, source_result, args.risk_model / 'checkpoint.pt',
        args.risk_model / 'parent_actor.pt', args.risk_model / 'portable/manifest.json',
        ROOT / 'training/experiments/release_actor.py',
        ROOT / 'training/experiments/long_risk_actor.py',
        ROOT / 'training/neural_navigation/release_observation.py',
        ROOT / 'training/neural_navigation/release_runtime.py',
        BASE / 'long_risk_protocol_20261016/PLAN.json',
        BASE / 'long_risk_protocol_20261016/new_sensor_cases.json']
    inputs = {str(path.relative_to(ROOT)): digest(path) for path in paths}
    write(args.output / 'PLAN.json', dict(inputs=inputs, contract=contract(), seed=args.seed,
        maximum_supervised_updates=args.maximum_updates, new_ppo_steps=0,
        only_complete_safe_positive_sources=True, head_dimensions=[312, 32, 32, 12],
        initialize_exact_old_risk_decisions=True, failed_latest_trace_excluded=True,
        old_noise_cases_used_as_development=True, fresh_four_cases_not_used=True,
        localizer_router_parent_actor_docking_and_guard_frozen=True,
        physical_navigation_must_be_independently_verified=True))
    prior_state = torch.load(args.risk_model / 'checkpoint.pt', map_location='cpu', weights_only=False)
    prior = LongRiskActor(torch.jit.load(str(args.risk_model / 'parent_actor.pt'), map_location='cpu')).eval()
    prior.load_state_dict(prior_state['actor'])
    actor = ReleaseActor(torch.jit.load(str(args.risk_model / 'parent_actor.pt'), map_location='cpu')).eval()
    actor.initialize_risk_head(prior)
    frozen = state_digest(actor.parent)
    with np.load(args.dataset, allow_pickle=False) as archive:
        data = {key: archive[key].copy() for key in archive.files}
    observations, masks, labels = data['observations'], data['masks'], data['choices']
    assert observations.shape == (1790, 625)
    np.testing.assert_array_equal(masks_from_observation(observations[:, :282]), masks)
    assert masks[:, 0].all() and masks[np.arange(len(labels)), labels].all()
    x = torch.from_numpy(observations)
    with torch.no_grad():
        np.testing.assert_array_equal(actor(x).numpy(), prior(x[:, :481]).numpy())
        features, baseline = actor.features(x)
        active = actor.support(x)
    nominal = np.abs(baseline.numpy()[:, None] - ACTIONS[None]).max(2).argmin(1)
    inactive_mismatches = np.flatnonzero(~active.numpy() & (nominal != labels))
    admission = dict(completed=True, full_safe_source_frames=len(labels), active_frames=int(active.sum()),
        active_stop_labels=int((active.numpy() & (labels == 0)).sum()),
        active_motion_labels=int((active.numpy() & (labels != 0)).sum()),
        inactive_target_disagreements=inactive_mismatches.tolist(),
        initial_old_decisions_exact=True, source_reports=sources['sources'])
    write(args.output / 'SOURCE_ADMISSION.json', admission)
    if len(inactive_mismatches):
        write(args.output / 'RESULTS.json', dict(completed=True, skipped=True,
            reason='Positive behavior outside trainable support', new_optimizer_updates=0,
            exported_for_physical_validation=False, new_ppo_steps=0, goal_achieved=False))
        return
    features = features[active].detach()
    targets = torch.from_numpy(labels)[active]
    mask = torch.from_numpy(masks)[active]
    stop, move = targets == 0, targets != 0
    observed = {}
    for i, (value, target) in enumerate(zip(features.numpy(), targets.numpy())):
        previous = observed.setdefault(value.tobytes(), (int(target), i))
        if previous[0] != target:
            raise ValueError('Conflicting identical release input: ' + repr((previous[1], i)))
    optimizer = torch.optim.Adam(actor.head.parameters(), lr=.001)
    records = []
    for update in range(1, args.maximum_updates + 1):
        logits = (actor.head(features) + 2. * features[:, 76:88]).masked_fill(~mask, -1e8)
        losses = F.cross_entropy(logits, targets, reduction='none')
        loss = .5 * losses[stop].mean() + .5 * losses[move].mean()
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(actor.head.parameters(), 1.)
        optimizer.step()
        if update % 10 == 0:
            budget.tick()
        if update % 50 == 0 or update == args.maximum_updates:
            with torch.no_grad():
                probability = (actor.head(features) + 2. * features[:, 76:88]).masked_fill(~mask, -1e8).softmax(1)
                p = probability[torch.arange(len(targets)), targets]
            record = dict(update=update, active_correct=int((probability.argmax(1) == targets).sum()),
                active_examples=len(targets), loss=float(loss.detach()),
                minimum_stop_target_probability=float(p[stop].min()),
                minimum_motion_target_probability=float(p[move].min()))
            records.append(record)
            write(args.output / 'STATUS.json', dict(status='bounded_release_head_fit', **record))
            if min(record['minimum_stop_target_probability'], record['minimum_motion_target_probability']) >= .8:
                break
    assert state_digest(actor.parent) == frozen
    assert not any(parameter.requires_grad for parameter in actor.parent.parameters())
    with torch.no_grad():
        predictions = actor(x).numpy()
    correct = int((predictions == ACTIONS[labels]).all(1).sum())
    shutil.copy2(args.risk_model / 'parent_actor.pt', args.output / 'parent_actor.pt')
    shutil.copy2(args.dataset, args.output / 'demonstrations.npz')
    torch.save(dict(actor=actor.state_dict(), optimizer=optimizer.state_dict(),
        torch_rng=torch.get_rng_state(), numpy_rng=np.random.get_state(), python_rng=random.getstate(),
        parent_actor_sha256=digest(args.output / 'parent_actor.pt'), contract=contract(),
        initialized_from_risk_checkpoint_sha256=digest(args.risk_model / 'checkpoint.pt'),
        supervised_updates=update, new_ppo_steps=0, exact_physical_rollout_resume=False),
        args.output / 'checkpoint.pt')
    accepted = correct == len(labels)
    if accepted:
        invalid, near, stopped = (observations[:16].copy() for _ in range(3))
        invalid[:, 435] = 0.
        near[:, 218:220] = 0.
        near[:, 302] = 1.
        stopped[:, 235:244] = 0.
        stopped[:, 267:269] = 0.
        checks = np.r_[observations, invalid, near, stopped]
        export(actor, args.risk_model / 'portable', args.output / 'portable', checks)
        restored = ReleaseActor(torch.jit.load(str(args.output / 'parent_actor.pt'), map_location='cpu')).eval()
        state = torch.load(args.output / 'checkpoint.pt', map_location='cpu', weights_only=False)
        restored.load_state_dict(state['actor'])
        opt = torch.optim.Adam(restored.head.parameters(), lr=.001)
        opt.load_state_dict(state['optimizer'])
        assert opt.state_dict()['state']
        with torch.no_grad():
            np.testing.assert_array_equal(restored(torch.from_numpy(checks)).numpy(), actor(torch.from_numpy(checks)).numpy())
    for name, expected in inputs.items():
        assert digest(ROOT / name) == expected, name
    budget.tick()
    write(args.output / 'RESULTS.json', dict(completed=True, at=datetime.now(timezone.utc).isoformat(),
        correct_source_labels=correct, total_source_labels=len(labels), updates=update,
        fit_records=records, frozen_parent_tensor_sha256=frozen,
        checkpoint_sha256=digest(args.output / 'checkpoint.pt'),
        optimizer_and_rng_saved=True, exported_for_physical_validation=accepted,
        new_ppo_steps=0, new_physical_steps=0, resource=budget.statistics(),
        goal_achieved=False, deployment_approved=False))
    write(args.output / 'STATUS.json', dict(status='complete', exported_for_physical_validation=accepted))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--risk-model', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=231016)
    parser.add_argument('--maximum-updates', type=int, default=1500)
    args = parser.parse_args()
    if not 1 <= args.maximum_updates <= 2000:
        parser.error('Bounded experiment requires 1..2000 supervised updates')
    main(args)
