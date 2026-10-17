"""Data-only ablation: retain verified normal decisions in the408D head.

Initialization, architecture, phase weighting, optimizer, seed and maximum
updates match guard_reason2000. Only the admitted source set changes.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import shutil

import numpy as np
import torch
import torch.nn.functional as F

from experiments.context_release_actor import ContextReleaseActor, contract as support_contract
from experiments.guard_reason_release_actor import GuardReasonReleaseActor, contract as head_contract, export
from experiments.long_risk_actor import LongRiskActor, digest
from experiments.release_fit_weights import fit_weights
from experiments.train_long_risk_actor import state_digest
from neural_navigation.experiment_budget import ExperimentBudget
from neural_navigation.masked_actions import ACTIONS, masks_from_observation
from neural_navigation.release_observation import contract


ROOT = Path(__file__).resolve().parents[2]


def write(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def main(args):
    dataset, risk, control, output = [getattr(args, key).resolve()
        for key in ['dataset', 'risk_model', 'control', 'output']]
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.manual_seed(231016); np.random.seed(231016); random.seed(231016)
    budget = ExperimentBudget(args.cpu_cores, memory_mib=1024, directory=output)
    sources = json.loads((dataset.parent / 'RESULTS.json').read_text())
    original_plan = json.loads((control / 'PLAN.json').read_text())
    original_result = json.loads((control / 'RESULTS.json').read_text())
    assert sources['completed'] and sources['admitted'] and sources['original_prefix_byte_exact']
    assert sources['frames'] == 3728 and sources['parent_frames'] == 3252
    assert digest(dataset) == sources['dataset_sha256']
    assert original_plan['seed'] == 231016 and original_plan['maximum_supervised_updates'] == 2000
    assert original_plan['head_contract'] == head_contract()
    assert original_result['completed'] and original_result['updates'] == 2000
    assert digest(control / 'checkpoint.pt') == original_result['checkpoint_sha256']
    for path in [risk / 'checkpoint.pt', risk / 'parent_actor.pt']:
        assert digest(path) == original_plan['inputs'][str(path.relative_to(ROOT))]
    parent_plan = json.loads((dataset.parent / 'PLAN.json').read_text())
    original_dataset = ROOT / next(name for name in parent_plan['inputs']
        if name.endswith('context_feedback_sources_20261016/demonstrations.npz'))
    files = [Path(__file__), dataset, dataset.parent / 'RESULTS.json',
        dataset.parent / 'PLAN.json', dataset.parent / 'SOURCE_ADMISSION.json', original_dataset,
        control / 'PLAN.json', control / 'RESULTS.json', control / 'checkpoint.pt',
        risk / 'checkpoint.pt', risk / 'parent_actor.pt', risk / 'portable/manifest.json']
    for module in ['guard_reason_release_actor', 'context_release_actor', 'release_actor',
            'long_risk_actor', 'release_fit_weights', 'train_long_risk_actor']:
        files.append(ROOT / ('training/experiments/' + module + '.py'))
    hashes = {str(p.relative_to(ROOT)): digest(p) for p in files}
    plan = dict(inputs=hashes, seed=231016, maximum_supervised_updates=2000,
        control=str(control.relative_to(ROOT)), only_changed_factor='476verified_normal_retention_labels',
        initialization='same long_risk_bc1000 checkpoint as guard_reason2000; not continued from its fitted weights',
        same_architecture_support_optimizer_learning_rate_and_update_limit=True,
        original3252_arrays_byte_exact=True, phase_weight_rule_unchanged=True,
        phase_weights_computed_on_original312features=True,
        contract=contract(), head_contract=head_contract(), learning_support=support_contract(),
        source_fit_gate='all3728labels_correct; no old or safe-stop label discarded',
        no_physical_tests_before_source_fit_and_export_checks=True,
        frozen_parent_localizer_router_docking_guard_and_permissions=True,
        maximum_rss_mib=1024, cpu_soft_cores=args.cpu_cores,
        new_ppo_steps=0, deployment_approved=False)
    write(output / 'PLAN.json', plan)
    prior = LongRiskActor(torch.jit.load(str(risk / 'parent_actor.pt'), map_location='cpu')).eval()
    prior.load_state_dict(torch.load(risk / 'checkpoint.pt', map_location='cpu', weights_only=False)['actor'])
    initial = ContextReleaseActor(torch.jit.load(str(risk / 'parent_actor.pt'), map_location='cpu')).eval()
    initial.initialize_risk_head(prior)
    actor = GuardReasonReleaseActor(torch.jit.load(str(risk / 'parent_actor.pt'), map_location='cpu')).eval()
    actor.initialize_release_head(initial)
    frozen = state_digest(actor.parent)
    with np.load(dataset, allow_pickle=False) as archive:
        arrays = {k: archive[k].copy() for k in archive.files}
    with np.load(original_dataset, allow_pickle=False) as archive:
        for key in archive.files:
            assert arrays[key][:3252].tobytes() == archive[key].tobytes(), key
    observations, masks, labels = [arrays[k] for k in ['observations', 'masks', 'choices']]
    assert observations.shape == (3728, 625)
    np.testing.assert_array_equal(masks_from_observation(observations[:, :282]), masks)
    assert masks[:, 0].all() and masks[np.arange(len(labels)), labels].all()
    x = torch.from_numpy(observations)
    with torch.no_grad():
        torch.testing.assert_close(actor.logits(x), initial.logits(x), rtol=0, atol=0)
        torch.testing.assert_close(actor(x), initial(x), rtol=0, atol=0)
        features, baseline = actor.features(x)
        active = actor.support(x)
    nominal = np.abs(baseline.numpy()[:, None] - ACTIONS[None]).max(2).argmin(1)
    assert not ((~active.numpy()) & (nominal != labels)).any()
    weights, weight_report = fit_weights(features[active, :312].numpy(), labels[active.numpy()],
        observations[active.numpy(), 434] * 5, mode='recovery_phase',
        recovery_motion_share=.5, recovery_wait_s=2.)
    assert weight_report == sources['training_weight_preflight']
    np.savez_compressed(output / 'fixed_source_weights.npz', active=active.numpy(), weights=weights)
    plan.update(initial_actor_tensor_sha256=state_digest(actor), fit_weighting=weight_report)
    write(output / 'PLAN.json', plan)
    features = features[active].detach()
    targets = torch.from_numpy(labels)[active]
    mask = torch.from_numpy(masks)[active]
    weights = torch.from_numpy(weights).to(dtype=features.dtype)
    optimizer = torch.optim.Adam(actor.head.parameters(), lr=.001)
    records = []
    budget.tick()
    for update in range(1, 2001):
        logits = (actor.head(features) + 2. * features[:, 76:88]).masked_fill(~mask, -1e8)
        loss = (F.cross_entropy(logits, targets, reduction='none') * weights).sum()
        optimizer.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(actor.head.parameters(), 1.)
        optimizer.step()
        if update % 10 == 0:
            budget.tick()
        if update % 50 == 0 or update == 2000:
            with torch.no_grad():
                probability = (actor.head(features) + 2. * features[:, 76:88]).masked_fill(~mask, -1e8).softmax(1)
                p = probability[torch.arange(len(targets)), targets]
                all_predictions = actor(x).numpy()
            correct = np.all(all_predictions == ACTIONS[labels], axis=1)
            records.append(dict(update=update, loss=float(loss.detach()),
                correct_source_labels=int(correct.sum()), total_source_labels=len(labels),
                correct_old3252_labels=int(correct[:3252].sum()),
                correct_new476_labels=int(correct[3252:].sum()),
                safe_stop_labels_retained=bool(correct[labels == 0].all()),
                minimum_stop_target_probability=float(p[targets == 0].min()),
                minimum_motion_target_probability=float(p[targets != 0].min())))
            write(output / 'STATUS.json', dict(status='normal_retention_fit', **records[-1]))
            if min(records[-1]['minimum_stop_target_probability'], records[-1]['minimum_motion_target_probability']) >= .8:
                break
    assert state_digest(actor.parent) == frozen
    assert not any(p.requires_grad for p in actor.parent.parameters())
    with torch.no_grad():
        predictions = actor(x).numpy()
    correct = np.all(predictions == ACTIONS[labels], axis=1)
    accepted = bool(correct.all())
    shutil.copy2(risk / 'parent_actor.pt', output / 'parent_actor.pt')
    torch.save(dict(actor=actor.state_dict(), optimizer=optimizer.state_dict(),
        torch_rng=torch.get_rng_state(), numpy_rng=np.random.get_state(), python_rng=random.getstate(),
        parent_actor_sha256=digest(output / 'parent_actor.pt'), contract=contract(),
        head_contract=head_contract(), learning_support=support_contract(),
        supervised_updates=update, new_ppo_steps=0, exact_physical_rollout_resume=False,
        source_sha256=digest(dataset), loss_weights_sha256=digest(output / 'fixed_source_weights.npz')),
        output / 'checkpoint.pt')
    if accepted:
        export(actor, risk / 'portable', output / 'portable', observations)
    restored = GuardReasonReleaseActor(torch.jit.load(str(output / 'parent_actor.pt'), map_location='cpu')).eval()
    saved = torch.load(output / 'checkpoint.pt', map_location='cpu', weights_only=False)
    restored.load_state_dict(saved['actor'])
    restored_optimizer = torch.optim.Adam(restored.head.parameters(), lr=.001)
    restored_optimizer.load_state_dict(saved['optimizer'])
    assert restored_optimizer.state_dict()['state']
    with torch.no_grad():
        np.testing.assert_array_equal(restored(x).numpy(), predictions)
    for name, expected in hashes.items():
        assert digest(ROOT / name) == expected, name
    budget.tick()
    write(output / 'RESULTS.json', dict(completed=True, at=datetime.now(timezone.utc).isoformat(),
        updates=update, correct_source_labels=int(correct.sum()), total_source_labels=len(labels),
        wrong_source_indices=np.flatnonzero(~correct).tolist(),
        original3252_all_retained=bool(correct[:3252].all()),
        original2848_all_retained=bool(correct[:2848].all()),
        added476_all_retained=bool(correct[3252:].all()),
        safe_stop_labels_all_retained=bool(correct[labels == 0].all()),
        fit_weighting=weight_report, fit_records=records, initial_logits_and_actions_exact=True,
        frozen_parent_tensor_sha256=frozen, source_sha256=digest(dataset),
        checkpoint_sha256=digest(output / 'checkpoint.pt'), optimizer_and_rng_saved=True,
        exported_for_physical_validation=accepted, new_ppo_steps=0, new_physical_steps=0,
        resource=budget.statistics(), goal_achieved=False, deployment_approved=False))
    write(output / 'STATUS.json', dict(status='complete', exported_for_physical_validation=accepted))
    (output / 'driver_snapshot.py').write_text(Path(__file__).read_text())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ['dataset', 'risk-model', 'control', 'output']:
        parser.add_argument('--' + key, type=Path, required=True)
    parser.add_argument('--cpu-cores', type=float, default=.1)
    args = parser.parse_args()
    if not .01 <= args.cpu_cores <= .1:
        parser.error('Use an offline fit budget between.01and.1CPUcore')
    try:
        main(args)
    except Exception as error:
        if args.output.is_dir():
            write(args.output / 'FAILURE.json', dict(error=repr(error), automatically_restarted=False))
        raise
