"""Paired500-update continuations from one identical fitted408D checkpoint.

The treatment changes only ordinary-motion loss aggregation. Stop and resumed
motion retain their original weights; neither treatment issues robot commands.
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

from experiments.guard_reason_release_actor import GuardReasonReleaseActor, export
from experiments.long_risk_actor import digest
from experiments.phase_tail_loss import phase_tail_loss
from experiments.train_long_risk_actor import state_digest
from neural_navigation.experiment_budget import ExperimentBudget
from neural_navigation.masked_actions import ACTIONS


ROOT = Path(__file__).resolve().parents[2]


def write(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def main(args):
    model, dataset, output = [getattr(args, key).resolve() for key in ['model', 'dataset', 'output']]
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    budget = ExperimentBudget(.1, memory_mib=1024, directory=output)
    source_fit = json.loads((model / 'RESULTS.json').read_text())
    assert source_fit['completed'] and source_fit['updates'] == 2000
    assert source_fit['wrong_source_indices'] == [3182, 3245, 3279]
    assert digest(model / 'checkpoint.pt') == source_fit['checkpoint_sha256']
    assert digest(dataset) == source_fit['source_sha256']
    saved = torch.load(model / 'checkpoint.pt', map_location='cpu', weights_only=False)
    with np.load(dataset, allow_pickle=False) as archive:
        observations, labels, masks = [archive[k].copy() for k in ['observations', 'choices', 'masks']]
    with np.load(model / 'fixed_source_weights.npz', allow_pickle=False) as archive:
        active, fixed_weights = archive['active'].copy(), archive['weights'].copy()
    files = [Path(__file__), model / 'checkpoint.pt', model / 'parent_actor.pt',
        model / 'RESULTS.json', model / 'fixed_source_weights.npz', dataset,
        ROOT / 'training/experiments/phase_tail_loss.py',
        ROOT / 'training/experiments/test_phase_tail_loss.py']
    for name in ['guard_reason_release_actor', 'context_release_actor', 'release_actor', 'long_risk_actor']:
        files.append(ROOT / ('training/experiments/' + name + '.py'))
    hashes = {str(p.relative_to(ROOT)): digest(p) for p in files}
    write(output / 'PLAN.json', dict(at=datetime.now(timezone.utc).isoformat(), inputs=hashes,
        prior_supervised_updates=2000, maximum_new_updates_per_model=500,
        maximum_new_models=2, ordered_models=['mean500', 'tail500'],
        same_initial_checkpoint_optimizer_rng_data_architecture_and_learning_rate=True,
        only_changed_factor='ordinary_motion_half_mean_half_log_mean_exp_CE_vs_original_mean',
        phase_gradient_budgets=dict(stop=.5, ordinary=.25, resumed=.25),
        tail_focus_share=.5, tail_alpha=1., all_examples_have_positive_gradient=True,
        common_early_stop='allsource labels correct and minimumactive target probability>=.6',
        physical_candidate_selection='prefer mean500 if fullfit; otherwise tail500 if fullfit; else no physical trial',
        no_physics_before_full_source_fit_and_portable_checks=True,
        no_dataset_or_runtime_contract_change=True, source_fit_gate='all3728labels_correct',
        no_optimizer_updates_to_parent_localizer_router_docking_or_guard=True,
        cpu_soft_cores=.1, maximum_rss_mib=1024, new_ppo_steps=0, deployment_approved=False))
    (output / 'driver_snapshot.py').write_text(Path(__file__).read_text())
    result = dict(completed=False, rows=[], new_ppo_steps=0, new_physical_episodes=0,
        goal_achieved=False, deployment_approved=False)
    x = torch.from_numpy(observations)
    initial_digest = None
    for name, focus in [('mean500', 0.), ('tail500', .5)]:
        folder = output / name
        folder.mkdir()
        actor = GuardReasonReleaseActor(torch.jit.load(str(model / 'parent_actor.pt'), map_location='cpu')).eval()
        actor.load_state_dict(saved['actor'])
        current = state_digest(actor)
        if initial_digest is None:
            initial_digest = current
        assert current == initial_digest
        optimizer = torch.optim.Adam(actor.head.parameters(), lr=.001)
        optimizer.load_state_dict(saved['optimizer'])
        torch.set_rng_state(saved['torch_rng'])
        np.random.set_state(saved['numpy_rng']); random.setstate(saved['python_rng'])
        frozen = state_digest(actor.parent)
        with torch.no_grad():
            all_features, _ = actor.features(x)
            np.testing.assert_array_equal(actor.support(x).numpy(), active)
        features = all_features[active].detach()
        target = torch.from_numpy(labels[active])
        legal = torch.from_numpy(masks[active])
        weights = torch.from_numpy(fixed_weights).to(features.dtype)
        waiting = torch.from_numpy(observations[active, 434] * 5)
        records = []
        budget.tick()
        for update in range(1, 501):
            logits = (actor.head(features) + 2. * features[:, 76:88]).masked_fill(~legal, -1e8)
            ce = F.cross_entropy(logits, target, reduction='none')
            loss = phase_tail_loss(ce, target, waiting, weights, focus_share=focus)
            optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.head.parameters(), 1.)
            optimizer.step()
            if update % 10 == 0:
                budget.tick()
            if update % 25 == 0 or update == 500:
                with torch.no_grad():
                    prediction = actor(x).numpy()
                    probability = (actor.head(features) + 2. * features[:, 76:88]).masked_fill(~legal, -1e8).softmax(1)
                    minimum = float(probability[torch.arange(len(target)), target].min())
                correct = (prediction == ACTIONS[labels]).all(1)
                records.append(dict(new_update=update, cumulative_supervised_updates=2000 + update,
                    loss=float(loss.detach()), correct_source_labels=int(correct.sum()),
                    total_source_labels=len(labels), safe_stop_labels_retained=bool(correct[labels == 0].all()),
                    minimum_active_target_probability=minimum,
                    wrong_source_indices=np.flatnonzero(~correct).tolist()))
                write(output / 'STATUS.json', dict(status='paired_continuation', model=name, **records[-1]))
                if correct.all() and minimum >= .6:
                    break
        assert state_digest(actor.parent) == frozen
        with torch.no_grad():
            prediction = actor(x).numpy()
        correct = (prediction == ACTIONS[labels]).all(1)
        accepted = bool(correct.all())
        shutil.copy2(model / 'parent_actor.pt', folder / 'parent_actor.pt')
        torch.save(dict(actor=actor.state_dict(), optimizer=optimizer.state_dict(),
            torch_rng=torch.get_rng_state(), numpy_rng=np.random.get_state(), python_rng=random.getstate(),
            parent_actor_sha256=digest(folder / 'parent_actor.pt'),
            contract=saved['contract'], head_contract=saved['head_contract'], learning_support=saved['learning_support'],
            supervised_updates=2000 + update, new_supervised_updates=update,
            source_sha256=digest(dataset), new_ppo_steps=0, exact_physical_rollout_resume=False), folder / 'checkpoint.pt')
        if accepted:
            risk = ROOT / 'models/stop_feedback_20261015/long_risk_bc1000_s231016/portable'
            invalid, near, stopped = (observations[:16].copy() for _ in range(3))
            invalid[:, 435] = 0.; near[:, 218:220] = 0.; near[:, 302] = 1.
            stopped[:, 235:244] = stopped[:, 267:269] = 0.
            export(actor, risk, folder / 'portable', np.r_[observations, invalid, near, stopped])
        row = dict(model=name, model_path=str(folder.relative_to(ROOT)),
            new_supervised_updates=update, cumulative_supervised_updates=2000 + update,
            initial_actor_tensor_sha256=initial_digest,
            correct_source_labels=int(correct.sum()), total_source_labels=len(labels),
            wrong_source_indices=np.flatnonzero(~correct).tolist(),
            original2848_all_retained=bool(correct[:2848].all()),
            original3252_all_retained=bool(correct[:3252].all()),
            added476_all_retained=bool(correct[3252:].all()),
            safe_stop_labels_all_retained=bool(correct[labels == 0].all()),
            exported_for_physical_validation=accepted, checkpoint_sha256=digest(folder / 'checkpoint.pt'),
            frozen_parent_tensor_sha256=frozen, optimizer_and_rng_saved=True, records=records,
            source_sha256=digest(dataset), new_ppo_steps=0, deployment_approved=False)
        write(folder / 'RESULTS.json', dict(completed=True, **row))
        result['rows'].append(row)
        budget.tick()
        result['resource'] = budget.statistics()
        write(output / 'RESULTS.json', result)
        del actor, optimizer, all_features, features
    for path, expected in hashes.items():
        assert digest(ROOT / path) == expected, path
    candidates = [row for row in result['rows'] if row['exported_for_physical_validation']]
    result.update(completed=True, at=datetime.now(timezone.utc).isoformat(),
        selected_for_bounded_physical_validation=candidates[0]['model_path'] if candidates else None,
        inputs_verified=len(hashes), resource=budget.statistics())
    write(output / 'RESULTS.json', result)
    write(output / 'STATUS.json', dict(status='complete', selected=result['selected_for_bounded_physical_validation']))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ['model', 'dataset', 'output']:
        parser.add_argument('--' + key, type=Path, required=True)
    args = parser.parse_args()
    try:
        main(args)
    except Exception as error:
        if args.output.is_dir():
            write(args.output / 'FAILURE.json', dict(error=repr(error), automatically_restarted=False))
        raise
