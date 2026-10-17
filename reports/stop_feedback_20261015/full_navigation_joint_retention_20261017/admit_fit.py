"""Fit known complete-mission retention without weakening existing protection.

All4468old references and692old STOP labels remain constraints. The additional
five safe aux119 missions are now explicitly training/development data.
"""
import copy
from datetime import datetime
import hashlib
import json
from pathlib import Path
import random
import shutil
import sys
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
BASE = HERE.parent
PRIOR = BASE / 'full_navigation_residual_20261017'
SOURCE = ROOT / 'models/stop_feedback_20261015/full_navigation_margin_resume_s231017'
PARENT = ROOT / 'models/stop_feedback_20261015/v2_behavior_retention_s231017/portable'
MODEL = ROOT / 'models/stop_feedback_20261015/full_navigation_joint_retention_s231017'


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temp.replace(path)


def main():
    assert not MODEL.exists() and not (HERE / 'FIT_RESULTS.json').exists()
    regression = read(HERE / 'REGRESSION_AUDIT.json')
    assert regression['completed'] and regression['trainable_voluntary_wait_frames'] == 569
    previous_validation = BASE / 'v2_behavior_retention_validation_20261017/RESULTS.json'
    previous = read(previous_validation)
    known = previous['groups']['observed6']['rows']
    rows = [row for row in known if row['safe_success']]
    assert len(rows) == 5 and {r['case_id'] for r in rows} == {
        'pre_ppo_holdout_dynamic_00', 'pre_ppo_holdout_dynamic_01', 'pre_ppo_holdout_dynamic_02',
        'pre_ppo_holdout_dynamic_03', 'pre_ppo_holdout_sensor_00'}
    archive = PRIOR / 'training_data_terminal.npz'
    assert digest(archive) == read(PRIOR / 'ADMISSION_TERMINAL.json')['archive_sha256']
    assert digest(SOURCE / 'checkpoint.pt') == read(PRIOR / 'MARGIN_RESUME_FIT_RESULTS.json')['checkpoint_sha256']
    manifest = read(PARENT / 'manifest.json')
    protected = [name for name in manifest['files'] if name not in ('actor.pt', 'navigation_preservation.json')]
    for name in protected:
        assert digest(PARENT / name) == digest(SOURCE / 'portable' / name), name
    paths = [Path(__file__), HERE / 'actor.py', HERE / 'REGRESSION_AUDIT.json', previous_validation,
        archive, SOURCE / 'checkpoint.pt', SOURCE / 'parent625_actor.pt', SOURCE / 'portable/manifest.json']
    paths += [SOURCE / 'portable' / name for name in read(SOURCE / 'portable/manifest.json')['files']]
    paths += [ROOT / row['trace_path'] for row in rows]
    hashes = {str(path.relative_to(ROOT)):digest(path) for path in set(paths)}
    write(HERE / 'FIT_PLAN.json', dict(at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(), inputs=hashes,
        source_model=str(SOURCE.relative_to(ROOT)), model=str(MODEL.relative_to(ROOT)), seed=231018,
        one_main_factor='Joint complete-mission retention data after actual observed regression',
        old_reference_rows=4468, original692_stop_labels_checked_after_every_update=True,
        original658_stop_subset_retained=True, known5_successes_now_training_development=True,
        failed_trajectory_not_used_as_a_positive_demonstration=True,
        maximum_accepted_updates=4000, maximum_proposed_updates=6000, learning_rate=.00015,
        optimizer_reinitialized_for_changed_dataset=True, source_supervised_updates=2500,
        loss='case-balanced new mission CE plus separate original4282 and safe sensor01 margin hinges',
        old_margin=.2, old_hinge_weight_per_source=4., require_all_reference_labels_before_physics=True,
        original625_observation_12actions_4cm_body_guard_memory_localizer_router_terminal_and_deadlines=True,
        original_fixed28_known_safety_hard97_6_106_conditional_fresh8_preserved=True,
        soft_cpu_cores=.35, hard_service_memory_mib=3072, rss_budget_mib=2800,
        new_ppo_steps=0, no_automatic_promotion=True, goal_achieved=False, deployment_approved=False))

    import numpy as np
    import torch
    import torch.nn.functional as F
    from actor import FullNavigationResidualActor, contract
    from neural_navigation.experiment_budget import ExperimentBudget
    from neural_navigation.aligned_release_runtime import AlignedReleaseRuntime
    from neural_navigation.goal_pose_runtime import check_files
    from neural_navigation.masked_actions import COMMANDS, masks_from_observation
    torch.set_num_threads(1)
    torch.manual_seed(231018)
    np.random.seed(231018)
    random.seed(231018)
    budget = ExperimentBudget(.35, 2800, HERE)
    with np.load(archive, allow_pickle=False) as data:
        observations = [data['observations'].copy()]
        actions = [data['actions'].copy()]
        masks = [data['masks'].copy()]
        parent_groups = data['source_group'].copy()
    assert len(actions[0]) == 4468 and len(parent_groups) == 4468
    new_groups = [np.full(4468, -1, np.int64)]
    source_rows = []
    for index, row in enumerate(rows):
        path = ROOT / row['trace_path']
        assert digest(path) == row['trace_sha256']
        assert row['protocol_valid'] and not any(row[key] for key in ('collisions', 'false_arrival', 'physical_guards', 'timeout'))
        with np.load(path, allow_pickle=False) as data:
            assert data['observations'].shape == (row['frames'], 625)
            assert data['labels'][-1, 7] < .2
            np.testing.assert_allclose(data['commands'][:-1], COMMANDS[data['choices'][:-1]], rtol=0, atol=2e-8)
            observations.append(data['observations'].copy())
            actions.append(data['choices'].astype(np.int64))
            masks.append(data['masks'].copy())
            new_groups.append(np.full(len(data['choices']), index, np.int64))
            source_rows.append(dict(case_id=row['case_id'], rows=len(data['choices']),
                trace_path=row['trace_path'], trace_sha256=row['trace_sha256'], now_training_development=True))
    observations = np.concatenate(observations).astype(np.float32)
    actions = np.concatenate(actions).astype(np.int64)
    masks = np.concatenate(masks).astype(bool)
    case_index = np.concatenate(new_groups)
    np.testing.assert_array_equal(masks, masks_from_observation(observations[:, :282]))
    assert masks[np.arange(len(actions)), actions].all()
    labels, conflicts = {}, []
    for i, (x, target) in enumerate(zip(observations, actions)):
        prior = labels.setdefault(x.tobytes(), (int(target), i))
        if prior[0] != int(target):
            conflicts.append([prior[1], i])
    assert not conflicts, conflicts[:10]
    x = torch.from_numpy(observations)
    target = torch.from_numpy(actions)
    legal = torch.from_numpy(masks)
    actor = FullNavigationResidualActor(torch.jit.load(str(SOURCE / 'parent625_actor.pt'), map_location='cpu')).eval()
    checkpoint = torch.load(SOURCE / 'checkpoint.pt', map_location='cpu', weights_only=False)
    assert checkpoint['cumulative_supervised_updates'] == 2500
    assert checkpoint['parent_sha256'] == digest(SOURCE / 'parent625_actor.pt')
    actor.load_state_dict(checkpoint['actor'])
    del checkpoint
    parent_state = {name:value.detach().clone() for name,value in actor.parent.state_dict().items()}
    features, baselines, supports = [], [], []
    with torch.no_grad():
        for start in range(0, len(x), 32):
            f, b = actor.features(x[start:start + 32])
            features.append(f)
            baselines.append(b)
            supports.append(actor.support(x[start:start + 32]))
            budget.tick()
    features, baseline, support = torch.cat(features), torch.cat(baselines), torch.cat(supports)
    nominal = (baseline[:, None] - actor.actions[None]).abs().amax(2).argmin(1)
    assert torch.equal(actor.masks(x), legal)
    assert not ((~support) & (nominal != target)).any(), 'New labels outside unchanged learning support'
    features_seen, feature_conflicts = {}, []
    for i in torch.where(support)[0].tolist():
        prior = features_seen.setdefault(features[i].numpy().tobytes(), (int(target[i]), i))
        if prior[0] != int(target[i]):
            feature_conflicts.append([prior[1], i])
    assert not feature_conflicts, feature_conflicts[:10]
    stop_anchor = (torch.arange(len(x)) < 4468) & (target == 0)
    assert int(stop_anchor.sum()) == 692 and int((target[:4282] == 0).sum()) == 658
    old_sets = [torch.from_numpy(np.r_[parent_groups == group, np.zeros(len(x) - 4468, bool)]) & support for group in (0, 1)]
    new_sets = [torch.from_numpy(case_index == i) & support for i in range(5)]
    assert all(bool(group.any()) for group in old_sets + new_sets)
    data_path = HERE / 'training_data.npz'
    np.savez_compressed(data_path, observations=observations, actions=actions, masks=masks,
        case_index=case_index, parent_source_group=np.r_[parent_groups, np.full(len(x) - 4468, -1, np.int64)])
    admission = dict(completed=True, admitted=True, at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
        total_rows=len(x), old_reference_rows=4468, additional_known_mission_rows=len(x) - 4468,
        exact625_conflicts=conflicts, exact408_supported_feature_conflicts=feature_conflicts,
        outside_support_parent_exact=True, all_original_action_masks_valid=True,
        sources=source_rows, all_known6_now_training_development=True,
        archive_sha256=digest(data_path), original_protected_stop_rows=692,
        original658_stop_subset_retained=True, frozen_components_identical=True,
        new_physical_episodes=0, new_ppo_steps=0, goal_achieved=False, deployment_approved=False)
    write(HERE / 'ADMISSION.json', admission)

    @torch.no_grad()
    def metrics():
        logits = (actor.head(features) + 2. * features[:, 76:88]).masked_fill(~legal, -1e8)
        selected = torch.where(support, logits.argmax(1), nominal)
        correct = selected == target
        alternate = logits.clone().scatter_(1, target[:, None], -1e8).amax(1)
        margin = logits.gather(1, target[:, None]).flatten() - alternate
        return dict(correct_labels=int(correct.sum()), total_labels=len(x), old4468_correct=int(correct[:4468].sum()),
            known_mission_correct=int(correct[4468:].sum()), protected_stop_correct=int(correct[stop_anchor].sum()),
            protected_stop_total=692, all_stop_correct=int(correct[target == 0].sum()), all_stop_total=int((target == 0).sum()),
            minimum_supported_margin=float(margin[support].min()))

    @torch.no_grad()
    def retained():
        logits = (actor.head(features[stop_anchor]) + 2. * features[stop_anchor, 76:88]).masked_fill(~legal[stop_anchor], -1e8)
        return bool((torch.where(support[stop_anchor], logits.argmax(1), nominal[stop_anchor]) == 0).all())

    initial = metrics()
    assert initial['old4468_correct'] == 4468 and retained()
    optimizer = torch.optim.Adam(actor.head.parameters(), lr=.00015)
    accepted = rejected = 0
    history = [dict(accepted_updates=0, **initial)]
    proposed = 0
    for proposed in range(1, 6001):
        old_head, old_optimizer = copy.deepcopy(actor.head.state_dict()), copy.deepcopy(optimizer.state_dict())
        logits = (actor.head(features) + 2. * features[:, 76:88]).masked_fill(~legal, -1e8)
        new_loss = torch.stack([F.cross_entropy(logits[indices], target[indices]) for indices in new_sets]).mean()
        alternatives = logits.scatter(1, target[:, None], -1e8).amax(1)
        margins = logits.gather(1, target[:, None]).flatten() - alternatives
        old_loss = sum(F.relu(.2 - margins[indices]).mean() for indices in old_sets)
        loss = new_loss + 4. * old_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(actor.head.parameters(), 1.)
        optimizer.step()
        if not retained():
            actor.head.load_state_dict(old_head)
            optimizer.load_state_dict(old_optimizer)
            for group in optimizer.param_groups:
                group['lr'] *= .5
            rejected += 1
            if optimizer.param_groups[0]['lr'] < 1e-8:
                break
        else:
            accepted += 1
        if proposed % 5 == 0:
            budget.tick()
        if proposed % 20 == 0 or accepted >= 4000:
            row = dict(accepted_updates=accepted, proposed_updates=proposed, rejected_updates=rejected,
                loss=float(loss.detach()), **metrics())
            history.append(row)
            write(HERE / 'FIT_STATUS.json', dict(status='joint_retention_fit', **row))
            if row['correct_labels'] == len(x) and row['minimum_supported_margin'] >= .01:
                break
        if accepted >= 4000:
            break
    final = metrics()
    passed = final['correct_labels'] == len(x) and final['protected_stop_correct'] == 692
    for name, value in actor.parent.state_dict().items():
        assert torch.equal(value, parent_state[name]), name
    MODEL.mkdir(parents=True, exist_ok=False)
    shutil.copy2(SOURCE / 'parent625_actor.pt', MODEL / 'parent625_actor.pt')
    shutil.copy2(HERE / 'actor.py', MODEL / 'training_actor.py')
    torch.save(dict(actor=actor.state_dict(), optimizer=optimizer.state_dict(), torch_rng=torch.get_rng_state(),
        numpy_rng=np.random.get_state(), python_rng=random.getstate(), actor_contract=contract(),
        training_parent_file='parent625_actor.pt', parent_sha256=digest(SOURCE / 'parent625_actor.pt'),
        source_checkpoint_sha256=digest(SOURCE / 'checkpoint.pt'), data_sha256=admission['archive_sha256'],
        source_supervised_updates=2500, new_supervised_updates=accepted, cumulative_supervised_updates=2500 + accepted,
        optimizer_reinitialized_for_changed_data=True, new_ppo_steps=0, deployment_approved=False), MODEL / 'checkpoint.pt')
    exported = runtime_passed = False
    if passed:
        bundle = MODEL / 'portable'
        shutil.copytree(SOURCE / 'portable', bundle)
        sample = x[::max(1, len(x) // 64)][:64]
        traced = torch.jit.trace_module(actor, {'forward':(sample,), 'logits':(sample,), 'support':(sample,)})
        traced.save(str(bundle / 'actor.pt'))
        with torch.no_grad():
            for start in range(0, len(x), 32):
                np.testing.assert_array_equal(traced(x[start:start + 32]).numpy(), actor.actions[target[start:start + 32]].numpy())
                budget.tick()
        exported = True
        preservation = read(bundle / 'navigation_preservation.json')
        preservation.update(schema='full_navigation_joint_retention_v1', learning_support=contract(),
            all4468_prior_labels_retained=True, all692_prior_stops_retained=True,
            all5_prior_safe_known_missions_now_training=True, new_supervised_updates=accepted,
            source_supervised_updates=2500, new_ppo_steps=0, physical_validation_not_yet_passed=True)
        write(bundle / 'navigation_preservation.json', preservation)
        new_manifest = read(SOURCE / 'portable/manifest.json')
        new_manifest.update(deployment_approved=False, navigation_preservation_contract=preservation,
            parent_manifest_sha256=digest(SOURCE / 'portable/manifest.json'))
        for name in ('actor.pt', 'navigation_preservation.json'):
            new_manifest['files'][name] = digest(bundle / name)
        write(bundle / 'manifest.json', new_manifest)
        write(bundle / 'MODEL_CARD.json', dict(method='full_navigation_joint_retention_supervised_fit',
            source_supervised_updates=2500, new_supervised_updates=accepted, new_ppo_steps=0,
            all_known6_are_training_development=True, deployment_approved=False))
        check_files(bundle, new_manifest)
        for name, expected in read(SOURCE / 'portable/manifest.json')['files'].items():
            if name not in ('actor.pt', 'navigation_preservation.json'):
                assert digest(bundle / name) == expected, name
        runtime = AlignedReleaseRuntime(bundle, ROOT / 'training/assets/room_navigation_scripted/decision_map.yaml',
            [6.225, 2.725, 0.], [7.125, 5.725], 'shortcut')
        assert runtime.policy.policy.observation_dimensions == 625
        runtime_passed = True
        del runtime
        budget.tick()
    for name, expected in hashes.items():
        assert digest(ROOT / name) == expected, name
    result = dict(completed=True, at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(), fit_passed=passed,
        initial=initial, final=final, accepted_supervised_updates=accepted, source_supervised_updates=2500,
        cumulative_supervised_updates=2500 + accepted, proposed_updates=proposed, rejected_updates=rejected,
        history=history, export_parity_passed=exported, strict_runtime_admission_passed=runtime_passed,
        model=str(MODEL.relative_to(ROOT)), checkpoint_sha256=digest(MODEL / 'checkpoint.pt'),
        protected_frozen_components_unchanged=True, all_known6_now_training_development=True,
        new_ppo_steps=0, new_physical_episodes=0, resource=budget.statistics(),
        goal_achieved=False, deployment_approved=False)
    write(HERE / 'FIT_RESULTS.json', result)
    write(MODEL / 'RESULTS.json', result)
    write(HERE / 'FIT_STATUS.json', dict(status='complete', fit_passed=passed))
    print(json.dumps({k:v for k,v in result.items() if k != 'history'}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        write(HERE / 'FIT_FAILURE.json', dict(error=repr(error), no_auto_restart=True))
        raise
