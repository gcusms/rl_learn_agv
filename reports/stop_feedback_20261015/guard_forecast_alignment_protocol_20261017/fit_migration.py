"""Bounded v2 semantic migration, after the exact live physics dependency.

This is supervised preservation before new on-policy learning, not PPO and
not a navigation-success claim. Only the existing recovery head is trainable.
"""
from datetime import datetime
import hashlib
import json
from pathlib import Path
import random
import shutil
import subprocess
import time
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / 'reports/stop_feedback_20261015'
HERE = Path(__file__).resolve().parent
OUT = BASE / 'guard_forecast_migration_20261017'
MODEL = ROOT / 'models/stop_feedback_20261015/guard_aligned_migration_s231017'
SOURCE = ROOT / 'models/stop_feedback_20261015/normal_suite_wait_pair_v2_s231016/wait_continuation'
RISK = ROOT / 'models/stop_feedback_20261015/long_risk_bc1000_s231016/portable'
DEPENDENCY = 'rl-late106-adaptive-lowmem-20261017.service'
INVOCATION = 'b1303e4883054713a35aaa59e2cf56c5'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    paths = [Path(__file__), HERE/'aligned_demonstrations.npz', HERE/'ALIGNED_DEMONSTRATIONS.json',
        HERE/'NATIVE_ALIGNMENT_PARITY.json', SOURCE/'checkpoint.pt', SOURCE/'parent_actor.pt', SOURCE/'RESULTS.json',
        BASE/'late106_adaptive_lowmem_20261017/PLAN.json']
    for bundle in [SOURCE/'portable', RISK]:
        manifest = json.loads((bundle/'manifest.json').read_text())
        paths += [bundle/'manifest.json'] + [bundle/name for name in manifest['files']]
    for folder in ['neural_navigation', 'experiments', 'navrl']:
        paths += list((ROOT/'training'/folder).rglob('*.py'))
    hashes = {str(p.relative_to(ROOT)): digest(p) for p in set(paths)}
    write(OUT/'PLAN.json', dict(at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(), inputs=hashes,
        dependency=DEPENDENCY, exact_invocation=INVOCATION, seed=231017, maximum_new_supervised_updates=800,
        minimum_new_supervised_updates=20, learning_rate=.0002, loss='masked full-batch cross entropy',
        all_training_labels_required=True, only_changed_factor='v2 forecast semantics and head migration',
        unchanged_old_success_labels=True, no_heldout_or_failed_labels=True,
        optimizer_fresh_for_new_semantic_version=True,
        frozen_components=['parent_actor', 'localizer', 'router', 'directional_docking'],
        original_hardware_action_masks_and_stop_permission=True,
        maximum_output_mebibytes=30, soft_cpu_cores=.35, memory_hard_max_mib=3072,
        new_ppo_steps=0, goal_achieved=False, deployment_approved=False))
    while True:
        checked = subprocess.run(['systemctl', '--user', 'show', DEPENDENCY,
            '--property=ActiveState,MainPID,InvocationID,ExecMainStatus'], capture_output=True, text=True, check=True)
        state = dict(line.split('=', 1) for line in checked.stdout.splitlines() if '=' in line)
        if state['InvocationID'] not in ('', INVOCATION):
            raise RuntimeError('Pinned physical dependency changed invocation')
        if state['MainPID'] == '0':
            completed = BASE/'late106_adaptive_lowmem_20261017/RESULTS.json'
            if (state['ActiveState'] != 'inactive' or state['ExecMainStatus'] != '0'
                    or not completed.exists() or not json.loads(completed.read_text())['completed']):
                raise RuntimeError('Physical dependency did not finish normally; migration not started')
            hashes[str(completed.relative_to(ROOT))] = digest(completed)
            break
        write(OUT/'STATUS.json', dict(status='waiting_for_exact_physical_dependency', dependency=state,
            neural_libraries_not_loaded=True, new_optimizer_updates=0, new_ppo_steps=0))
        time.sleep(5.)
    for name, expected in hashes.items():
        if digest(ROOT/name) != expected:
            raise RuntimeError('Pinned input changed: ' + name)
    write(OUT/'DEPENDENCIES.json', dict(inputs=hashes))

    import numpy as np
    import torch
    import torch.nn.functional as F
    from experiments.guard_reason_release_actor import GuardReasonReleaseActor, contract as head_contract, export
    from experiments.context_release_actor import contract as support_contract
    from experiments.train_long_risk_actor import state_digest
    from neural_navigation.aligned_release_runtime import AlignedReleaseRuntime, RUNTIME_NAME, contract
    from neural_navigation.aligned_risk_observation import contract as risk_contract
    from neural_navigation.experiment_budget import ExperimentBudget
    from neural_navigation.goal_pose_runtime import check_files
    from neural_navigation.masked_actions import ACTIONS, masks_from_observation

    torch.set_num_threads(1)
    torch.manual_seed(231017)
    np.random.seed(231017)
    random.seed(231017)
    budget = ExperimentBudget(.35, memory_mib=2500, directory=OUT)
    admission = json.loads((HERE/'ALIGNED_DEMONSTRATIONS.json').read_text())
    parity = json.loads((HERE/'NATIVE_ALIGNMENT_PARITY.json').read_text())
    assert admission['data_preflight_passed'] and admission['admitted_training_rows'] == 3799
    assert parity['completed'] and parity['entire481_prefix_byte_exact'] and parity['compared_training_rows'] == 3799
    assert digest(HERE/'aligned_demonstrations.npz') == admission['archive_sha256']
    with np.load(HERE/'aligned_demonstrations.npz', allow_pickle=False) as data:
        arrays = {key: data[key].copy() for key in data.files}
    x = torch.from_numpy(arrays['observations'])
    assert x.shape == (3799, 625) and torch.isfinite(x).all()
    np.testing.assert_array_equal(x[:, :439], arrays['original_observations'][:, :439])
    np.testing.assert_array_equal(x[:, 481:], arrays['original_observations'][:, 481:])
    np.testing.assert_array_equal(masks_from_observation(x[:, :282].numpy()), arrays['masks'])
    assert arrays['masks'][np.arange(len(x)), arrays['actions']].all()
    assert digest(SOURCE/'parent_actor.pt') == digest(RISK/'parent_actor.pt')
    assert digest(SOURCE/'portable/hardware.json') == digest(RISK/'hardware.json')

    actor = GuardReasonReleaseActor(torch.jit.load(str(SOURCE/'parent_actor.pt'), map_location='cpu')).eval()
    saved = torch.load(SOURCE/'checkpoint.pt', map_location='cpu', weights_only=False)
    actor.load_state_dict(saved['actor'])
    frozen_parent = state_digest(actor.parent)
    optimizer = torch.optim.Adam(actor.head.parameters(), lr=.0002)
    target = torch.from_numpy(arrays['actions'].astype(np.int64))
    legal = torch.from_numpy(arrays['masks'])
    with torch.no_grad():
        features, baseline = actor.features(x)
        support = actor.support(x)
    nominal = abs(baseline[:, None] - actor.actions[None]).amax(2).argmin(1)
    assert not ((~support) & (nominal != target)).any(), 'Frozen parent labels must remain exact'
    assert torch.equal(actor.masks(x), legal)
    features = features[support].detach()
    labels, mask = target[support], legal[support]

    @torch.no_grad()
    def metrics(update):
        logits = (actor.head(features) + 2.*features[:, 76:88]).masked_fill(~mask, -1e8)
        correct = actor(x).eq(actor.actions[target]).all(1)
        alternatives = logits.clone()
        alternatives.scatter_(1, labels[:, None], -1e8)
        margin = logits.gather(1, labels[:, None]).flatten() - alternatives.amax(1)
        return dict(update=update, correct_labels=int(correct.sum()), total_labels=len(x),
            stop_correct=int(correct[target == 0].sum()), stop_total=int((target == 0).sum()),
            outside_support_exact=bool(correct[~support].all()), minimum_label_logit_margin=float(margin.min()))

    records = [metrics(0)]
    write(OUT/'STATUS.json', dict(status='supervised_v2_semantic_migration', **records[-1]))
    budget.tick()
    for update in range(1, 801):
        logits = (actor.head(features) + 2.*features[:, 76:88]).masked_fill(~mask, -1e8)
        loss = F.cross_entropy(logits, labels)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(actor.head.parameters(), 1.)
        optimizer.step()
        if update % 5 == 0:
            budget.tick()
        if update % 20 == 0:
            row = metrics(update)
            row['loss'] = float(loss.detach())
            records.append(row)
            write(OUT/'STATUS.json', dict(status='supervised_v2_semantic_migration', **row))
            if row['correct_labels'] == len(x) and row['minimum_label_logit_margin'] >= .01:
                break
    final = metrics(update)
    passed = final['correct_labels'] == len(x) and final['outside_support_exact']
    assert state_digest(actor.parent) == frozen_parent
    MODEL.mkdir(parents=True, exist_ok=False)
    shutil.copy2(SOURCE/'parent_actor.pt', MODEL/'parent_actor.pt')
    torch.save(dict(actor=actor.state_dict(), optimizer=optimizer.state_dict(),
        torch_rng=torch.get_rng_state(), numpy_rng=np.random.get_state(), python_rng=random.getstate(),
        contract=contract(), head_contract=head_contract(), learning_support=support_contract(),
        prior_supervised_updates=saved['supervised_updates'], new_supervised_updates=update,
        source_sha256=digest(HERE/'aligned_demonstrations.npz'),
        parent_checkpoint_sha256=digest(SOURCE/'checkpoint.pt'), parent_actor_sha256=digest(MODEL/'parent_actor.pt'),
        new_ppo_steps=0, exact_physical_rollout_resume=False, deployment_approved=False), MODEL/'checkpoint.pt')
    if passed:
        invalid, near, stop = [arrays['observations'][:24].copy() for _ in range(3)]
        invalid[:, 435] = 0.
        near[:, 218:220] = 0.
        stop[:, 235:244] = stop[:, 267:269] = 0.
        checks = np.concatenate([arrays['observations'], invalid, near, stop])
        export(actor, RISK, MODEL/'portable', checks)
        bundle = MODEL/'portable'
        manifest = json.loads((bundle/'manifest.json').read_text())
        for name, value in [('release_observation.json', contract()), ('long_risk_observation.json', risk_contract())]:
            write(bundle/name, value)
            manifest['files'][name] = digest(bundle/name)
        preservation = json.loads((bundle/'navigation_preservation.json').read_text())
        preservation.update(schema='guard_aligned_release_head_v2', long_risk_semantics=risk_contract(),
            input_semantics_changed_requires_matching_v2=True, unchanged_runtime_prefix_dimensions=439,
            unchanged_native_suffix=[481, 625], source_model=str(SOURCE.relative_to(ROOT)),
            new_supervised_migration_updates=update, new_ppo_steps=0)
        write(bundle/'navigation_preservation.json', preservation)
        manifest.update(required_runtime=RUNTIME_NAME, observation_contract=contract(),
            navigation_preservation_contract=preservation, additional_ppo_steps=0, deployment_approved=False)
        manifest['files']['navigation_preservation.json'] = digest(bundle/'navigation_preservation.json')
        write(bundle/'manifest.json', manifest)
        check_files(bundle, manifest)
        portable = torch.jit.load(str(bundle/'actor.pt'), map_location='cpu').eval()
        with torch.no_grad():
            np.testing.assert_array_equal(portable(torch.from_numpy(checks)).numpy(), actor(torch.from_numpy(checks)).numpy())
        for name in ['hardware.json', 'localizer.pt', 'router.pt', 'parent_actor.pt', 'docking_contract.json']:
            assert digest(bundle/name) == digest(SOURCE/'portable'/name), name
        # Constructor admission checks the complete new bundle, not just JSON.
        runtime = AlignedReleaseRuntime(bundle, ROOT/'training/assets/room_navigation_scripted/decision_map.yaml',
            [6.225, 2.725, .01737695], [7.125, 5.725], 'shortcut')
        assert runtime.policy.policy.observation_dimensions == 625
        del runtime
        write(bundle/'MODEL_CARD.json', dict(method='supervised v2 input-semantic preservation before PPO',
            new_supervised_updates=update, new_ppo_steps=0, source_training_rows=len(x),
            source_fit_passed=True, export_parity_passed=True, explicit_v2_runtime_load_passed=True,
            no_new_physical_navigation_result=True, goal_achieved=False, deployment_approved=False,
            manifest_sha256=digest(bundle/'manifest.json')))
    budget.tick()
    for name, expected in hashes.items():
        assert digest(ROOT/name) == expected, name
    result = dict(completed=True, model_path=str(MODEL.relative_to(ROOT)),
        new_supervised_updates=update, new_ppo_steps=0, all_source_labels_fit=passed,
        source_training_rows=len(x), trainable_rows=int(support.sum()),
        frozen_components_unchanged=True, optimizer_and_rng_saved=True, final=final, records=records,
        exported_for_bounded_physical_validation=passed, exported_for_physical_validation=passed,
        checkpoint_sha256=digest(MODEL/'checkpoint.pt'), source_sha256=digest(HERE/'aligned_demonstrations.npz'),
        source_semantic_version=risk_contract()['schema'], new_physical_episodes=0,
        inputs_verified=len(hashes), at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
        resource=budget.statistics(), goal_achieved=False, deployment_approved=False)
    write(MODEL/'RESULTS.json', result)
    write(OUT/'RESULTS.json', result)
    write(OUT/'STATUS.json', dict(status='complete', all_source_labels_fit=passed, new_ppo_steps=0))


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        if OUT.exists():
            write(OUT/'FAILURE.json', dict(error=repr(error), automatically_restarted=False))
        raise
