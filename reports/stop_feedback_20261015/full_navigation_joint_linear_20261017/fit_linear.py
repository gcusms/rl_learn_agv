"""Constrained final-layer fit of the admitted5608joint references.

Freeze the full representation and parent. Solve for the smallest L1 change
that separates every allowed target with the same positive margin. No solver
iterate is applied to the actor; only a fully verified solution can be saved.
"""
import copy
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
BASE = HERE.parent
DATA = BASE / 'full_navigation_joint_retention_20261017'
SOURCE = ROOT / 'models/stop_feedback_20261015/full_navigation_margin_resume_s231017'
MODEL = ROOT / 'models/stop_feedback_20261015/full_navigation_joint_linear_s231017'


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
    admission = read(DATA / 'ADMISSION.json')
    assert admission['completed'] and admission['admitted'] and admission['total_rows'] == 5608
    archive = DATA / 'training_data.npz'
    assert digest(archive) == admission['archive_sha256']
    rejected = read(DATA / 'FIT_RESULTS.json')
    assert rejected['completed'] and not rejected['fit_passed'] and not rejected['export_parity_passed']
    paths = [Path(__file__), HERE / 'actor.py', DATA / 'ADMISSION.json', DATA / 'FIT_RESULTS.json',
        archive, SOURCE / 'checkpoint.pt', SOURCE / 'parent625_actor.pt', SOURCE / 'portable/manifest.json']
    manifest = read(SOURCE / 'portable/manifest.json')
    paths += [SOURCE / 'portable' / name for name in manifest['files']]
    hashes = {str(path.relative_to(ROOT)):digest(path) for path in set(paths)}
    write(HERE / 'FIT_PLAN.json', dict(at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(), inputs=hashes,
        source_model=str(SOURCE.relative_to(ROOT)), model=str(MODEL.relative_to(ROOT)),
        one_factor='Constrained final-layer solution replaces rejected unconstrained gradient update',
        data_unchanged_from_rejected_joint_fit=True, all5608_targets_required=True,
        original692_stop_labels_and658subset_required=True, frozen_parent_and_first2layers=True,
        objective='Minimum L1 final-weight-and-bias change subject to every allowed target margin >=0.05',
        solver='scipy.optimize.linprog HiGHS single-thread', solver_time_limit_seconds=45.,
        no_solver_iterates_applied_to_actor=True, maximum_accepted_updates=1,
        original625_inputs_12actions_body4cm_guard_memory_localizer_router_terminal_deadlines=True,
        all_known6_now_training_development=True, original_full28_known6_hard3_conditional8pairs_required=True,
        new_ppo_steps=0, soft_cpu_cores=.35, hard_service_memory_mib=3072, rss_budget_mib=2800,
        goal_achieved=False, deployment_approved=False))

    import numpy as np
    import torch
    from scipy.optimize import linprog
    from scipy.sparse import coo_matrix, eye, hstack, vstack
    from actor import FullNavigationResidualActor, contract
    from neural_navigation.experiment_budget import ExperimentBudget
    from neural_navigation.aligned_release_runtime import AlignedReleaseRuntime
    from neural_navigation.goal_pose_runtime import check_files
    torch.set_num_threads(1)
    budget = ExperimentBudget(.35, 2800, HERE)
    with np.load(archive, allow_pickle=False) as data:
        x = torch.from_numpy(data['observations'].copy())
        target = torch.from_numpy(data['actions'].copy())
        legal = torch.from_numpy(data['masks'].copy())
    actor = FullNavigationResidualActor(torch.jit.load(str(SOURCE / 'parent625_actor.pt'), map_location='cpu')).eval()
    checkpoint = torch.load(SOURCE / 'checkpoint.pt', map_location='cpu', weights_only=False)
    assert checkpoint['cumulative_supervised_updates'] == 2500
    actor.load_state_dict(checkpoint['actor'])
    frozen_state = {name:value.detach().clone() for name,value in actor.state_dict().items()
        if not name.startswith('head.4.')}
    del checkpoint
    features, baseline, support = [], [], []
    with torch.no_grad():
        for start in range(0, len(x), 32):
            f, b = actor.features(x[start:start + 32])
            features.append(f)
            baseline.append(b)
            support.append(actor.support(x[start:start + 32]))
            budget.tick()
    features, baseline, support = torch.cat(features), torch.cat(baseline), torch.cat(support)
    nominal = (baseline[:, None] - actor.actions[None]).abs().amax(2).argmin(1)
    assert torch.equal(actor.masks(x), legal)
    assert not ((~support) & (nominal != target)).any()
    with torch.no_grad():
        hidden = actor.head[:4](features).numpy().astype(np.float64)
        logits = (actor.head(features) + 2. * features[:, 76:88]).numpy().astype(np.float64)
    vectors = np.c_[hidden, np.ones(len(x))]
    parameter_count = 12 * 33
    valid = legal.numpy() & support.numpy()[:, None]
    valid[np.arange(len(x)), target.numpy()] = False
    samples, alternatives = np.nonzero(valid)
    wanted = target.numpy()[samples]
    nr = len(samples)
    constraint_rows = np.repeat(np.arange(nr), 66)
    constraint_columns = np.c_[alternatives[:, None] * 33 + np.arange(33),
        wanted[:, None] * 33 + np.arange(33)].reshape(-1)
    values = np.c_[vectors[samples], -vectors[samples]].reshape(-1)
    matrix = coo_matrix((values, (constraint_rows, constraint_columns)),
        shape=(nr, parameter_count)).tocsr()
    upper = logits[samples, wanted] - logits[samples, alternatives] - .05
    identity = eye(parameter_count, format='csr')
    inequalities = vstack([hstack([matrix, coo_matrix((nr, parameter_count))]),
        hstack([identity, -identity]), hstack([-identity, -identity])], format='csr')
    bounds = np.r_[upper, np.zeros(2 * parameter_count)]
    objective = np.r_[np.zeros(parameter_count), np.ones(parameter_count)]
    write(HERE / 'FIT_STATUS.json', dict(status='constrained_linear_solve', target_rows=len(x),
        pairwise_margin_constraints=nr, linear_parameters=parameter_count, variables=2 * parameter_count,
        source_labels_correct=int((np.where(support.numpy(), np.where(legal.numpy(), logits, -1e8).argmax(1), nominal.numpy()) == target.numpy()).sum())))
    budget.tick()
    solution = linprog(objective, A_ub=inequalities, b_ub=bounds,
        bounds=[(None, None)] * parameter_count + [(0., None)] * parameter_count,
        method='highs', options=dict(time_limit=45., threads=1, primal_feasibility_tolerance=1e-8,
            dual_feasibility_tolerance=1e-8))
    budget.tick()
    solver = dict(success=bool(solution.success), status=int(solution.status), message=solution.message,
        iterations=int(solution.nit), pairwise_constraints=nr,
        l1_parameter_change=float(solution.fun) if solution.fun is not None else None)
    if not solution.success:
        write(HERE / 'FIT_RESULTS.json', dict(completed=True, fit_passed=False, solver=solver,
            no_actor_update_or_export=True, resource=budget.statistics(), goal_achieved=False, deployment_approved=False))
        return
    delta = solution.x[:parameter_count].reshape(12, 33)
    # One candidate update, then evaluate every original and new reference.
    with torch.no_grad():
        actor.head[4].weight.add_(torch.from_numpy(delta[:, :32]).to(torch.float32))
        actor.head[4].bias.add_(torch.from_numpy(delta[:, 32]).to(torch.float32))
        adjusted = (actor.head(features) + 2. * features[:, 76:88]).masked_fill(~legal, -1e8)
        chosen = torch.where(support, adjusted.argmax(1), nominal)
        correct = chosen == target
        competitors = adjusted.clone().scatter_(1, target[:, None], -1e8).amax(1)
        margin = adjusted.gather(1, target[:, None]).flatten() - competitors
    fit_passed = bool(correct.all() and (margin[support] > .049).all())
    metrics = dict(correct_labels=int(correct.sum()), total_labels=len(x), old4468_correct=int(correct[:4468].sum()),
        new1140_correct=int(correct[4468:].sum()), original692_stop_correct=int(correct[:4468][target[:4468] == 0].sum()),
        original658_stop_correct=int(correct[:4282][target[:4282] == 0].sum()),
        minimum_supported_margin=float(margin[support].min()), maximum_parameter_change=float(abs(delta).max()))
    assert metrics['original692_stop_correct'] == 692 and metrics['original658_stop_correct'] == 658
    for name, expected in frozen_state.items():
        assert torch.equal(actor.state_dict()[name], expected), name
    if not fit_passed:
        write(HERE / 'FIT_RESULTS.json', dict(completed=True, fit_passed=False, solver=solver, final=metrics,
            no_export=True, resource=budget.statistics(), goal_achieved=False, deployment_approved=False))
        return
    MODEL.mkdir(parents=True, exist_ok=False)
    shutil.copy2(SOURCE / 'parent625_actor.pt', MODEL / 'parent625_actor.pt')
    shutil.copy2(HERE / 'actor.py', MODEL / 'training_actor.py')
    torch.save(dict(actor=actor.state_dict(), torch_rng=torch.get_rng_state(), actor_contract=contract(),
        training_parent_file='parent625_actor.pt', parent_sha256=digest(SOURCE / 'parent625_actor.pt'),
        source_checkpoint_sha256=digest(SOURCE / 'checkpoint.pt'), data_sha256=admission['archive_sha256'],
        source_supervised_updates=2500, constrained_final_layer_updates=1,
        cumulative_supervised_updates=2501, new_supervised_updates=1, new_ppo_steps=0,
        optimizer_type='constrained_linear_program', solver=solver, solution_delta=delta,
        all_known6_now_training_development=True, deployment_approved=False), MODEL / 'checkpoint.pt')
    bundle = MODEL / 'portable'
    shutil.copytree(SOURCE / 'portable', bundle)
    sample = x[::max(1, len(x) // 64)][:64]
    traced = torch.jit.trace_module(actor, {'forward':(sample,), 'logits':(sample,), 'support':(sample,)})
    traced.save(str(bundle / 'actor.pt'))
    with torch.no_grad():
        for start in range(0, len(x), 32):
            np.testing.assert_array_equal(traced(x[start:start + 32]).numpy(), actor.actions[target[start:start + 32]].numpy())
            budget.tick()
    preservation = read(bundle / 'navigation_preservation.json')
    preservation.update(schema='full_navigation_joint_linear_retention_v1', all5608labels_retained=True,
        original692stops_and658subset_retained=True, frozen_representation=True,
        constrained_final_layer_updates=1, source_supervised_updates=2500,
        all_known6_now_training_development=True, physical_validation_not_yet_passed=True)
    write(bundle / 'navigation_preservation.json', preservation)
    new_manifest = copy.deepcopy(manifest)
    new_manifest.update(deployment_approved=False, navigation_preservation_contract=preservation,
        parent_manifest_sha256=digest(SOURCE / 'portable/manifest.json'))
    for name in ('actor.pt', 'navigation_preservation.json'):
        new_manifest['files'][name] = digest(bundle / name)
    write(bundle / 'manifest.json', new_manifest)
    write(bundle / 'MODEL_CARD.json', dict(method='minimum_L1_constrained_last_layer_fit',
        source_supervised_updates=2500, new_constrained_updates=1, new_ppo_steps=0,
        all_known6_now_training_development=True, deployment_approved=False))
    check_files(bundle, new_manifest)
    for name, expected in manifest['files'].items():
        if name not in ('actor.pt', 'navigation_preservation.json'):
            assert digest(bundle / name) == expected, name
    runtime = AlignedReleaseRuntime(bundle, ROOT / 'training/assets/room_navigation_scripted/decision_map.yaml',
        [6.225, 2.725, 0.], [7.125, 5.725], 'shortcut')
    assert runtime.policy.policy.observation_dimensions == 625
    del runtime
    for name, expected in hashes.items():
        assert digest(ROOT / name) == expected, name
    budget.tick()
    result = dict(completed=True, fit_passed=True, at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
        solver=solver, final=metrics, export_parity_passed=True, strict_runtime_admission_passed=True,
        accepted_supervised_updates=1, constrained_final_layer_updates=1, source_supervised_updates=2500,
        cumulative_supervised_updates=2501, model=str(MODEL.relative_to(ROOT)), checkpoint_sha256=digest(MODEL / 'checkpoint.pt'),
        all_known6_now_training_development=True, frozen_parent_representation_and_all_other_components=True,
        new_ppo_steps=0, new_physical_episodes=0, resource=budget.statistics(), goal_achieved=False, deployment_approved=False)
    write(HERE / 'FIT_RESULTS.json', result)
    write(MODEL / 'RESULTS.json', result)
    write(HERE / 'FIT_STATUS.json', dict(status='complete', fit_passed=True))
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        write(HERE / 'FIT_FAILURE.json', dict(error=repr(error), no_auto_restart=True))
        raise
