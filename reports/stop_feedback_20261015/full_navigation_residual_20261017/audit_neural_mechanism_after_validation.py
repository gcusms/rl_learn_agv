"""Serialize a read-only neural cause probe after the exact physical validator.

Before the dependency is terminal this process uses only the standard library.
Counterfactual feature probes never issue commands or become training labels.
"""
from datetime import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
BASE = HERE.parent
MODEL = ROOT / 'models/stop_feedback_20261015/full_navigation_margin_resume_s231017'
VALIDATION = BASE / 'full_navigation_margin_resume_validation_20261017'
DEPENDENCY = 'rl-full-navigation-margin-resume-cycle-20261017.service'
INVOCATION = 'ef95cccf487943edb1a993b9732a2520'
OWN_UNIT = 'rl-full-navigation-mechanism-post-20261017.service'


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def status(**value):
    value.update(at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
        goal_achieved=False, deployment_approved=False)
    write(HERE / 'NEURAL_MECHANISM_STATUS.json', value)


def wait_for_dependency():
    while True:
        result = subprocess.run(['systemctl', '--user', 'show', DEPENDENCY,
            '-p', 'MainPID', '-p', 'ActiveState', '-p', 'InvocationID',
            '-p', 'ExecMainStatus'], text=True, capture_output=True, check=True)
        state = dict(line.split('=', 1) for line in result.stdout.splitlines() if '=' in line)
        assert state.get('InvocationID') in ('', INVOCATION), 'Exact dependency was restarted'
        if state['MainPID'] == '0' and state['ActiveState'] in ('inactive', 'failed'):
            assert state['ActiveState'] == 'inactive' and state['ExecMainStatus'] == '0', state
            report = read(VALIDATION / 'RESULTS.json')
            assert report['completed'] and report['model'] == str(MODEL.relative_to(ROOT))
            running = subprocess.run(['systemctl', '--user', 'list-units',
                '--type=service', '--state=running', '--no-legend', '--no-pager', 'rl-*'],
                text=True, capture_output=True, check=True)
            others = [line.strip() for line in running.stdout.splitlines()
                if line.strip() and line.split()[0] != OWN_UNIT]
            if others:
                status(stage='waiting_for_other_rl_services', other_services=others,
                    model_and_physics_not_loaded=True)
                time.sleep(5.)
                continue
            return report, state
        assert 'torch' not in sys.modules and 'numpy' not in sys.modules
        status(stage='waiting_for_exact_physical_validation', dependency=state,
            model_and_physics_not_loaded=True, no_parallel_model_or_physics=True)
        time.sleep(5.)


def main():
    assert not (HERE / 'NEURAL_MECHANISM.json').exists()
    report, dependency = wait_for_dependency()
    snapshot = HERE / 'NEURAL_SOURCE_VALIDATION_SNAPSHOT.json'
    assert not snapshot.exists()
    write(snapshot, report)
    fit_path = HERE / 'MARGIN_RESUME_FIT_RESULTS.json'
    fit = read(fit_path)
    assert fit['fit_passed'] and fit['strict_runtime_admission_passed']
    assert digest(MODEL / 'checkpoint.pt') == fit['checkpoint_sha256']
    paths = [Path(__file__), HERE / 'actor.py', MODEL / 'checkpoint.pt',
        MODEL / 'parent625_actor.pt', fit_path, snapshot, HERE / 'training_data_terminal.npz',
        ROOT / 'training/experiments/native625_onpolicy.py']
    manifest = read(MODEL / 'portable/manifest.json')
    for name, expected in manifest['files'].items():
        assert digest(MODEL / 'portable' / name) == expected, name
    paths += [MODEL / 'portable/manifest.json'] + [MODEL / 'portable' / name
        for name in manifest['files']]
    rows = [row for group in report['groups'].values() for row in group['rows']]
    rows += [row for row in report.get('fresh_rows', []) if row['variant'] == 'candidate']
    assert rows
    for row in rows:
        path = ROOT / row['trace_path']
        assert digest(path) == row['trace_sha256']
        paths.append(path)
    hashes = {str(path.relative_to(ROOT)): digest(path) for path in paths}
    status(stage='read_only_neural_probe', source_validation_completed=True,
        source_physical_rows=len(rows), model_and_physics_not_loaded=False,
        soft_cpu_cores=.2, hard_service_memory_mib=2048)

    import numpy as np
    import torch
    from actor import FullNavigationResidualActor
    from experiments.native625_onpolicy import Native625OnPolicy
    from neural_navigation.experiment_budget import ExperimentBudget
    torch.set_num_threads(1)
    torch.manual_seed(231017)
    budget = ExperimentBudget(.2, 1800, HERE)
    checkpoint = torch.load(MODEL / 'checkpoint.pt', map_location='cpu', weights_only=False)
    assert checkpoint['cumulative_supervised_updates'] == 2500
    assert checkpoint['parent_sha256'] == digest(MODEL / 'parent625_actor.pt')
    assert checkpoint['data_sha256'] == digest(HERE / 'training_data_terminal.npz')
    actor = FullNavigationResidualActor(torch.jit.load(str(MODEL / 'parent625_actor.pt'),
        map_location='cpu')).eval()
    actor.load_state_dict(checkpoint['actor'])
    actor.requires_grad_(False)
    portable = torch.jit.load(str(MODEL / 'portable/actor.pt'), map_location='cpu').eval()
    learner = Native625OnPolicy(actor, learning_rate=.0001, temperature=4.)
    del checkpoint

    def allowed(x):
        return np.c_[np.ones(len(x), bool), x[:, 235:244] > .5, x[:, 267:269] > .5]

    def supported(x, masks):
        return ((x[:, 435] > .5) & (x[:, 438] > .5) & masks[:, 1:].any(1)
            & (np.linalg.norm(x[:, 218:220] * 12., axis=1) > .3))

    def key(mask):
        return np.packbits(mask).tobytes()

    donors = {}

    def add_donors(x, masks, source):
        for index in np.flatnonzero(supported(x, masks)):
            cause = x[index, 303:399]
            bucket = donors.setdefault(key(masks[index]), {})
            signature = cause.tobytes()
            if signature not in bucket and len(bucket) < 8:
                bucket[signature] = (cause.copy(), dict(source=source, frame=int(index)))

    with np.load(HERE / 'training_data_terminal.npz', allow_pickle=False) as archive:
        reference = archive['observations'].copy()
        target = archive['actions'].copy()
        reference_masks = archive['masks'].copy()
    add_donors(reference, reference_masks, 'admitted_supervised_reference')
    with torch.no_grad():
        for start in range(0, len(reference), 32):
            x = torch.from_numpy(reference[start:start + 32])
            expected = actor.actions[target[start:start + 32]].numpy()
            np.testing.assert_array_equal(actor(x).numpy(), expected)
            np.testing.assert_array_equal(portable(x).numpy(), expected)
            budget.tick()
    for row in rows:
        with np.load(ROOT / row['trace_path'], allow_pickle=False) as data:
            add_donors(data['observations'], data['masks'], row['case_id'])
        budget.tick()

    case_results = []
    examples = []
    for row in rows:
        with np.load(ROOT / row['trace_path'], allow_pickle=False) as data:
            observations = data['observations'].copy()
            masks = data['masks'].copy()
            choices = data['choices'].copy()
            controlled = data['model_controlled'].copy() if 'model_controlled' in data else np.ones(len(choices), bool)
        np.testing.assert_array_equal(masks, allowed(observations))
        totals = dict(controlled_frames=0, eligible_frames=0, only_stop_frames=0,
            voluntary_wait_frames=0, cause_swap_frames=0, greedy_changed_frames=0,
            probability_changed_frames=0, maximum_probability_change=0.,
            single_frame_parity_fallbacks=0)
        for start in range(0, len(observations), 32):
            stop = min(start + 32, len(observations))
            x = torch.from_numpy(observations[start:stop])
            original_masks = torch.from_numpy(masks[start:stop])
            take = controlled[start:stop]
            expected_choices = choices[start:stop]
            with torch.no_grad():
                features, baseline = actor.features(x)
                support = actor.support(x)
                nominal = (baseline[:, None] - actor.actions[None]).abs().amax(2).argmin(1)
                logits = (actor.head(features) + 2. * features[:, 76:88]).masked_fill(~original_masks, -1e8)
                selected = torch.where(support, logits.argmax(1), nominal).numpy()
                mismatch = take & (selected != expected_choices)
                for local in np.flatnonzero(mismatch):
                    # A single sensor frame is the runtime's authoritative shape.
                    single = x[local:local + 1]
                    sf, sb = actor.features(single)
                    actual = actor(single).numpy()[0]
                    predicted = portable(single).numpy()[0]
                    np.testing.assert_array_equal(actual, actor.actions[expected_choices[local]].numpy())
                    np.testing.assert_array_equal(actual, predicted)
                    features[local] = sf[0]
                    baseline[local] = sb[0]
                    totals['single_frame_parity_fallbacks'] += 1
                # Actual likelihoods use the same captured masks; no optimizer runs.
                controlled_indices = np.flatnonzero(take)
                if len(controlled_indices):
                    terms = learner.evaluate(x[controlled_indices], original_masks[controlled_indices],
                        torch.from_numpy(expected_choices[controlled_indices]))
                    assert torch.isfinite(terms['log_probability']).all()
                    assert torch.isfinite(terms['values']).all()
                    np.testing.assert_array_equal(terms['eligible'].numpy(), support[controlled_indices].numpy())
                assert masks[start:stop][np.arange(stop - start), expected_choices].all()
                totals['controlled_frames'] += int(take.sum())
                totals['eligible_frames'] += int((take & support.numpy()).sum())
                totals['only_stop_frames'] += int((take & ~masks[start:stop, 1:].any(1)).sum())
                totals['voluntary_wait_frames'] += int((take & (expected_choices == 0)
                    & masks[start:stop, 1:].any(1)).sum())
                for local in np.flatnonzero(take & support.numpy()):
                    original = observations[start + local, 303:399]
                    candidates = donors.get(key(masks[start + local]), {})
                    alternate = next((value for signature, value in candidates.items()
                        if signature != original.tobytes()), None)
                    if alternate is None:
                        continue
                    changed_features = features[local:local + 1].clone()
                    changed_features[:, 312:408] = torch.from_numpy(alternate[0])[None]
                    original_logit = (actor.head(features[local:local + 1])
                        + 2. * features[local:local + 1, 76:88]).masked_fill(~original_masks[local:local + 1], -1e8)
                    changed_logit = (actor.head(changed_features)
                        + 2. * features[local:local + 1, 76:88]).masked_fill(~original_masks[local:local + 1], -1e8)
                    # The actual frozen-parent decision, masks and all other features stay fixed.
                    delta = float((torch.softmax(changed_logit / 4., 1)
                        - torch.softmax(original_logit / 4., 1)).abs().max())
                    before, after = int(original_logit.argmax(1)), int(changed_logit.argmax(1))
                    assert masks[start + local, before] and masks[start + local, after]
                    totals['cause_swap_frames'] += 1
                    totals['probability_changed_frames'] += int(delta > 1e-6)
                    totals['greedy_changed_frames'] += int(before != after)
                    totals['maximum_probability_change'] = max(totals['maximum_probability_change'], delta)
                    if before != after and len(examples) < 12:
                        examples.append(dict(case_id=row['case_id'], frame=start + int(local),
                            original_choice=before, counterfactual_choice=after,
                            changed_only_new_head_current_cause_features=True,
                            donor=alternate[1], maximum_probability_change=delta))
            budget.tick()
        case_results.append(dict(case_id=row['case_id'], variant=row['variant'], **totals))
        status(stage='read_only_neural_probe', completed_cases=len(case_results), total_cases=len(rows),
            no_new_physics_or_training=True)
    for name, expected in hashes.items():
        assert digest(ROOT / name) == expected, name
    budget.tick()
    result = dict(completed=True, at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
        model=str(MODEL.relative_to(ROOT)), source_validation_completed=True,
        dependency_terminal_state=dependency, cases=case_results,
        all4468_reference_rebuild_and_export_actions_match=True,
        actual_captured_masks_and_native625_likelihoods_valid=True,
        new_head_cause_greedy_changes=sum(r['greedy_changed_frames'] for r in case_results),
        counterfactual_examples=examples,
        cause_probe='swap real one_hot cause groups from another recorded state with identical12action mask; freeze actual parent decision and remaining312features',
        counterfactuals_not_physical_or_independent_navigation_evidence=True,
        does_not_establish_understanding_or_safe_useful_recovery_by_itself=True,
        new_training_updates=0, new_ppo_steps=0, new_physical_episodes=0,
        inputs=hashes, resource=budget.statistics(), goal_achieved=False, deployment_approved=False)
    write(HERE / 'NEURAL_MECHANISM.json', result)
    status(stage='complete', completed=True, new_head_cause_greedy_changes=result['new_head_cause_greedy_changes'],
        new_training_updates=0, new_physical_episodes=0)
    print(json.dumps({k:result[k] for k in ('completed', 'model', 'new_head_cause_greedy_changes', 'resource')}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        write(HERE / 'NEURAL_MECHANISM_FAILURE.json', dict(error=repr(error), no_auto_restart=True,
            goal_achieved=False, deployment_approved=False))
        status(stage='failed', error=repr(error), no_auto_restart=True)
        raise
