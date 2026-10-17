"""Reproduce complete safe sensor traces and add only causal risk history."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from neural_navigation.decision_feedback_runtime import StopFeedbackRuntime
from neural_navigation.experiment_budget import ExperimentBudget
from neural_navigation.long_risk_observation import LongRiskHistory, contract
from neural_navigation.masked_actions import ACTIONS, masks_from_observation


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'reports/stop_feedback_20261015'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def main(output):
    output.mkdir(parents=True, exist_ok=False)
    budget = ExperimentBudget(.35, memory_mib=2048, directory=output)
    parent = ROOT / 'models/stop_feedback_20261015/early_yield_bc400_s231016/bc_portable'
    source_names = ['early_yield_physical_20261016', 'early_yield_condition02_20261016',
        'early_yield_condition05_20261016']
    paths = [Path(__file__), parent / 'actor.pt', parent / 'manifest.json',
        BASE / 'long_risk_parity_20261016/RESULTS.json']
    assert json.loads(paths[-1].read_text())['completed']
    for folder in ('neural_navigation', 'navrl'):
        paths += list((ROOT / 'training' / folder).rglob('*.py'))
    for index, name in enumerate(source_names):
        source = BASE / name
        paths += [source / 'RESULTS.json', source / 'case.json']
        row = json.loads((source / 'RESULTS.json').read_text())['rows'][0]
        paths.append(ROOT / row['trace_path'])
        source_parent = 'actual95_transfer_bc400_s231016' if index == 0 else 'early_yield_bc400_s231016'
        bundle = ROOT / 'models/stop_feedback_20261015' / source_parent / 'bc_portable'
        manifest = json.loads((bundle / 'manifest.json').read_text())
        paths += [bundle / 'manifest.json', *[bundle / name for name in manifest['files']]]
    inputs = {str(path.relative_to(ROOT)): digest(path) for path in set(paths)}
    write(output / 'PLAN.json', dict(inputs=inputs, contract=contract(),
        only_complete_safe_sources=True, parent_actor_sha256=digest(parent / 'actor.pt'),
        new_physical_steps=0, new_optimizer_updates=0, failed_noise_cases_excluded=True))
    torch.set_num_threads(1)
    baseline = torch.jit.load(str(parent / 'actor.pt'), map_location='cpu').eval()
    observations, choices, masks, nominal, source_ids, times = [], [], [], [], [], []
    summaries = []
    for source_index, name in enumerate(source_names):
        source = BASE / name
        report = json.loads((source / 'RESULTS.json').read_text())
        assert report['completed'] and report['full_safe_expert_trajectory_exists']
        row = report['rows'][0]
        assert row['safe_success'] and row['physical_arrival'] and row['protocol_valid']
        assert not any(row[key] for key in ('collisions', 'timeout', 'false_arrival', 'physical_guards'))
        trace_path = ROOT / row['trace_path']
        assert digest(trace_path) == row['trace_sha256']
        with np.load(trace_path, allow_pickle=False) as archive:
            data = {key: archive[key].copy() for key in archive.files}
        source_parent = 'actual95_transfer_bc400_s231016' if source_index == 0 else 'early_yield_bc400_s231016'
        bundle = ROOT / 'models/stop_feedback_20261015' / source_parent / 'bc_portable'
        case = json.loads((source / 'case.json').read_text())
        runtime = StopFeedbackRuntime(bundle, ROOT / 'training' / case['scene'] / 'decision_map.yaml',
            np.asarray(case['start']) + np.asarray(case.get('initial_prior_offset', [0, 0, 0])),
            case['goal'], 'shortcut')
        history = LongRiskHistory()
        first = len(choices)

        class Replay:
            def __init__(self, actor):
                self.actor, self.cursor = actor, 0

            def predict(self, observation):
                i = self.cursor
                np.testing.assert_allclose(observation, data['observations'][i], rtol=0, atol=2e-6)
                np.testing.assert_array_equal(self.actor.predict(observation), ACTIONS[data['reference_choices'][i]])
                augmented = history.augment(observation, runtime.core, runtime.guard)
                np.testing.assert_array_equal(augmented[:439], observation)
                with torch.no_grad():
                    command = baseline(torch.from_numpy(observation[None])).numpy()[0]
                nominal_choice = int(np.linalg.norm(ACTIONS - command, axis=1).argmin())
                np.testing.assert_array_equal(ACTIONS[nominal_choice], command)
                mask = masks_from_observation(observation[:282])
                np.testing.assert_array_equal(mask, data['masks'][i])
                target = int(data['choices'][i])
                assert mask[target]
                observations.append(augmented)
                choices.append(target)
                nominal.append(nominal_choice)
                masks.append(mask)
                source_ids.append(source_index)
                times.append(float(data['labels'][i, 0]))
                self.cursor += 1
                return ACTIONS[target].copy()

        replay = Replay(runtime.policy.policy)
        runtime.policy.policy = replay
        for i in range(len(data['choices'])):
            result = runtime.step(data['scans'][i], data['odometry'][i], data['measured_velocity'][i],
                float(data['labels'][i, 0]), capture_observation=True)
            np.testing.assert_allclose(result['command'], data['commands'][i], rtol=0, atol=2e-8)
            np.testing.assert_allclose(runtime.core.estimate, data['labels'][i, 4:7], rtol=0, atol=2e-8)
            budget.tick()
            if i % 25 == 0:
                write(output / 'STATUS.json', dict(status='causal_safe_source_replay',
                    source=name, frame=i, total_frames=len(choices)))
        assert replay.cursor == len(data['choices'])
        x = np.asarray(observations[first:])
        history_valid = x[:, 439:].reshape(-1, 3, 14)[:, :, 12] > .5
        legal_motion = np.asarray(masks[first:])[:, 1:].any(axis=1)
        sensing_valid = (x[:, 435] > .5) & (x[:, 438] > .5)
        far = np.linalg.norm(x[:, 218:220] * 12., axis=1) > .3
        active = history_valid.any(axis=1) & legal_motion & sensing_valid & far
        mismatch = np.asarray(nominal[first:]) != np.asarray(choices[first:])
        summaries.append(dict(source=name, frames=len(x), sensor_history_supported=int(active.sum()),
            inactive_target_disagreements=int((~active & mismatch).sum()),
            inactive_disagreement_times=np.asarray(times[first:])[~active & mismatch].tolist(),
            active_stop_labels=int((active & (np.asarray(choices[first:]) == 0)).sum()),
            active_motion_labels=int((active & (np.asarray(choices[first:]) != 0)).sum()),
            trace_path=row['trace_path'], trace_sha256=row['trace_sha256'],
            no_teacher_deployed=True))
    arrays = dict(observations=np.asarray(observations, np.float32), choices=np.asarray(choices, np.int64),
        masks=np.asarray(masks, bool), parent_choices=np.asarray(nominal, np.int64),
        source_ids=np.asarray(source_ids, np.int64), sensor_times=np.asarray(times, float))
    np.savez_compressed(output / 'demonstrations.npz', **arrays)
    for name, expected in inputs.items():
        assert digest(ROOT / name) == expected, name
    write(output / 'RESULTS.json', dict(completed=True, at=datetime.now(timezone.utc).isoformat(),
        frames=len(choices), sources=summaries, dataset_sha256=digest(output / 'demonstrations.npz'),
        new_physical_steps=0, new_optimizer_updates=0, new_ppo_steps=0,
        resource=budget.statistics(), goal_achieved=False, deployment_approved=False))
    write(output / 'STATUS.json', dict(status='complete', frames=len(choices)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    try:
        main(args.output)
    except Exception as error:
        if args.output.exists():
            write(args.output / 'FAILURE.json', dict(error=repr(error), automatically_restarted=False))
        raise
