"""Verify runtime snapshot branches on an actual recorded recovery history.

Recorded sensor inputs and actions are replayed. No counterfactual physical
outcome, navigation success or speedup of the full simulator is inferred.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

from experiments.long_risk_actor import digest
from experiments.recovery_state_snapshot import RecoverySnapshot
from experiments.refit_phase_tail_pair import write
from neural_navigation.experiment_budget import ExperimentBudget
from neural_navigation.masked_actions import ACTIONS, masks_from_observation
from neural_navigation.release_runtime import ReleaseRuntime


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'reports/stop_feedback_20261015'


class RecordingState:
    def __init__(self, choices):
        self.renderer = None
        self.cursor = 0
        self.choices = choices.copy()


class RecordedActions:
    def __init__(self, state):
        self.state = state

    def predict(self, observation):
        choice = self.state.choices[self.state.cursor]
        self.state.cursor += 1
        return ACTIONS[choice].copy()


def main(args):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    budget = ExperimentBudget(.1, memory_mib=1400, directory=output)
    tasks = BASE / 'context_feedback_longfit_protocol_20261016/tasks.json'
    specification = json.loads(tasks.read_text())['actual106']
    source = BASE / 'context_broad_diagnostic_20261016/physical/hard3_w11_training_stall_t106.npz'
    bundle = ROOT / 'models/stop_feedback_20261015/sensor_retention_pair_s231016/retained500/portable'
    manifest = json.loads((bundle / 'manifest.json').read_text())
    files = [Path(__file__), ROOT / 'training/experiments/recovery_state_snapshot.py',
        ROOT / 'training/experiments/test_recovery_state_snapshot.py', tasks, source,
        bundle / 'manifest.json']
    files += [bundle / name for name in manifest['files']]
    for folder in ['neural_navigation', 'navrl']:
        files += list((ROOT / 'training' / folder).rglob('*.py'))
    files += [p for p in (ROOT / 'training' / specification['case']['scene']).iterdir() if p.is_file()]
    hashes = {str(p.relative_to(ROOT)): digest(p) for p in set(files)}
    write(output / 'PLAN.json', dict(inputs=hashes, prefix_frames=530,
        suffix_frames_per_branch=15, repeated_restores=2,
        no_mujoco_environment_created=True, recorded_sensor_and_action_replay_only=True,
        no_new_physical_steps_or_training=True, no_counterfactual_success_inferred=True,
        shared_models_must_remain_unchanged=True, new_ppo_steps=0,
        goal_achieved=False, deployment_approved=False))
    with np.load(source, allow_pickle=False) as archive:
        trace = {key: archive[key].copy() for key in archive.files}
    assert trace['observations'].shape == (700, 625)
    prefix = int(np.flatnonzero(trace['model_controlled'])[0])
    assert prefix == 530
    state = RecordingState(trace['choices'])
    case = specification['case']
    initial = np.asarray(case['start']) + np.asarray(case.get('initial_prior_offset', [0., 0., 0.]))
    runtime = ReleaseRuntime(bundle, ROOT / 'training' / case['scene'] / 'decision_map.yaml',
        initial, case['goal'], 'shortcut')
    runtime.policy.policy = RecordedActions(state)

    def step(state, runtime, index):
        assert state.cursor == index and runtime.policy.policy.state is state
        result = runtime.step(trace['scans'][index], trace['odometry'][index],
            trace['measured_velocity'][index], float(trace['labels'][index, 0]),
            capture_observation=True)
        observation = np.asarray(result['policy_observation'], np.float32)
        np.testing.assert_array_equal(observation, trace['observations'][index])
        np.testing.assert_array_equal(masks_from_observation(observation[:282]), trace['masks'][index])
        np.testing.assert_array_equal(result['proposed_action'], ACTIONS[trace['choices'][index]])
        np.testing.assert_allclose(result['command'], trace['commands'][index], rtol=0, atol=2e-8)
        budget.tick()
        return observation, np.asarray(result['command'])

    prefix_cpu = time.process_time()
    last = time.monotonic()
    for index in range(prefix):
        step(state, runtime, index)
        if time.monotonic() - last > 30.:
            write(output / 'STATUS.json', dict(status='causal_prefix_replay', frames=index + 1,
                prefix_frames=prefix))
            last = time.monotonic()
    prefix_cpu = time.process_time() - prefix_cpu
    snapshot_cpu = time.process_time()
    snapshot = RecoverySnapshot(state, runtime)
    snapshot_cpu = time.process_time() - snapshot_cpu
    original = [step(state, runtime, i) for i in range(prefix, prefix + 15)]
    rows = []
    for repeat in range(2):
        started = time.process_time()
        restored_state, restored_runtime = snapshot.restore()
        restore_cpu = time.process_time() - started
        assert restored_state.cursor == prefix
        for offset, index in enumerate(range(prefix, prefix + 15)):
            observation, command = step(restored_state, restored_runtime, index)
            np.testing.assert_array_equal(observation, original[offset][0])
            np.testing.assert_array_equal(command, original[offset][1])
        rows.append(dict(repeat=repeat, matched_suffix_frames=15,
            restore_cpu_seconds=restore_cpu, complete625_observations_and_commands_bit_exact=True))
    snapshot.verify_shared_models()
    for name, expected in hashes.items():
        assert digest(ROOT / name) == expected, name
    write(output / 'RESULTS.json', dict(completed=True, at=datetime.now(timezone.utc).isoformat(),
        original_prefix_frames=prefix, original_suffix_frames=15, restored_suffix_frames=30,
        repeated_restores=rows, original_prefix_cpu_seconds=prefix_cpu,
        snapshot_creation_cpu_seconds=snapshot_cpu,
        full_sensor_runtime_restoration_verified=True, input_hashes_verified=len(hashes),
        physical_environment_restoration_not_yet_verified=True,
        full_simulator_or_training_speedup_not_claimed=True,
        simulator_loaded='mujoco' in sys.modules, new_physical_steps=0,
        new_optimizer_updates=0, new_ppo_steps=0, goal_achieved=False,
        deployment_approved=False, resource=budget.statistics()))
    write(output / 'STATUS.json', dict(status='complete'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    try:
        main(args)
    except Exception as error:
        if args.output.is_dir():
            write(args.output / 'FAILURE.json', dict(error=repr(error), automatically_restarted=False))
        raise
