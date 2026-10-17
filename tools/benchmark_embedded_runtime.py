"""Bounded causal replay and CPU/RSS measurements on the machine running this file.

No simulation, PPO, ROS node, or robot commands. An optional paired scalar
backend tests identical observations and decisions, not closed-loop safety.
"""
import argparse
from contextlib import contextmanager
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'training'), str(ROOT)]
import numpy as np
import torch
from neural_navigation.experiment_budget import ExperimentBudget
from neural_navigation.release_runtime import ReleaseRuntime
from neural_navigation.masked_actions import ACTIONS, masks_from_observation
from neural_navigation import directional_risk, predictive_constraint, quality_prediction_context
from neural_navigation.directional_risk import directional_body_risk_reference
from neural_navigation.directional_risk_batch import batched_directional_body_risk


@contextmanager
def backend(function):
    # This script has one inference thread and no live ROS workers. Replacing
    # imports is scoped to one call and restored even when a check fails.
    modules = [directional_risk, predictive_constraint, quality_prediction_context]
    previous = [m.directional_body_risk for m in modules]
    try:
        for module in modules:
            module.directional_body_risk = function
        yield
    finally:
        for module, original in zip(modules, previous):
            module.directional_body_risk = original


def percentiles(values):
    return {f'p{q}': float(np.percentile(values, q)) for q in [50, 95, 99, 100]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--frames-per-case', type=int, default=64)
    parser.add_argument('--cpu-budget', type=float, default=.35)
    parser.add_argument('--compare-reference', action='store_true')
    parser.add_argument('--bundle', type=Path,
        help='Use a separately rebuilt native bundle, e.g. generated/candidate_native on ARM64')
    args = parser.parse_args()
    if args.frames_per_case < 16:
        parser.error('At least 16 frames per case for causal warmup')
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    reference_ros = ROOT / 'reference_ros'
    if not reference_ros.is_dir():
        reference_ros = ROOT / 'exports/realtest_20261016/reference_ros'
    sys.path.insert(0, str(reference_ros))
    from rl_navigation.fast_guard import install_acceleration
    install_acceleration()  # Both backends use the existing deployed map check.
    base = ROOT / 'reports/stop_feedback_20261015'
    model = ROOT / 'models/stop_feedback_20261015/release_context_bc2000_s231016'
    bundle = args.bundle.resolve() if args.bundle else model / 'portable'
    original_manifest = json.loads((model / 'portable/manifest.json').read_text())
    manifest = json.loads((bundle / 'manifest.json').read_text())
    # A target-native rebuild may replace only the shared library. Benchmark
    # the same model and contract, rather than silently measuring other weights.
    expected_files = {k: v for k, v in original_manifest['files'].items()
        if k != 'localization_kernel.so'}
    actual_files = {k: v for k, v in manifest['files'].items()
        if k != 'localization_kernel.so'}
    if actual_files != expected_files or manifest['observation_contract'] != original_manifest['observation_contract']:
        raise ValueError('Benchmark bundle must retain the archived model and observation contract')
    validation = json.loads((base / 'release_validation_20261016/RESULTS.json').read_text())
    cases = {}
    for relative in ['proactive_augmented_protocol_20261016/fresh_cases.json',
            'early_yield_remaining_protocol_20261016/heldout_cases.json',
            'long_risk_protocol_20261016/new_sensor_cases.json']:
        cases.update({c['id']: c for c in json.loads((base / relative).read_text())})
    selected = ['early_yield_sensor_holdout_05_0', 'long_risk_sensor_02_0', 'long_risk_sensor_05_1']
    rows = {r['case_id']: r for r in validation['rows']}
    budget = ExperimentBudget(args.cpu_budget, memory_mib=2048, directory=args.output)
    functions = {'batch': batched_directional_body_risk}
    if args.compare_reference:
        functions['reference'] = directional_body_risk_reference
    wall = {name: [] for name in functions}
    cpu = {name: [] for name in functions}
    startups = {name: [] for name in functions}
    frame_count = 0
    case_reports = []
    last_progress = time.monotonic()
    for case_id in selected:
        case = cases[case_id]
        row = rows[case_id]
        initial = np.asarray(case['start']) + np.asarray(case.get('initial_prior_offset', [0., 0., 0.]))
        runtimes = {}
        for name, function in functions.items():
            start = time.perf_counter()
            with backend(function):
                runtimes[name] = ReleaseRuntime(bundle,
                    ROOT / 'training' / case['scene'] / 'decision_map.yaml', initial, case['goal'], 'shortcut')
            startups[name].append(time.perf_counter() - start)
            budget.tick()
        with np.load(ROOT / row['trace_path'], allow_pickle=False) as trace:
            frames = min(args.frames_per_case, len(trace['scans']))
            for index in range(frames):
                results = {}
                # Alternate order so a single backend does not always get the
                # cache/thermal advantage. Keep each runtime's causal state.
                order = list(functions) if index % 2 else list(reversed(functions))
                for name in order:
                    start_wall, start_cpu = time.perf_counter(), time.process_time()
                    with backend(functions[name]):
                        result = runtimes[name].step(trace['scans'][index], trace['odometry'][index],
                            trace['measured_velocity'][index], float(trace['labels'][index, 0]), capture_observation=True)
                    cpu_ms = (time.process_time() - start_cpu) * 1000
                    wall_ms = (time.perf_counter() - start_wall) * 1000
                    # The first 8 causal frames warm model/kernel caches and
                    # are still checked, but do not enter steady-state timing.
                    if index >= 8:
                        wall[name].append(wall_ms)
                        cpu[name].append(cpu_ms)
                    observation = np.asarray(result['policy_observation'], np.float32)
                    np.testing.assert_array_equal(observation, trace['observations'][index])
                    np.testing.assert_array_equal(masks_from_observation(observation[:282]), trace['masks'][index])
                    np.testing.assert_allclose(result['command'], trace['commands'][index], atol=1e-9, rtol=0)
                    np.testing.assert_array_equal(ACTIONS[trace['choices'][index]], result['proposed_action'])
                    np.testing.assert_allclose(result['diagnostics']['estimated_pose'], trace['labels'][index, 4:7], atol=1e-12, rtol=0)
                    results[name] = result
                    budget.tick()
                if len(results) == 2:
                    a, b = results['batch'], results['reference']
                    assert a['stop_reason'] == b['stop_reason']
                    assert a['current_action_rejection_reasons'] == b['current_action_rejection_reasons']
                frame_count += 1
                if time.monotonic() - last_progress > 25.:
                    print(json.dumps(dict(case_id=case_id, frame=index + 1, frames=frames,
                        checked_sensor_frames=frame_count)), flush=True)
                    last_progress = time.monotonic()
        case_reports.append(dict(case_id=case_id, replayed_frames=frames,
            original_full_episode_success=row['safe_success'], original_trace_sha256=row['trace_sha256']))
    forbidden = [name for name in ['mujoco', 'gymnasium', 'stable_baselines3', 'rclpy'] if name in sys.modules]
    assert not forbidden, forbidden
    result = dict(completed=True, measured_at_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        machine=platform.machine(), platform=platform.platform(), python=sys.version,
        torch=importlib.metadata.version('torch'), numpy=np.__version__,
        cpu_name=next((line.split(':', 1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines()
            if line.startswith('model name') or line.startswith('Hardware')), platform.processor()),
        cases=case_reports, sensor_frames=frame_count, backends=list(functions),
        timings={name: dict(steady_state_samples=len(cpu[name]), cpu_ms=percentiles(cpu[name]),
            wall_ms=percentiles(wall[name]), startup_seconds=startups[name]) for name in functions},
        observations_masks_choices_commands_and_estimates_match_recording=True,
        paired_stop_and_rejection_reasons_identical=bool(args.compare_reference),
        resource=budget.statistics(), paired_process_holds_two_runtime_models=bool(args.compare_reference),
        checkpoint_sha256=hashlib.sha256((model / 'checkpoint.pt').read_bytes()).hexdigest(),
        runtime_bundle=str(bundle),
        runtime_manifest_sha256=hashlib.sha256((bundle / 'manifest.json').read_bytes()).hexdigest(),
        native_kernel=manifest.get('native_kernel'),
        optimized_source_sha256=hashlib.sha256((ROOT / 'training/neural_navigation/directional_risk_batch.py').read_bytes()).hexdigest(),
        uses_existing_ros_guard_acceleration=True, physics_episodes=0, vehicle_started=False,
        forbidden_runtime_imports=forbidden,
        limitations=['Recorded causal replay is not a closed-loop navigation success evaluation.',
            'Measures this machine only; includes sensor core/model/guard, excludes ROS transport, sensor alignment, IPC, other nodes and power.',
            'Quota-related wall delays are not proof of target-board deadlines.',
            'Only the selected frame prefixes are checked; complete real-car or full-scene regression is not established.'])
    if args.compare_reference:
        result['cpu_p50_reduction_fraction'] = 1. - np.median(cpu['batch']) / np.median(cpu['reference'])
        result['cpu_p95_reduction_fraction'] = 1. - np.percentile(cpu['batch'], 95) / np.percentile(cpu['reference'], 95)
    (args.output / 'RESULTS.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
