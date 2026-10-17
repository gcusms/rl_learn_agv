"""Profile recorded inputs without physics, training, ROS or vehicle commands."""
import argparse
import cProfile
import io
import json
from pathlib import Path
import pstats
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'training'), str(ROOT)]
import numpy as np
import torch
from neural_navigation.experiment_budget import ExperimentBudget
from neural_navigation.release_runtime import ReleaseRuntime


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--frames', type=int, default=64)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    base = ROOT / 'reports/stop_feedback_20261015'
    bundle = ROOT / 'models/stop_feedback_20261015/release_context_bc2000_s231016/portable'
    case = next(c for c in json.loads((base / 'early_yield_remaining_protocol_20261016/heldout_cases.json').read_text())
        if c['id'] == 'early_yield_sensor_holdout_05_0')
    row = json.loads((base / 'release_validation_20261016/RESULTS.json').read_text())['rows'][0]
    trace = np.load(ROOT / row['trace_path'], allow_pickle=False)
    budget = ExperimentBudget(.35, memory_mib=2048, directory=args.output)
    started = time.perf_counter()
    runtime = ReleaseRuntime(bundle, ROOT / 'training' / case['scene'] / 'decision_map.yaml',
        case['start'], case['goal'], 'shortcut')
    startup = time.perf_counter() - started
    profiler = cProfile.Profile()
    latencies = []
    for index in range(min(args.frames, len(trace['scans']))):
        profiler.enable()
        started = time.process_time()
        result = runtime.step(trace['scans'][index], trace['odometry'][index],
            trace['measured_velocity'][index], float(trace['labels'][index, 0]), capture_observation=True)
        latencies.append((time.process_time() - started) * 1000)
        profiler.disable()
        np.testing.assert_array_equal(result['policy_observation'], trace['observations'][index])
        np.testing.assert_allclose(result['command'], trace['commands'][index], atol=1e-9, rtol=0)
        budget.tick()
    profiler.dump_stats(str(args.output / 'profile.pstats'))
    stream = io.StringIO()
    stats = pstats.Stats(profiler, stream=stream).sort_stats('cumulative')
    stats.print_stats(35)
    (args.output / 'PROFILE.txt').write_text(stream.getvalue())
    summary = dict(completed=True, frames=len(latencies), exact_observations=True,
        startup_seconds=startup, profiled_cpu_ms={f'p{q}': float(np.percentile(latencies, q)) for q in [50, 95, 99, 100]},
        resource=budget.statistics(), simulator_loaded='mujoco' in sys.modules,
        ros_node_executed=False, vehicle_started=False, target_board_measured=False)
    (args.output / 'RESULTS.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))
    print(stream.getvalue())


if __name__ == '__main__':
    main()
