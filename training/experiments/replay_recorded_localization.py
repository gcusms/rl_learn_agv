"""Replay recorded sensor inputs; no motor publisher or absolute pose claim."""
import argparse
import json
from pathlib import Path
import time

import numpy as np

from navrl.map import StaticMap
from neural_navigation.experiment_budget import ExperimentBudget
from neural_navigation.fast_static_localizer import FastStaticLocalizer
from neural_navigation.localization_bundle import localization_configuration
from neural_navigation.sensors import LidarSpec


def replay(record, bundle, output, cpu_cores=.25):
    record, bundle, output = Path(record), Path(bundle), Path(output)
    config = localization_configuration(bundle)
    if config is None:
        raise ValueError('An explicit geometric localization bundle is required')
    output.mkdir(parents=True, exist_ok=False)
    budget = ExperimentBudget(cpu_cores, directory=output)
    runs = list((record / 'snapshots/runs').iterdir())
    if len(runs) != 1:
        raise ValueError('Select a recording with exactly one map snapshot')
    static_map = StaticMap(runs[0] / 'prepared_map/map.yaml', radius=.3)
    lidar = LidarSpec(json.loads((bundle / 'hardware.json').read_text())['lidar'])
    loc = FastStaticLocalizer(static_map, lidar, config,
        kernel_path=str((bundle / 'localization_kernel.so').resolve()),
        uncertainty=json.loads((bundle / 'uncertainty_contract.json').read_text())
            if (bundle / 'uncertainty_contract.json').exists() else None)
    rows = [json.loads(line)['data'] for line in (record / 'decisions.jsonl').read_text().splitlines() if line.strip()]
    if not rows:
        raise ValueError('Recording contains no decisions')
    previous = object()
    poses, recorded, odometry, segments, reasons, timings = [], [], [], [], [], []
    for row in rows:
        if row['mission_id'] != previous:
            loc.reset(row['initial'], None)
            previous = row['mission_id']
            segments.append(dict(mission=previous, start=len(poses)))
        odom = np.asarray(row['inputs']['odom'], float)
        started = time.perf_counter()
        poses.append(loc.update(odom, np.asarray(row['inputs']['scan'], float), row['scan_stamp_ns'] / 1e9))
        timings.append((time.perf_counter() - started) * 1000)
        odometry.append(odom)
        recorded.append(row['result']['diagnostics']['estimated_pose'])
        reasons.append(loc.correction_reason)
        budget.tick()
    poses, recorded, odometry = map(np.asarray, (poses, recorded, odometry))
    for index, segment in enumerate(segments):
        start = segment.pop('start')
        stop = segments[index + 1]['start'] if index + 1 < len(segments) else len(poses)
        def displacement(values):
            return float(np.linalg.norm(values[stop - 1, :2] - values[start, :2]))
        segment.update(frames=stop - start, recorded_displacement_m=displacement(recorded),
            candidate_displacement_m=displacement(poses), input_odometry_displacement_m=displacement(odometry),
            candidate_map_corrections=reasons[start:stop].count('accepted_static_correction'))
    result = dict(scope='fixed recorded sensor inputs; no independent absolute pose reference; not closed-loop navigation',
        frames=len(rows), deployment_approved=False, localization_contract=config, missions=segments,
        reasons={reason: reasons.count(reason) for reason in sorted(set(reasons))},
        localizer_wall_ms=dict(p50=float(np.percentile(timings, 50)), p95=float(np.percentile(timings, 95)),
            p99=float(np.percentile(timings, 99)), maximum=float(max(timings))), resource=budget.statistics())
    np.savez_compressed(output / 'replay.npz', candidate=poses, recorded=recorded, odometry=odometry)
    (output / 'replay.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--record', type=Path, required=True)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cpu-cores', type=float, default=.25)
    args = parser.parse_args()
    result = replay(args.record, args.bundle, args.output, args.cpu_cores)
    print(json.dumps({key: value for key, value in result.items() if key != 'missions'}, indent=2))
