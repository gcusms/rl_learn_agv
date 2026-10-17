"""Read-only diagnostics for recorded wheel/IMU and map/ray disagreements.

Neither manual initial poses nor scan-match candidates are reference truth.
This audit never changes a map, acceptance threshold, model or ROS topic.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import label

from navrl.map import StaticMap
from neural_navigation.global_relocalizer import GlobalRelocalizer
from neural_navigation.sensors import LidarSpec


def distribution(values):
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    if not len(values):
        return dict(n=0)
    return dict(n=len(values), p50=float(np.median(values)),
                p95=float(np.percentile(values, 95)), max=float(values.max()))


def ray_details(matcher, pose, scan, components, sizes):
    """Inspect raw occupied cells; do not remove isolated map cells."""
    values = np.asarray(scan, float)[matcher.indices]
    finite = np.isfinite(values) & (values >= matcher.lidar.range_min) & (values < matcher.maximum_range)
    indices = matcher.indices[finite]
    values = values[finite]
    angles = matcher.lidar.angles[indices] + matcher.lidar.mount[3] + pose[2]
    c, s = np.cos(pose[2]), np.sin(pose[2])
    origin = pose[:2] + matcher.lidar.mount[:2] @ np.array([[c, s], [-s, c]])
    distance = np.arange(0., matcher.maximum_range, matcher.map.res / 2)
    active = distance[None, :] < np.maximum(0., values[:, None] - matcher.config['ray_endpoint_allowance_m'])
    points = origin + distance[None, :, None] * np.c_[np.cos(angles), np.sin(angles)][:, None, :]
    occupied = matcher.map.lookup(points, matcher.map.occupied, outside=False) & active
    ids = matcher.map.lookup(points, components, outside=0)
    unknown = matcher.map.lookup(points, matcher.map.unknown, outside=True)
    result = []
    for i in range(len(values)):
        blocked = np.flatnonzero(occupied[i])
        first = None if not len(blocked) else int(blocked[0])
        runs = np.diff(np.r_[False, occupied[i], False].astype(int))
        starts, ends = np.flatnonzero(runs == 1), np.flatnonzero(runs == -1)
        result.append(dict(beam=int(indices[i]), range_m=float(values[i]),
            contradicted=first is not None,
            first_occupied_m=None if first is None else float(distance[first]),
            first_component_cells=None if first is None else int(sizes[ids[i, first]]),
            maximum_occupied_run_m=0. if not len(starts) else float(max(ends-starts)*matcher.map.res/2),
            unknown_before_first=False if first is None else bool(unknown[i, :first].any())))
    return result, origin, origin + values[:, None] * np.c_[np.cos(angles), np.sin(angles)]


def audit_wheels(sensors, feedback):
    # Encoder-derived velocity is a measurement estimate, not physical truth.
    wheel = sensors['/wheel/odom']
    imu = sensors['/imu']
    command = sensors['/command/odom']
    selected = wheel[(wheel[:, 1] >= imu[0, 1]) & (wheel[:, 1] <= imu[-1, 1])]
    gyro = np.interp(selected[:, 1], imu[:, 1], imu[:, 2])
    residual = selected[:, 7] - gyro
    turning = np.abs(selected[:, 7]) > .15
    stopped = (np.max(np.abs(selected[:, 5:7]), axis=1) < .005) & (np.abs(selected[:, 7]) < .01)
    reconstructed = []
    for row in feedback:
        data = row['data']
        if not data.get('fresh') or not data.get('measurement_valid'):
            continue
        lf, lr, rf, rr = np.array(data['measured_rps']) * np.pi * data['geometry_m']['wheel_diameter']
        k = (data['geometry_m']['wheelbase'] + data['geometry_m']['track_width']) / 2
        body = np.array([(lf+lr-rf-rr)/4, (-lf+lr-rf+rr)/4, -(lf+lr+rf+rr)/(4*k)])
        published = data['body_velocity']
        reconstructed.append(body - [published['vx_m_s'], published['vy_m_s'], published['wz_rad_s']])
    expected = np.column_stack([np.interp(selected[:, 1], command[:, 1], command[:, c]) for c in (5, 7)])
    return dict(frames=len(wheel), fresh_valid_feedback=len(reconstructed),
        velocity_reconstruction_max_abs=float(np.max(np.abs(reconstructed))),
        wheel_gyro_rmse_rad_s=float(np.sqrt(np.mean(residual**2))),
        wheel_gyro_absolute_error_rad_s=distribution(np.abs(residual)),
        turning_samples=int(turning.sum()),
        turning_direction_agreement=float(np.mean(np.sign(selected[turning, 7]) == np.sign(gyro[turning]))),
        stopped_gyro_rad_s=distribution(np.abs(gyro[stopped])),
        command_wheel_vx_absolute_difference_m_s=distribution(abs(expected[:, 0]-selected[:, 5])),
        wheel_vy_absolute_m_s=distribution(abs(selected[:, 6])),
        independent_ground_truth=False, wheel_scale_calibrated=False,
        note='Body velocity reconstruction checks wheel order/sign algebra only, not physical mounting or distance calibration.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--analysis', type=Path, required=True)
    parser.add_argument('--record', type=Path, required=True)
    parser.add_argument('--hardware', type=Path, required=True)
    parser.add_argument('--recovery', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    sensors = np.load(args.analysis/'sensors.npz')
    feedback = json.loads((args.analysis/'feedback.json').read_text())
    rows = [json.loads(line)['data'] for line in (args.record/'decisions.jsonl').open()]
    by_id = {row['inference_id']: row for row in rows}
    run, = (args.record/'snapshots/runs').iterdir()
    m = StaticMap(run/'prepared_map/map.yaml', radius=.3)
    lidar = LidarSpec(json.loads(args.hardware.read_text())['lidar'])
    matcher = GlobalRelocalizer(m, lidar)
    components, count = label(m.occupied, structure=np.ones((3, 3)))
    sizes = np.bincount(components.ravel())
    result = dict(independent_pose_truth_available=False, thresholds_changed=False,
                  wheel_imu=audit_wheels(sensors, feedback), cases=[],
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    extent = [*m.origin[:1], m.origin[0]+m.width*m.res, m.origin[1], m.origin[1]+m.height*m.res]
    image = np.where(m.unknown, .65, np.where(m.occupied, .05, .95))
    for case, ax in zip(json.loads(args.recovery.read_text())['cases'], axes.ravel()):
        row = by_id[case['inference_id']]
        modes = case['recovery']['search']['modes']
        if not modes:
            continue
        scan = np.array(row['inputs']['scan'], float)
        pose = np.array(modes[0]['pose'])
        rays, origin, endpoints = ray_details(matcher, pose, scan, components, sizes)
        bad = np.array([ray['contradicted'] for ray in rays])
        same = [r for r in rows if r['mission_id'] == case['mission'] and r['inference_id'] >= row['inference_id']]
        confirmation = []
        for later in same[:8]:
            check = matcher.verify(pose, np.array(later['inputs']['scan'], float), refine=False)
            confirmation.append(dict(inference_id=later['inference_id'], **check))
        result['cases'].append(dict(mission=case['mission'], inference_id=row['inference_id'],
            candidate_pose=pose.tolist(), manual_or_default_initial=row['initial'],
            candidate_is_truth=False, rays=rays, fixed_pose_fresh_frames=confirmation))
        ax.imshow(image, extent=extent, cmap='gray', vmin=0, vmax=1)
        for endpoint in endpoints[bad]:
            ax.plot([origin[0], endpoint[0]], [origin[1], endpoint[1]], c='tomato', lw=.5, alpha=.6)
        ax.scatter(*endpoints[~bad].T, s=6, c='royalblue', label='endpoint / no ray contradiction')
        ax.scatter(*endpoints[bad].T, s=10, c='crimson', label='ray crosses mapped occupancy')
        ax.scatter(*pose[:2], marker='>', s=70, c='green', label='top candidate (not truth)')
        ax.set_title(f"Mission {case['mission']}: {bad.sum()}/{len(bad)} contradicted")
        ax.set_aspect('equal'); ax.set_xlabel('map x / m'); ax.set_ylabel('map y / m')
    axes[0, 0].legend(fontsize=6)
    fig.tight_layout(); fig.savefig(args.output/'map_scan_candidates.png', dpi=130); plt.close(fig)
    (args.output/'RESULTS.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result['wheel_imu'], indent=2))
    for case in result['cases']:
        bad = [r for r in case['rays'] if r['contradicted']]
        print(json.dumps(dict(mission=case['mission'], contradicted=len(bad),
            tiny_components=sum(r['first_component_cells'] <= 4 for r in bad),
            thick_runs=sum(r['maximum_occupied_run_m'] >= .15 for r in bad),
            confirmation_reasons=[r['reason'] for r in case['fixed_pose_fresh_frames']])))


if __name__ == '__main__':
    main()
