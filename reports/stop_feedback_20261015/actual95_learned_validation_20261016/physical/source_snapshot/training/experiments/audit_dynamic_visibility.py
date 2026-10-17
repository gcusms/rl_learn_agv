"""Score recorded laser visibility against simulator actors, offline only.

Actor truth is used exclusively to label sensor/perception diagnostics. No
truth enters tracking, actor observations, rewards, or runtime decisions.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from navrl.geometry import ROOT, wrap
from navrl.map import StaticMap
from neural_navigation.sensors import LidarSpec


def endpoints(scan, pose, lidar):
    scan = np.asarray([np.nan if x is None else x for x in scan], float)
    valid = np.isfinite(scan) & (scan >= lidar.range_min) & (scan < lidar.range_max)
    yaw = pose[2]
    rotation = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
    origin = np.asarray(pose[:2]) + rotation @ lidar.mount[:2]
    angles = lidar.angles[valid] + yaw + lidar.mount[3]
    return origin, origin + scan[valid, None] * np.c_[np.cos(angles), np.sin(angles)]


def score_frame(row, previous, lidar, static_map, radii):
    origin, true_ends = endpoints(row['scan'], row['truth'], lidar)
    _, estimated_ends = endpoints(row['scan'], row['estimate'], lidar)
    retained = static_map.lookup(estimated_ends, static_map.distance, outside=0.) > .22
    actors = np.asarray(row['actors'], float).reshape(-1, 2)
    result = []
    for index, center in enumerate(actors):
        radius = radii[index]
        distance = float(np.linalg.norm(center-origin))
        bearing = float(wrap(np.arctan2(*(center-origin)[::-1]) - row['truth'][2] - lidar.mount[3]))
        half_angle = np.arcsin(min(1., radius/max(distance, 1e-9)))
        fov = any(bearing+shift+half_angle >= lidar.angle_min and
                  bearing+shift-half_angle <= lidar.angle_max for shift in (-2*np.pi, 0., 2*np.pi))
        speed = None
        if previous is not None and index < len(previous['actors']):
            dt = row['timestamp']-previous['timestamp']
            if dt <= 0:
                raise ValueError('Strictly increasing trace timestamps required')
            speed = float(np.linalg.norm(center-np.asarray(previous['actors'][index]))/dt)
        close = np.linalg.norm(true_ends-center, axis=1) <= radius+.04
        result.append(dict(actor_index=index, center_distance_m=distance,
            moving=speed is not None and .025 < speed <= .5, speed_m_s=speed,
            placement_discontinuity=speed is not None and speed > .5,
            body_intersects_fov=bool(fov), nearby_scan_endpoints=int(close.sum()),
            retained_nonstatic_endpoints=int((close & retained).sum())))
    return result


def audit(trace, hardware):
    case = trace['case']
    scene = ROOT/case.get('scene', 'assets/room_navigation_scripted')
    metadata = json.loads((scene/'physical_scene.json').read_text())
    radii = [float(actor['radius_m']) for actor in metadata['dynamic_obstacles']]
    lidar = LidarSpec(hardware['lidar'])
    static_map = StaticMap(scene/'decision_map.yaml', radius=.3)
    previous = None
    frames = []
    counts = dict(near_moving_actor_frames=0, no_track_with_retained_returns=0,
                  no_track_in_blind_region=0, no_track_visible_without_retained_returns=0,
                  placement_discontinuity_frames=0)
    for row in trace['samples']:
        actors = score_frame(row, previous, lidar, static_map, radii)
        near = [x for x in actors if x['moving'] and x['center_distance_m'] < 1.5]
        tracks = row['diagnostics']['scan_moving_tracks']
        counts['placement_discontinuity_frames'] += any(x['placement_discontinuity'] for x in actors)
        counts['near_moving_actor_frames'] += bool(near)
        labels = []
        if near and tracks == 0:
            if any(x['retained_nonstatic_endpoints'] >= 4 for x in near):
                labels.append('no_track_with_retained_returns')
            elif not any(x['body_intersects_fov'] for x in near):
                labels.append('no_track_in_blind_region')
            else:
                labels.append('no_track_visible_without_retained_returns')
        for label in labels:
            counts[label] += 1
        if labels:
            frames.append(dict(timestamp=row['timestamp'], labels=labels, actors=near,
                               scan_moving_tracks=tracks,
                               route_switch_reason=row['diagnostics']['route_switch_reason']))
        previous = row
    return dict(case_id=case['id'], counts=counts, diagnostic_frames=frames)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--trace', type=Path, action='append', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Use fresh output')
    hardware = json.loads((args.bundle/'hardware.json').read_text())
    cases = [audit(json.loads(path.read_text()), hardware) for path in args.trace]
    args.output.write_text(json.dumps(dict(cases=cases,
        limitation='Geometric actor labels use truth only offline; no control, training or tracker inputs change. '
        'Nearby endpoints are geometric proximity labels, not exact simulator ray object IDs. '
        'Four retained endpoints need not form one accepted cluster. Recorded tracker state updates at routing cadence, '
        'so present scan evidence can precede the next predictor update. No causal claim from frame counts.'), indent=2))
    print(json.dumps([{k: case[k] for k in ('case_id', 'counts')} for case in cases]))


if __name__ == '__main__':
    main()
