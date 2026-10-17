"""Replay only scan/map fusion from captured sensor poses, never a controller."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from navrl.geometry import ROOT
from navrl.map import StaticMap
from neural_navigation.laser_fusion import LaserMapFusion
from neural_navigation.sensors import LidarSpec


class ReplayState:
    def __init__(self, scene, hardware, config):
        self.static_map = StaticMap(scene/'decision_map.yaml', radius=.3)
        self.lidar = LidarSpec(hardware['lidar'])
        self.motion_prediction_config = config
        self.free = self.static_map.free[2::4, 2::4].astype(np.float32)
        self.grid_h, self.grid_w = self.free.shape
        self.grid_res = self.static_map.res*4
        self.expiry = np.zeros_like(self.free)
        self.dynamic_memory_seconds = 8.
        self.localizer = SimpleNamespace()

    def cell(self, point):
        row, col = self.static_map.cells(point)
        return int(row)//4, int(col)//4

    def xy(self, cell):
        row, col = cell
        return self.static_map.origin[:2]+np.array([(col+.5)*self.grid_res+self.static_map.res/2,
            (self.grid_h-row-.5)*self.grid_res-self.static_map.res/2])


def audit(trace, hardware, config):
    state = ReplayState(ROOT/trace['case'].get('scene', 'assets/room_navigation_scripted'), hardware, config)
    fusion = LaserMapFusion(state)
    last_update = -100.
    frames = []
    mismatches = []
    for row in trace['samples']:
        state.elapsed = row['timestamp']
        state.estimate = np.asarray(row['estimate'])
        state.scan = np.array([np.nan if x is None else x for x in row['scan']])
        diagnostic = row['diagnostics']
        state.localizer.std = np.asarray(diagnostic['localization_std'])
        state.localizer.scan_update_accepted = diagnostic['localization_scan_update_accepted']
        state.localizer.map_agreement = diagnostic['localization_map_agreement']
        updated = state.elapsed-last_update >= .4
        if updated:
            fusion.update(state)
            last_update = state.elapsed
        moving = fusion.predictor.moving_tracks
        forecast = fusion.forecast_cells
        if moving != diagnostic['scan_moving_tracks'] or forecast != diagnostic['scan_forecast_cells']:
            mismatches.append(state.elapsed)
        if updated:
            frames.append(dict(timestamp=state.elapsed, moving_tracks=moving,
                forecast_cells=forecast, tracks=[dict(id=t['id'], hits=t['hits'],
                    center=t['center'].tolist(), velocity=t['velocity'].tolist(),
                    speed_m_s=float(np.linalg.norm(t['velocity'])), age_s=state.elapsed-t['time'],
                    points=len(t['points'])) for t in fusion.predictor.tracks]))
    return dict(case_id=trace['case']['id'], exact_recorded_counter_match=not mismatches,
                mismatch_timestamps=mismatches, predictor_updates=frames)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--trace', type=Path, action='append', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Fresh output required')
    torch.set_num_threads(1)
    hardware = json.loads((args.bundle/'hardware.json').read_text())
    config = torch.load(args.bundle/'router.pt', map_location='cpu', weights_only=True).get('motion_prediction')
    if not config:
        parser.error('Motion predictor configuration required')
    cases = [audit(json.loads(path.read_text()), hardware, config) for path in args.trace]
    args.output.write_text(json.dumps(dict(cases=cases, configuration=config,
        limitation='Read-only fusion replay, no neural actions or simulator actors consumed. Counter matching '
        'supports reproducibility of these two diagnostic outputs, not complete tracker correctness or navigation success.'), indent=2))
    print(json.dumps([{k: c[k] for k in ('case_id', 'exact_recorded_counter_match', 'mismatch_timestamps')} for c in cases]))


if __name__ == '__main__':
    main()
