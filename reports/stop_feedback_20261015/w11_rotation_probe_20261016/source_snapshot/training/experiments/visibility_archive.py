"""Isolated scan-track archive for bounded, confirmed reacquisition.

Fresh forecasts retain the original 0.8s age limit. Older motion tracks are
association candidates only; they cannot create stale future occupancy.
No simulator actors or alternate steering controller are accepted.
"""
import numpy as np
from navrl.geometry import wrap
from neural_navigation.motion_prediction import ScanMotionPredictor


class VisibilityArchivePredictor(ScanMotionPredictor):
    def __init__(self, config, core, archive_s=4.):
        if not np.isfinite(archive_s) or not .8 <= archive_s <= 4.:
            raise ValueError('Archive lifetime must be finite and bounded at four seconds')
        self.core = core
        self.archive_s = float(archive_s)
        super().__init__(config)

    def reset(self):
        super().reset()
        self.reassociations = 0
        self.clear_removals = 0

    def _observed_clear(self, track, now):
        core = self.core
        pose = core.estimate
        yaw = pose[2]
        lidar = core.lidar
        rotation = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
        origin = pose[:2]+rotation@lidar.mount[:2]
        points = track['points']+track['velocity']*(now-track['time'])
        delta = points-origin
        distance = np.linalg.norm(delta, axis=1)
        bearing = wrap(np.arctan2(delta[:, 1], delta[:, 0])-yaw-lidar.mount[3])
        indices = np.clip(np.searchsorted(lidar.angles, bearing), 0, len(core.scan)-1)
        scan = np.asarray(core.scan)
        valid = np.isfinite(scan)&(scan >= lidar.range_min)&(scan < lidar.range_max)
        ranges = np.where(valid, scan, 0.)
        observed = np.minimum.reduce([ranges[np.clip(indices+k, 0, len(scan)-1)] for k in (-1, 0, 1)])
        clear = ((bearing >= lidar.angle_min)&(bearing <= lidar.angle_max)&
                 (observed > distance+.22))
        return bool(len(clear) >= 4 and np.all(clear))

    def update(self, points, timestamp, alignment_supported=True):
        if self.last_time is not None and timestamp <= self.last_time:
            return self.predicted_points
        self.last_time = timestamp
        self.predicted_points = np.empty((0, 2))
        self.moving_tracks = 0
        if not alignment_supported:
            self.tracks = []
            return self.predicted_points
        points = np.asarray(points, float).reshape(-1, 2)
        points = points[np.isfinite(points).all(axis=1)]
        self.tracks = [t for t in self.tracks if timestamp-t['time'] <=
                       (self.archive_s if t.get('confirmed_motion', False) else .8)]
        groups = []
        if len(points):
            cuts = np.flatnonzero(np.linalg.norm(np.diff(points, axis=0), axis=1) > .18)+1
            for group in np.split(points, cuts):
                if len(group) < 4:
                    continue
                center = np.median(group, axis=0)
                if np.max(np.linalg.norm(group-center, axis=1)) <= .7:
                    groups.append((center, group))
        pairs = []
        for ti, track in enumerate(self.tracks):
            dt = timestamp-track['time']
            expected = track['center']+track['velocity']*dt
            distances = [float(np.linalg.norm(center-expected)) for center, _ in groups]
            for gi, distance in enumerate(distances):
                gate = .22+(.1 if dt > .8 else .5)*dt
                # Do not reuse old identity when two present groups are
                # similarly plausible. Fresh confirmation is safer here.
                ambiguous = dt > .8 and any(abs(other-distance) < .05
                    for oi, other in enumerate(distances) if oi != gi)
                if distance <= gate and not ambiguous:
                    pairs.append((distance, ti, gi))
        used_tracks = set()
        used_groups = set()
        for _, ti, gi in sorted(pairs):
            if ti in used_tracks or gi in used_groups:
                continue
            used_tracks.add(ti)
            used_groups.add(gi)
            track = self.tracks[ti]
            center, group = groups[gi]
            dt = timestamp-track['time']
            velocity = (center-track['center'])/dt
            if np.linalg.norm(velocity) > .5:
                track['velocity'] = np.zeros(2)
                track['hits'] = 0
                track['confirmed_motion'] = False
            elif np.dot(velocity, track['velocity']) < 0 and np.linalg.norm(velocity) > .04:
                track['velocity'] = velocity
                track['hits'] = 0
                track['confirmed_motion'] = False
            elif dt > .8 and np.linalg.norm(velocity) < .04:
                # A stopped reappearance must not inherit an old moving
                # forecast merely through exponential velocity smoothing.
                track['velocity'] = np.zeros(2)
                track['hits'] = 0
                track['confirmed_motion'] = False
            else:
                track['velocity'] = .55*track['velocity']+.45*velocity
                if dt > .8:
                    self.reassociations += 1
            track.update(center=center, points=group, time=timestamp,
                         hits=track['hits']+1, clear_hits=0)
            if track['hits'] >= 3 and np.linalg.norm(track['velocity']) >= .04:
                track['confirmed_motion'] = True
        retained = []
        for ti, track in enumerate(self.tracks):
            if ti not in used_tracks:
                track['clear_hits'] = track.get('clear_hits', 0)+1 if self._observed_clear(track, timestamp) else 0
                if track['clear_hits'] >= 2:
                    self.clear_removals += 1
                    continue
            retained.append(track)
        self.tracks = retained
        for gi, (center, group) in enumerate(groups):
            if gi not in used_groups:
                self.tracks.append(dict(id=self.next_id, center=center, points=group,
                    velocity=np.zeros(2), time=timestamp, hits=1,
                    confirmed_motion=False, clear_hits=0))
                self.next_id += 1
        predicted = []
        for track in self.tracks:
            age = timestamp-track['time']
            if age > .8 or track['hits'] < 3 or np.linalg.norm(track['velocity']) < float(self.config.get('minimum_speed_m_s', .04)):
                continue
            self.moving_tracks += 1
            horizon = float(self.config.get('horizon_s', 2.4))
            for future in np.arange(.4, horizon+.001, .4):
                displaced = track['points']+track['velocity']*(age+future)
                uncertainty = float(self.config.get('uncertainty_m_s', .025))*(age+future)
                for offset in [[0, 0], [uncertainty, 0], [-uncertainty, 0], [0, uncertainty], [0, -uncertainty]]:
                    predicted.extend(displaced+offset)
        if predicted:
            self.predicted_points = np.asarray(predicted)
        return self.predicted_points
