"""Experimental positive-free clearing, used only in recorded-data shadows.

Preserve original strong-ray clearing and add three consecutive finite-ray
confirmations for a smaller measured discrepancy. Unknown, occluded, blind or
old observations never establish free space. No deployed memory is modified.
"""
import numpy as np

from neural_navigation.safety_memory import NearFieldMemory


class EvidenceMemory(NearFieldMemory):
    def __init__(self, config, margin=.03, frames=3, max_gap=.4):
        assert np.isfinite([margin, max_gap]).all()
        assert 0 < margin < float(config.get('free_evidence_margin_m', .1))
        assert isinstance(frames, int) and frames >= 3 and max_gap > 0
        self.evidence_margin = float(margin)
        self.evidence_frames = frames
        self.evidence_max_gap = float(max_gap)
        super().__init__(config)

    def reset(self):
        super().reset()
        self.stable_free = {}
        self.last_positive_clears = []

    def update(self, core):
        self.last_positive_clears = []
        now = float(core.elapsed)
        scan = np.asarray(core.scan)
        spec = core.lidar
        valid = np.isfinite(scan) & (scan >= spec.range_min) & (scan < spec.range_max)
        keys = list(self.entries)
        if keys:
            pose = np.asarray(core.odom)
            c, s = np.cos(pose[2]), np.sin(pose[2])
            body = (np.array([self.entries[key][0] for key in keys]) - pose[:2]) @ np.array([[c, -s], [s, c]])
            mc, ms = np.cos(spec.mount[3]), np.sin(spec.mount[3])
            points = (body - spec.mount[:2]) @ np.array([[mc, -ms], [ms, mc]])
            distance = np.linalg.norm(points, axis=1)
            bearing = np.arctan2(points[:, 1], points[:, 0])
            index = np.clip(np.searchsorted(spec.angles, bearing), 0, len(scan) - 1)
            ranges = np.where(valid, scan, 0.)
            observed = np.minimum.reduce([ranges[np.clip(index + offset, 0, len(scan) - 1)] for offset in (-1, 0, 1)])
            evidence = ((bearing >= spec.angle_min) & (bearing <= spec.angle_max)
                & (observed > distance + self.evidence_margin))
            for offset, key in enumerate(keys):
                point, seen = self.entries[key]
                token = (key, float(seen))
                if not evidence[offset]:
                    self.stable_free.pop(token, None)
                    continue
                previous, count = self.stable_free.get(token, (now, 0))
                if now - previous > self.evidence_max_gap + 1e-8:
                    count = 0
                if count == 0 or now > previous + 1e-8:
                    count += 1
                self.stable_free[token] = (now, count)
                if count >= self.evidence_frames:
                    self.last_positive_clears.append(dict(key=list(key), seen_s=float(seen),
                        sensor_time_s=now, confirmations=count, observed_min_range_m=float(observed[offset]),
                        retained_point_range_m=float(distance[offset]),
                        range_excess_m=float(observed[offset] - distance[offset]),
                        bearing_in_supported_fov=True, finite_three_neighbor_rays=True))
                    self.entries.pop(key)
                    self.free_confirmations.pop(key, None)
        # Keep all original strong-evidence, distance, insertion, capacity and
        # age-uncertainty rules. Extra clearing precedes the unchanged update.
        result = super().update(core)
        current_tokens = {(key, float(value[1])) for key, value in self.entries.items()}
        self.stable_free = {key: value for key, value in self.stable_free.items() if key in current_tokens}
        return result
