"""Opt-in experiment: preserve immediate strong evidence plus repeated weak evidence.

This is not installed into the default runtime. Both rules use only finite
measured rays in continuous input odometry, with the original range, uncertainty
and point-cap behavior. No elapsed-time, map truth or unknown-return clearing.
"""
import numpy as np

from neural_navigation.safety_memory import NearFieldMemory


class HybridFreeEvidenceMemory:
    def __init__(self, config):
        self.config = dict(config)
        self.large_margin = float(config.get('hybrid_immediate_margin_m', .1))
        self.small_margin = float(config.get('hybrid_confirmed_margin_m', .03))
        self.frames = config.get('hybrid_confirmed_frames', 3)
        self.max_gap = float(config.get('hybrid_confirmed_max_gap_s', .4))
        self.maximum = config.get('max_points', 4096)
        if (not np.isfinite([self.large_margin, self.small_margin, self.max_gap]).all()
                or not 0. < self.small_margin < self.large_margin
                or not isinstance(self.frames, int) or self.frames < 2
                or self.max_gap <= 0. or not isinstance(self.maximum, int) or self.maximum < 1):
            raise ValueError('Ordered positive margins, repeated frames, gap and point cap required')
        self.strong = NearFieldMemory(dict(config, free_evidence_margin_m=self.large_margin,
            free_evidence_frames=1, free_evidence_max_gap_s=self.max_gap))
        self.confirmed = NearFieldMemory(dict(config, free_evidence_margin_m=self.small_margin,
            free_evidence_frames=self.frames, free_evidence_max_gap_s=self.max_gap))
        self.reset()

    def reset(self):
        self.strong.reset()
        self.confirmed.reset()
        self.entries = {}
        self.free_confirmations = {}
        self.last_time = None

    def update(self, core):
        # Both rules start from exactly the same retained endpoints. Defer
        # point-cap eviction until after their evidence is combined: otherwise
        # independent nearest-point pruning could masquerade as free evidence.
        worker_limit = self.maximum + len(core.scan)
        for worker in (self.strong, self.confirmed):
            worker.entries = self.entries.copy()
            worker.config['max_points'] = worker_limit
            worker.free_confirmations = {k:v for k,v in worker.free_confirmations.items()
                                         if k in self.entries}
            worker.update(core)
        self.entries = {k:v for k,v in self.strong.entries.items() if k in self.confirmed.entries}
        keys = list(self.entries)
        now, pose = float(core.elapsed), np.asarray(core.odom)
        if keys:
            c, s = np.cos(pose[2]), np.sin(pose[2])
            points = np.array([self.entries[k][0] for k in keys])
            times = np.array([self.entries[k][1] for k in keys])
            body = (points-pose[:2]) @ np.array([[c,-s],[s,c]])
            if len(keys) > self.maximum:
                selected = np.argsort(np.linalg.norm(body, axis=1))[:self.maximum]
                self.entries = {keys[int(i)]:self.entries[keys[int(i)]] for i in selected}
                body, times = body[selected], times[selected]
            uncertainty = np.minimum(float(self.config.get('max_uncertainty_m', .04)),
                np.maximum(now-times, 0.) * float(self.config.get('uncertainty_m_s', .002)))
        else:
            body, uncertainty = np.empty((0,2)), np.empty(0)
        self.free_confirmations = {k:v for k,v in self.confirmed.free_confirmations.items()
                                   if k in self.entries}
        self.confirmed.free_confirmations = self.free_confirmations.copy()
        for worker in (self.strong, self.confirmed):
            worker.entries = self.entries.copy()
        self.last_time = now
        return body, uncertainty


def contract():
    return dict(schema='hybrid_finite_free_evidence_diagnostic_v1',
        immediate_finite_margin_m=.1, confirmed_finite_margin_m=.03,
        confirmation_distinct_frames=3, maximum_confirmation_gap_s=.4,
        free_evidence_combination='original_strong_single_frame_OR_repeated_weaker_finite_rays',
        current_hits_replace_cell_and_reset_confirmations=True,
        uncertainty_footprint_blind_sector_range_and_forget_radius_unchanged=True,
        point_cap_applied_only_after_evidence_combination=True,
        unknown_and_no_return_are_not_free=True, time_alone_does_not_clear=True,
        not_enabled_in_default_runtime=True, not_physically_validated=True)
