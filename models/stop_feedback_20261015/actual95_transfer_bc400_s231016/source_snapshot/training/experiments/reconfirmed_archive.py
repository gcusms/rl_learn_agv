"""Experimental archive with fresh confirmation after occluded reacquisition."""
import weakref
from experiments.visibility_archive import VisibilityArchivePredictor


class ReconfirmedArchivePredictor(VisibilityArchivePredictor):
    def __init__(self, config, core, archive_s=4.):
        super().__init__(config, core, archive_s)
        # A runtime core owns its predictor. Do not create an owning cycle.
        try:
            self.core = weakref.proxy(core)
        except TypeError:
            # Lightweight non-owning test fixtures may not support weakrefs.
            self.core = core

    def reset(self):
        super().reset()
        self.reconfirmation_resets = 0

    def update(self, points, timestamp, alignment_supported=True):
        if self.last_time is not None and timestamp <= self.last_time:
            return self.predicted_points
        for track in self.tracks:
            if timestamp-track['time'] > .8 and track['hits'] >= 3:
                # Retain identity and last velocity for association, but do
                # not instantly resurrect a confident route forecast. The
                # reacquisition raises hits to 2; one more supported scan
                # raises it to 3. Reversal/stop gates still reset to 1.
                track['hits'] = 1
                self.reconfirmation_resets += 1
        return super().update(points, timestamp, alignment_supported)
