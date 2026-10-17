"""Physical-age temporal assembly for the isolated forecast-alignment dataset."""
from collections import deque

import numpy as np


class CausalHistory:
    def __init__(self):
        self.reset()

    def reset(self):
        self.frames = deque()

    def append(self, frame, timestamp):
        frame = np.asarray(frame, np.float32)
        if frame.shape != (14,) or not np.isfinite(frame).all() or not np.isfinite(timestamp):
            raise ValueError('Finite history frame and sensor time required')
        if self.frames and timestamp <= self.frames[-1][0]:
            raise ValueError('Reset history before restarting or repeating sensor time')
        self.frames.append((float(timestamp), frame.copy()))
        while self.frames and self.frames[0][0] < timestamp-.451:
            self.frames.popleft()
        output = []
        for age in [.4, .2, 0.]:
            target = timestamp-age
            eligible = [value for value in self.frames if value[0] <= target+1e-8]
            sample = eligible[-1] if eligible else None
            output.append(np.zeros(14, np.float32) if sample is None or target-sample[0] > .05+1e-8
                          else sample[1].copy())
        return np.concatenate(output)
