"""Three causal samples at physical 0.2s offsets, independent of call rate."""
from collections import deque
import numpy as np


class TimedScanHistory:
    def __init__(self,period=.2):self.period=period;self.samples=deque()
    def reset(self):self.samples.clear()
    def update(self,timestamp,scan):
        if self.samples and timestamp<self.samples[-1][0]:raise ValueError('Backward history timestamp')
        sample=(float(timestamp),np.asarray(scan).copy())
        if self.samples and timestamp==self.samples[-1][0]:self.samples[-1]=sample
        else:self.samples.append(sample)
        while len(self.samples)>1 and self.samples[1][0]<timestamp-3*self.period:self.samples.popleft()
        history=[]
        for target in (timestamp-2*self.period,timestamp-self.period,timestamp):
            selected=self.samples[0][1]
            for t,value in self.samples:
                if t>target+1e-7:break
                selected=value
            history.append(selected.copy())
        return history
