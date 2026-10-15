"""Soft CPU throttling and resource checks for bounded local experiments."""
import resource
import shutil
import time
from pathlib import Path


def peak_rss_mib():
    # On Linux, ru_maxrss may retain the parent's pre-exec high water mark.
    # VmHWM belongs to this executable, like the native AMCL measurement.
    try:
        for line in Path('/proc/self/status').read_text().splitlines():
            if line.startswith('VmHWM:'):return float(line.split()[1])/1024
    except OSError:pass
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024


class ExperimentBudget:
    def __init__(self,cores=.5,memory_mib=4096,directory='.'):
        if not 0<cores<=1:raise ValueError('At most one CPU core for this experiment')
        self.cores=float(cores);self.memory_mib=memory_mib;self.directory=Path(directory)
        self.started=time.monotonic();self.cpu_started=time.process_time()
    def tick(self):
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024>self.memory_mib and peak_rss_mib()>self.memory_mib:
            raise RuntimeError('Experiment exceeded its conservative RSS budget')
        if shutil.disk_usage(self.directory).free<512*1024**2:
            raise RuntimeError('Less than 512MiB disk remains; stop without deleting user files')
        delay=(time.process_time()-self.cpu_started)/self.cores-(time.monotonic()-self.started)
        # Timing the limiter consumes CPU too. Chasing a sub-millisecond
        # positive remainder can feed itself with thousands of tiny sleeps,
        # especially at conservative quotas. A 2ms deadband avoids that loop;
        # the global average budget catches up on the following tick.
        while delay>.002:
            time.sleep(min(.2,delay))
            delay=(time.process_time()-self.cpu_started)/self.cores-(time.monotonic()-self.started)
    def statistics(self):
        wall=time.monotonic()-self.started;cpu=time.process_time()-self.cpu_started
        return dict(seconds=wall,cpu_seconds=cpu,average_cpu_cores=cpu/max(wall,1e-9),
                    peak_rss_mib=peak_rss_mib(),
                    resource_limits='single-thread CPU soft average quota; RSS/disk checked between steps')
