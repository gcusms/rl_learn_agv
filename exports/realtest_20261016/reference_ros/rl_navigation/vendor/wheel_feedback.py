"""Read-only v1 wheel telemetry; no ROS/serial imports, offline testable."""
from dataclasses import dataclass
import math
import struct

SUBTYPE, VERSION, SIZE = 0x70, 1, 116
HEADER = struct.Struct('<4B4I')
WHEEL = struct.Struct('<iIfffI')

@dataclass(frozen=True)
class Feedback:
    sequence: int
    timestamp_ms: int
    window_ms: int
    drops: int
    flags: int
    delta: tuple
    total: tuple
    raw_rps: tuple
    filtered_rps: tuple
    target_rps: tuple
    cpr: tuple

def decode(payload):
    if len(payload) != SIZE:
        raise ValueError('wheel feedback length')
    sub, ver, count, flags, seq, tick, window, drops = HEADER.unpack_from(payload)
    if (sub, ver, count) != (SUBTYPE, VERSION, 4) or not 1 <= window <= 1000:
        raise ValueError('wheel feedback version/count/window')
    wheels = [WHEEL.unpack_from(payload, 20+24*i) for i in range(4)]
    for delta, _, raw, filtered, target, cpr in wheels:
        if not 1 <= cpr <= 1000000 or not all(math.isfinite(x) for x in (raw, filtered, target)):
            raise ValueError('wheel feedback CPR/nonfinite speed')
        if abs(raw - delta*1000.0/(window*cpr)) > max(1e-5, abs(raw)*1e-5):
            raise ValueError('wheel feedback raw/count inconsistency')
    cols = list(zip(*wheels))
    return Feedback(seq, tick, window, drops, flags, *cols)

def signed_diff(now, before):
    return ((now-before+0x80000000) & 0xffffffff)-0x80000000

class Tracker:
    """Drop duplicate/stale frames; reset on MCU reboot, not counter wrap.

    Sequence/timestamp wrap supported. On this ordered UART stream, MCU reboot
    shows both counters regressing and either a startup sequence or a timestamp
    rollback over 1 s. The first
    reset frame is diagnostic only; the following frame resumes measurement.
    No values are extrapolated through a gap or after reception stops.
    """
    def __init__(self):
        self.last = None
        self.lost = self.resets = self.rejected = 0

    def accept(self, f):
        if self.last is not None:
            ds = signed_diff(f.sequence, self.last.sequence)
            dt = signed_diff(f.timestamp_ms, self.last.timestamp_ms)
            if ds <= 0 or dt <= 0:
                if ds < 0 and dt <= 0 and (f.sequence <= 5 or dt < -1000):
                    self.resets += 1
                    self.last = f
                    return False
                self.rejected += 1
                return False
            self.lost += max(0, ds-1)
        self.last = f
        return True

def body_velocity(rps, wheel_diameter, wheelbase, track_width,
                  signs=(1, 1, -1, -1), order=(0, 1, 2, 3), chassis='mecanum'):
    if not all(math.isfinite(x) and x > 0 for x in (wheel_diameter, wheelbase, track_width)):
        raise ValueError('positive chassis dimensions required')
    if len(rps) != 4 or len(signs) != 4 or sorted(order) != [0, 1, 2, 3] or any(s not in (-1, 1) for s in signs):
        raise ValueError('invalid wheel order/signs')
    v = [rps[order[i]]*signs[i]*math.pi*wheel_diameter for i in range(4)]
    vx = sum(v)/4
    if chassis == 'mecanum':
        k = (wheelbase+track_width)/2
        return vx, (-v[0]+v[1]+v[2]-v[3])/4, (-v[0]-v[1]+v[2]+v[3])/(4*k)
    if chassis == 'differential':
        return vx, 0.0, (-v[0]-v[1]+v[2]+v[3])/(2*track_width)
    raise ValueError('unsupported chassis')

class Window:
    """MCU-time count window, using cumulative counts to survive lost frames."""
    def __init__(self, minimum_ms=20):
        if not 10 <= minimum_ms <= 1000:
            raise ValueError('invalid measurement window')
        self.minimum_ms = minimum_ms
        self.start = None

    def update(self, f):
        if self.start is None:
            self.start = f
            return None
        dt = signed_diff(f.timestamp_ms, self.start.timestamp_ms)
        if dt <= 0 or dt > 1000 or f.cpr != self.start.cpr:
            self.start = f
            return None
        if dt < self.minimum_ms:
            return None
        speeds = tuple(signed_diff(a,b)*1000/(dt*c) for a,b,c in zip(f.total,self.start.total,f.cpr))
        self.start = f
        return speeds, dt

class EncoderRest:
    """Observed four-wheel count rest; never a claim about ground truth/slip.

    Every observed cumulative counter must stay unchanged for 500 ms, not only
    the two endpoints. Gaps <=50 ms are tolerated but hidden reversals remain
    unobservable. A larger gap, reset, nonzero target or CPR change resets it.
    Caller must separately reject stale host reception.
    """
    def __init__(self, duration_ms=500, maximum_gap_ms=50):
        self.duration_ms = duration_ms
        self.maximum_gap_ms = maximum_gap_ms
        self.reset()

    def reset(self):
        self.last = None
        self.since = None
        self.max_gap = 0

    def update(self, f):
        changed = self.last is not None and f.total != self.last.total
        gap = None if self.last is None else signed_diff(f.timestamp_ms, self.last.timestamp_ms)
        discontinuity = self.last is not None and (gap <= 0 or gap > self.maximum_gap_ms or f.cpr != self.last.cpr)
        if discontinuity or changed:
            self.since = None
            self.max_gap = 0
        if any(target != 0.0 for target in f.target_rps):
            self.since = None
            self.max_gap = 0
        elif self.since is None:
            self.since = f.timestamp_ms
        elif gap is not None:
            self.max_gap = max(self.max_gap, gap)
        elapsed = 0 if self.since is None else max(0, signed_diff(f.timestamp_ms, self.since))
        self.last = f
        return elapsed >= self.duration_ms, elapsed, self.max_gap
