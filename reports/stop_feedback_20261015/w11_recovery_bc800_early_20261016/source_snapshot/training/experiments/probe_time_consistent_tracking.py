"""Reuse exactly the four diagnostic cases with time-consistent fresh tracking."""
import experiments.probe_isolated_frame_prediction as probe
from experiments.time_consistent_tracking import TimeConsistentFreshRuntime


if __name__=='__main__':
    # The shared driver records the selected variant via its runtime symbol.
    probe.IsolatedFramePredictionRuntime=TimeConsistentFreshRuntime
    probe.main()
