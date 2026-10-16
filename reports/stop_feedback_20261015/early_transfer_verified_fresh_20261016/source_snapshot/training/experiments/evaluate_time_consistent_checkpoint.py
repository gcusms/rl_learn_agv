"""Frozen intermediate checkpoint evaluated with its declared sensor core."""
from functools import partial
import experiments.evaluate_quality_checkpoint as evaluator
from experiments.train_quality_navigation import export,evaluate
from experiments.time_consistent_tracking import TimeConsistentCore


if __name__=='__main__':
    evaluator.export=partial(export,sensor_core='time-consistent')
    evaluator.evaluate=partial(evaluate,core_factory=TimeConsistentCore)
    evaluator.main()
