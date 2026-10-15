"""Fixed 48-task evaluator explicitly using the trained time-consistent core."""
import experiments.evaluate_quality_regression as evaluator
from experiments.time_consistent_tracking import TimeConsistentFreshRuntime
import argparse
from pathlib import Path
import shutil
import sys


if __name__=='__main__':
    parser=argparse.ArgumentParser(add_help=False)
    parser.add_argument('--provenance',type=Path)
    options,remaining=parser.parse_known_args()
    sys.argv=[sys.argv[0],*remaining]
    output=Path(remaining[remaining.index('--output')+1])
    evaluator.QualitySensorNavigationRuntime=TimeConsistentFreshRuntime
    try:
        evaluator.main()
    finally:
        if options.provenance is not None and output.is_dir():
            shutil.copy2(options.provenance/'SOURCE_PROVENANCE.json',output/'SOURCE_PROVENANCE.json')
            shutil.copytree(options.provenance/'source_snapshot',output/'source_snapshot')
