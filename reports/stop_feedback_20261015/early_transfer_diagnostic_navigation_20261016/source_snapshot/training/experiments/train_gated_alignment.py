"""Matched gated-residual trial with the previously tested forward cost.

Only the reward coefficient differs from train_gated_recovery. The same
constructor gives identical initial network and Adam states under the seed.
"""
from pathlib import Path
import sys
import experiments.train_alignment_recovery as alignment
import experiments.train_gated_recovery as gated
import experiments.train_quality_navigation as trainer
from experiments.gated_recovery_policy import warm_start_gated


if __name__=='__main__':
    alignment.ALIGNMENT_COST=3.
    trainer.make_environment=gated.make_environment
    trainer.RecoveryCoverageProgress=alignment.AlignmentProgress
    trainer.resume_quality=warm_start_gated
    output=Path(sys.argv[sys.argv.index('--output')+1])
    try:trainer.main()
    finally:gated.annotate_gated(output/'candidate')
