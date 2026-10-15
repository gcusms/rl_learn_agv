"""Prepared focused recovery PPO entry; does not affect running broad trials."""
import argparse
import json
from pathlib import Path
import sys
import experiments.train_alignment_recovery as alignment
import experiments.train_gated_recovery as gated
import experiments.train_quality_navigation as trainer
from experiments.focused_recovery_policy import warm_start_focused


def annotate_focused(components):
    gated.annotate_gated(components)
    path=Path(components)/'config.json'
    if path.exists():
        config=json.loads(path.read_text());config.update(focused_recovery_support=True,
            residual_disabled_when='no trusted behind waypoint, no previous non-goal stop, and no previous idle wait; or no legal nonzero action',
            low_pose_previous_wait_remains_trainable=True,
            no_fixed_maneuver=True)
        path.write_text(json.dumps(config,indent=2))


def make_environment(components,*args,**kwargs):
    factory=gated.make_environment(components,*args,**kwargs)
    annotate_focused(components)
    return factory


if __name__=='__main__':
    parser=argparse.ArgumentParser(add_help=False)
    parser.add_argument('--alignment-cost',type=float,choices=(0.,3.),default=0.)
    options,remaining=parser.parse_known_args();alignment.ALIGNMENT_COST=options.alignment_cost
    sys.argv=[sys.argv[0],*remaining]
    trainer.make_environment=make_environment
    trainer.RecoveryCoverageProgress=alignment.AlignmentProgress
    trainer.resume_quality=warm_start_focused
    output=Path(remaining[remaining.index('--output')+1])
    try:trainer.main()
    finally:annotate_focused(output/'candidate')
