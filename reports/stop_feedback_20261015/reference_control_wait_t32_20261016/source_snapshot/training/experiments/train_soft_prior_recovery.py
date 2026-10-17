"""Prepared focused PPO entry with explicit recovery prior/reward controls."""
import argparse
import json
from pathlib import Path
import sys
import experiments.train_alignment_recovery as alignment
import experiments.train_focused_recovery as focused
import experiments.train_quality_navigation as trainer
from experiments.soft_prior_recovery_policy import warm_start_soft_prior

PRIOR_SCALE=1/6


def resume(checkpoint,env,seed=92611):
    return warm_start_soft_prior(checkpoint,env,seed,prior_scale=PRIOR_SCALE)


def annotate(components):
    focused.annotate_focused(components)
    path=Path(components)/'config.json'
    if path.exists():
        config=json.loads(path.read_text());config.update(recovery_frozen_prior_scale=PRIOR_SCALE,
            initial_recovery_sampling_changed=PRIOR_SCALE!=1,
            sampling_and_ppo_and_export_use_same_logits=True)
        path.write_text(json.dumps(config,indent=2))


def make_environment(components,*args,**kwargs):
    factory=focused.make_environment(components,*args,**kwargs)
    annotate(components);return factory


if __name__=='__main__':
    parser=argparse.ArgumentParser(add_help=False)
    parser.add_argument('--alignment-cost',type=float,choices=(0.,3.),default=0.)
    parser.add_argument('--recovery-prior-scale',type=float,choices=(1.,1/6),default=1/6)
    options,remaining=parser.parse_known_args()
    alignment.ALIGNMENT_COST=options.alignment_cost;PRIOR_SCALE=options.recovery_prior_scale
    sys.argv=[sys.argv[0],*remaining]
    trainer.make_environment=make_environment
    trainer.RecoveryCoverageProgress=alignment.AlignmentProgress
    trainer.resume_quality=resume
    output=Path(remaining[remaining.index('--output')+1])
    try:trainer.main()
    finally:annotate(output/'candidate')
