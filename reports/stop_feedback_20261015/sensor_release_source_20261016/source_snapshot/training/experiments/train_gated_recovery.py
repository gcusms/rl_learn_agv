"""Same training environment/rewards; explicit frozen-base residual migration."""
import json
from pathlib import Path
import sys
import experiments.train_alignment_recovery as alignment
import experiments.train_quality_navigation as trainer
from experiments.gated_recovery_policy import warm_start_gated


def annotate_gated(components):
    alignment.annotate(components);alignment.annotate_alignment(components)
    path=Path(components)/'config.json'
    if path.exists():
        data=json.loads(path.read_text());data.update(clear_state_repair=False,
            frozen_base_actor=True,gated_residual_recovery=True,
            reference_strength=0.,reference_rank_rollback=False,
            residual_disabled_when='existing sensor-clear predicate and nonzero frozen-base winner',
            learning_rate=3e-4,critic_learning_rate=3e-5,
            reward_unchanged=data['training_reward_scale']==1.)
        path.write_text(json.dumps(data,indent=2))


def make_environment(components,*args,**kwargs):
    factory=alignment.make_environment(components,*args,**kwargs)
    annotate_gated(components)
    return factory


if __name__=='__main__':
    alignment.ALIGNMENT_COST=0.
    trainer.make_environment=make_environment
    trainer.RecoveryCoverageProgress=alignment.AlignmentProgress
    trainer.resume_quality=warm_start_gated
    output=Path(sys.argv[sys.argv.index('--output')+1])
    try:trainer.main()
    finally:annotate_gated(output/'candidate')
