"""One optimizer mechanism change; matched zero-cost alignment environment."""
import sys
import json
from pathlib import Path
import experiments.train_alignment_recovery as alignment
import experiments.quality_masked_navigation as quality
import experiments.train_quality_navigation as trainer
from experiments.backtracking_recovery_ppo import BacktrackingRecoveryPPO
from experiments.repairable_recovery_ppo import RepairableRecoveryPPO


if __name__=='__main__':
    control='--control' in sys.argv
    if control:sys.argv.remove('--control')
    alignment.ALIGNMENT_COST=0.
    model_class=RepairableRecoveryPPO if control else BacktrackingRecoveryPPO
    quality.QualityAnchoredPPO=model_class;trainer.QualityAnchoredPPO=model_class
    trainer.make_environment=alignment.make_environment
    trainer.RecoveryCoverageProgress=alignment.AlignmentProgress
    output=Path(sys.argv[sys.argv.index('--output')+1])
    try:trainer.main()
    finally:
        alignment.annotate(output/'candidate');alignment.annotate_alignment(output/'candidate')
        path=output/'candidate/config.json'
        if path.exists():
            data=json.loads(path.read_text());data['optimizer_actor_backtracking']=not control
            data['optimizer_critic_retained_on_actor_rejection']=not control
            data['rank_and_constraint_budget_unchanged']=True
            path.write_text(json.dumps(data,indent=2))
