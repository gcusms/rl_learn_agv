"""Opt-in repairable clear-state constraint and masked recovery exploration."""
import json
from pathlib import Path
import sys
import experiments.quality_masked_navigation as quality
import experiments.train_quality_navigation as trainer
from experiments.train_recovery_temperature import make_environment,annotate
from experiments.repairable_recovery_ppo import RepairableRecoveryPPO


if __name__=='__main__':
    output=Path(sys.argv[sys.argv.index('--output')+1])
    quality.QualityAnchoredPPO=RepairableRecoveryPPO
    trainer.QualityAnchoredPPO=RepairableRecoveryPPO
    trainer.make_environment=make_environment
    try:trainer.main()
    finally:
        annotate(output/'candidate')
        path=output/'candidate/config.json'
        if path.exists():
            config=json.loads(path.read_text())
            config.update(clear_state_repair='training-cache-only gradient repair; failed repair restores actor',
                repair_preserves_adam_state=True,repair_preserves_critic=True)
            path.write_text(json.dumps(config,indent=2))
