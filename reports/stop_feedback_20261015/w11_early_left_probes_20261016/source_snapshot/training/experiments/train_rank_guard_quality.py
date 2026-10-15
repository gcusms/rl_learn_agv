"""Explicit opt-in training entry for clear-state action-rank protection."""
import json
from pathlib import Path
import sys
import experiments.quality_masked_navigation as quality
import experiments.train_quality_navigation as trainer
from experiments.greedy_stable_quality_ppo import GreedyStableQualityPPO


if __name__=='__main__':
    output=Path(sys.argv[sys.argv.index('--output')+1])
    quality.QualityAnchoredPPO=GreedyStableQualityPPO
    trainer.QualityAnchoredPPO=GreedyStableQualityPPO
    try:
        trainer.main()
    finally:
        path=output/'candidate/config.json'
        if path.exists():
            config=json.loads(path.read_text())
            config.update(reference_protection='clear_state_greedy_rank_and_mean_kl',
                rank_guard_margin_fraction=.25,reference_constraint_loss_is_not_pure_kl=True)
            path.write_text(json.dumps(config,indent=2))
