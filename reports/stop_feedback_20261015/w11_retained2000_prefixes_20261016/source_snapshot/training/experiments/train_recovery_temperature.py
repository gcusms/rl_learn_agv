"""Opt-in masked exploration with guarded pure-rotation recovery."""
import json
from pathlib import Path
import sys
import experiments.quality_masked_navigation as quality
import experiments.train_quality_navigation as trainer
from experiments.recovery_temperature_policy import RecoveryTemperaturePPO

native_make_environment=trainer.make_environment


def annotate(components):
    config_path=Path(components)/'config.json'
    if config_path.exists():
        config=json.loads(config_path.read_text())
        config.update(guard_unchanged=False,guard_change='existing clearance-increasing pure rotation enabled',
            recovery_temperature=3.,recovery_maximum_choices=3,
            exploration_applies_to='sampling, PPO log probability and entropy',
            reference_protection='clear_state_greedy_rank_and_mean_kl',
            reference_constraint_loss_is_not_pure_kl=True)
        config_path.write_text(json.dumps(config,indent=2))


def make_environment(components,*args,**kwargs):
    hardware=Path(components)/'hardware.json'
    data=json.loads(hardware.read_text())
    data.setdefault('near_field_memory',{})['clearance_increasing_rotation']=True
    hardware.write_text(json.dumps(data,indent=2))
    annotate(components)
    return native_make_environment(components,*args,**kwargs)


if __name__=='__main__':
    quality.QualityAnchoredPPO=RecoveryTemperaturePPO
    trainer.QualityAnchoredPPO=RecoveryTemperaturePPO
    trainer.make_environment=make_environment
    output=Path(sys.argv[sys.argv.index('--output')+1])
    try:trainer.main()
    finally:annotate(output/'candidate')
