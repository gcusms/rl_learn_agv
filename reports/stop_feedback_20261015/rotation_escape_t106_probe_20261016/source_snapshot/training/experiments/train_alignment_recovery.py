"""Matched0/3 alignment-cost PPO trials with the same confirmed-free memory."""
import argparse
import json
from pathlib import Path
import sys
from stable_baselines3.common.monitor import Monitor
import experiments.quality_masked_navigation as quality
import experiments.train_quality_navigation as trainer
from experiments.train_recovery_temperature import make_environment as recovery_environment,annotate
from experiments.repairable_recovery_ppo import RepairableRecoveryPPO
from experiments.recovery_sampling import RecoveryCoverageProgress
from experiments.alignment_reward import AlignmentReward

ALIGNMENT_COST=0.


def annotate_alignment(components):
    path=Path(components)/'config.json'
    if path.exists():
        config=json.loads(path.read_text());config.update(alignment_cost=ALIGNMENT_COST,
            native_reward_unchanged=ALIGNMENT_COST==0.,reward_unchanged=ALIGNMENT_COST==0. and config['training_reward_scale']==1.,
            alignment_cost_inputs='pre-action trusted path heading and actual executed forward velocity',
            memory_free_evidence=dict(margin_m=.03,frames=3,max_gap_s=.4),
            clear_state_repair=True,alignment_cost_changes_actions=False)
        path.write_text(json.dumps(config,indent=2))


class AlignmentProgress(RecoveryCoverageProgress):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.alignment=dict(eligible_frames=0,forward_motion_frames=0,native_cost_sum=0.)

    def _on_step(self):
        for info in self.locals['infos']:
            self.alignment['eligible_frames']+=int(info['alignment_context_eligible'])
            self.alignment['forward_motion_frames']+=int(info['alignment_forward_motion'])
            self.alignment['native_cost_sum']+=info['alignment_forward_cost']
        result=super()._on_step()
        if self.num_timesteps%250==0:(self.output/'alignment_coverage.json').write_text(json.dumps(self.alignment,indent=2))
        return result


def make_environment(components,templates,cases,seed,monitor=False,reward_scale=1.,scheduled_recovery=False,core_factory=None):
    hardware=Path(components)/'hardware.json';data=json.loads(hardware.read_text())
    data.setdefault('near_field_memory',{}).update(free_evidence_margin_m=.03,free_evidence_frames=3,free_evidence_max_gap_s=.4)
    hardware.write_text(json.dumps(data,indent=2))
    factory=recovery_environment(components,templates,cases,seed,False,reward_scale,scheduled_recovery,core_factory)
    annotate_alignment(components)
    coefficient=ALIGNMENT_COST
    def create():
        env=factory()
        if not monitor:return env
        return Monitor(AlignmentReward(env,coefficient),str(components/'worker_0.csv'),info_keywords=(
            'verified_mission_success','collision_events','safety_termination','training_case_json'))
    return create


if __name__=='__main__':
    parser=argparse.ArgumentParser(add_help=False);parser.add_argument('--alignment-cost',type=float,choices=(0.,3.),required=True)
    options,remaining=parser.parse_known_args();ALIGNMENT_COST=options.alignment_cost
    sys.argv=[sys.argv[0],*remaining];output=Path(remaining[remaining.index('--output')+1])
    quality.QualityAnchoredPPO=RepairableRecoveryPPO;trainer.QualityAnchoredPPO=RepairableRecoveryPPO
    trainer.make_environment=make_environment;trainer.RecoveryCoverageProgress=AlignmentProgress
    try:trainer.main()
    finally:
        annotate(output/'candidate');annotate_alignment(output/'candidate')
