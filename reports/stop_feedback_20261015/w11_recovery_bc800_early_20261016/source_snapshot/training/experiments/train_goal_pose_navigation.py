"""Conservative opt-in goal-pose PPO fine-tuning; never auto-promotes a model."""
import argparse,hashlib,json,resource,shutil,time
from pathlib import Path
import numpy as np
import torch
from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import BaseCallback
from experiments.masked_navigation import MaskedNavigationCurriculum
from experiments.quality_masked_navigation import QualityMaskedWrapper
from experiments.goal_pose_navigation import GoalPoseWrapper,migrate_goal_policy,export_goal_policy
from neural_navigation.odom_anchored_localizer import contract as localization_contract
from neural_navigation.goal_pose import contract as goal_contract
from neural_navigation.experiment_budget import ExperimentBudget


class ConservativeProgress(BaseCallback):
    def __init__(self,output,quota):
        super().__init__();self.output=output;self.quota=quota
        self.started=time.monotonic();self.cpu_started=time.process_time();self.episodes=[]
        self.budget=ExperimentBudget(quota,directory=output);self.counts={}
    def _on_step(self):
        for info,done in zip(self.locals['infos'],self.locals['dones']):
            category='alignment' if json.loads(info['training_case_json']).get('pose_alignment_task') else 'navigation'
            count=self.counts.setdefault(category,dict(steps=0,wait_with_motion=0,only_stop=0,turns=0,heading_reward=0.))
            count['steps']+=1;count['wait_with_motion']+=int(info.get('masked_chose_wait_with_nonzero_available',False))
            count['only_stop']+=int(info.get('masked_current_nonzero_choices',0)==0)
            count['turns']+=int(abs(info['commanded_velocity'][1])>.01)
            count['heading_reward']+=float(info.get('goal_heading_potential_reward',0.))
            if done:self.episodes.append({k:info.get(k) for k in ('verified_mission_success','collision_events','true_goal_distance_m','true_goal_yaw_error_rad','training_case_json')})
        self.budget.tick()
        if self.num_timesteps%25==0:
            self._write('training')
        return True
    def _write(self,phase):
        (self.output/'progress.json').write_text(json.dumps(dict(phase=phase,steps=self.num_timesteps,
            seconds=time.monotonic()-self.started,peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
            training_episodes=self.episodes,sampling=self.counts,resource=self.budget.statistics(),
            ppo_statistics={k:float(v) for k,v in self.model.logger.name_to_value.items() if k.startswith('train/') and np.isscalar(v)},
            scope='training diagnostics, not independent acceptance'),indent=2))


def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--steps',type=int,default=250);p.add_argument('--seed',type=int,default=101401)
    p.add_argument('--cpu-cores',type=float,default=.5);p.add_argument('--stage',choices=['static','dynamic','robust'],default='static')
    p.add_argument('--odom-anchored',action='store_true',help='Separate opt-in factor; run baseline localization without this flag for the heading-only ablation')
    p.add_argument('--alignment-probability',type=float,default=.5)
    p.add_argument('--alignment-max-steps',type=int,default=100)
    p.add_argument('--balanced-alignment',action='store_true',help='Alternate alignment and navigation episodes, beginning with alignment')
    a=p.parse_args()
    if a.output.exists() or a.steps<=0 or a.steps%250 or not 0<a.cpu_cores<=1:p.error('Fresh output, positive multiple of250 and quota at most one core required')
    a.output.mkdir(parents=True);components=a.output/'components';components.mkdir()
    for name in ('hardware.json','router.pt','localizer.pt'):shutil.copy2(a.checkpoint.parent/name,components/name)
    if a.odom_anchored:(components/'localization_contract.json').write_text(json.dumps(localization_contract(),indent=2))
    (components/'goal_pose_observation.json').write_text(json.dumps(goal_contract(),indent=2))
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    old=MaskablePPO.load(a.checkpoint,device='cpu');scale=float(getattr(old,'training_reward_scale',1.));gamma=old.gamma;del old
    base=MaskedNavigationCurriculum(components,stage=a.stage,route_mode='shortcut',max_steps=700,footprint_clearance_cost=1.)
    env=GoalPoseWrapper(QualityMaskedWrapper(base,reward_scale=scale),gamma=gamma,
        alignment_probability=a.alignment_probability,alignment_max_steps=a.alignment_max_steps,balanced_alignment=a.balanced_alignment)
    config=dict(parent_sha256=hashlib.sha256(a.checkpoint.read_bytes()).hexdigest(),seed=a.seed,steps=a.steps,goal_contract=goal_contract(),
        localizer_router_weights_frozen=True,localization_gate_enabled=a.odom_anchored,stage=a.stage,
        num_envs=1,torch_threads=1,soft_average_cpu_cores=a.cpu_cores,reward_scale=scale,deployment_approved=False,
        reference_strength=2.,reference_kl_budget=.02,reference_excludes_heading_required_within_m=.8,
        alignment_probability=a.alignment_probability,alignment_max_steps=a.alignment_max_steps,
        balanced_alignment=a.balanced_alignment,
        scope='bounded training-entry smoke test' if a.steps==250 else 'fine-tuning experiment; independent evaluation required')
    source=Path(__file__).resolve().parents[2]
    config['source_sha256']={str(p.relative_to(source)):hashlib.sha256(p.read_bytes()).hexdigest()
        for subdir in ('training/neural_navigation','training/experiments') for p in (source/subdir).glob('*.py')}
    (a.output/'config.json').write_text(json.dumps(config,indent=2));callback=ConservativeProgress(a.output,a.cpu_cores)
    try:
        model=migrate_goal_policy(a.checkpoint,env,seed=a.seed);model.save(a.output/'policy_initial.zip')
        export_goal_policy(model,components,a.output/'initial_portable')
        while model.num_timesteps<a.steps:
            model.learn(min(500,a.steps-model.num_timesteps),callback=callback,reset_num_timesteps=model.num_timesteps==0)
            model.save(a.output/f'policy_{model.num_timesteps}.zip')
        model.save(a.output/'policy.zip')
        config['reference_statistics']=model.last_reference_statistics
        (a.output/'config.json').write_text(json.dumps(config,indent=2))
        export_goal_policy(model,components,a.output/'portable');callback._write('complete')
    except Exception as error:
        (a.output/'ERROR.txt').write_text(repr(error));raise
    finally:env.close()

if __name__=='__main__':main()
