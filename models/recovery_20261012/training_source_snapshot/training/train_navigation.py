"""Train PPO on localized partial-FOV navigation; no ground-truth observations."""
import argparse
import json
from pathlib import Path
import torch
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback, CallbackList
from stable_baselines3.common.monitor import Monitor
from navrl.physical_env import make_navigation_env
from navrl.multimap import MultiMapNavigationEnv
from navrl.geometry import ROOT


class DemonstrationAnchor(BaseCallback):
    """Retain demonstrated waiting/turning while PPO improves its on-policy return."""
    def __init__(self,path,seed):
        super().__init__(); d=np.load(path)
        self.x=torch.as_tensor(d['observations'],dtype=torch.float32)
        self.y=torch.as_tensor(d['actions'],dtype=torch.float32); self.rng=np.random.default_rng(seed)
    def _on_step(self): return True
    def _on_rollout_start(self):
        for _ in range(2):
            idx=self.rng.integers(0,len(self.x),128)
            mean=self.model.policy.get_distribution(self.x[idx]).distribution.mean
            loss=.5*torch.nn.functional.mse_loss(mean,self.y[idx])
            self.model.policy.optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.policy.parameters(),.5)
            self.model.policy.optimizer.step()
        self.logger.record('train/demonstration_anchor_loss',float(loss.detach()))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--localizer',choices=['ros-amcl','particle'],default='ros-amcl')
    parser.add_argument('--steps',type=int,default=100000)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--learning-rate',type=float)
    parser.add_argument('--demo-anchor',type=Path)
    parser.add_argument('--output',type=Path,default=ROOT/'runs/navigation_ppo')
    parser.add_argument('--resume',type=Path)
    parser.add_argument('--static-only',action='store_true')
    parser.add_argument('--legacy-planar',action='store_true')
    parser.add_argument('--multi-map',action='store_true')
    parser.add_argument('--split',choices=['train','validation','test'],default='train')
    parser.add_argument('--curriculum',action='store_true')
    parser.add_argument('--max-episode-steps',type=int,default=800)
    args=parser.parse_args(); args.output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4)
    env=make_navigation_env(legacy_planar=args.legacy_planar,localizer=args.localizer,dynamic=not args.static_only,
                      max_steps=args.max_episode_steps,curriculum=args.curriculum)
    if args.multi_map:
        env.close();env=MultiMapNavigationEnv(split=args.split,localizer=args.localizer,dynamic=not args.static_only,max_steps=args.max_episode_steps)
    env=Monitor(env,str(args.output/'monitor.csv'))
    try:
        config=dict(simulation_backend=getattr(env.unwrapped,'physics_backend','legacy_planar'),scene_file=env.unwrapped.scene_file,localizer=args.localizer,seed=args.seed,requested_steps=args.steps,
                    dynamic=not args.static_only,curriculum=args.curriculum,
                    observation='three limited-FOV scans + localized path/goal + odometric velocity + action + covariance',
                    lidar=env.unwrapped.lidar.signature,ground_truth_in_observation=False,navigation_version=2,collision_terminates=False,goal_radius_m=.30)
        (args.output/'config.json').write_text(json.dumps(config,indent=2))
        if args.resume:
            previous=json.loads((args.resume.parent/'config.json').read_text())
            if previous['lidar']!=config['lidar']:
                raise ValueError('LiDAR configuration changed: retrain or explicitly migrate the policy')
            model=PPO.load(args.resume,env=env,device='cpu',tensorboard_log=str(args.output/'tensorboard'))
        else:
            model=PPO('MlpPolicy',env,device='cpu',n_steps=128,batch_size=64,n_epochs=4,
                      learning_rate=3e-4,gamma=.99,gae_lambda=.95,ent_coef=.01,
                      policy_kwargs={'net_arch':dict(pi=[128,128],vf=[128,128])},
                      seed=args.seed,tensorboard_log=str(args.output/'tensorboard'),verbose=1)
        if args.learning_rate is not None:
            from stable_baselines3.common.utils import get_schedule_fn
            model.learning_rate=args.learning_rate; model.lr_schedule=get_schedule_fn(args.learning_rate)
        model.set_random_seed(args.seed)
        starting_steps=model.num_timesteps
        config['parent_policy']=str(args.resume) if args.resume else None
        config['starting_steps']=starting_steps
        callback=CheckpointCallback(save_freq=5000,save_path=str(args.output/'checkpoints'),name_prefix='ppo')
        if args.demo_anchor:
            callback=CallbackList([callback,DemonstrationAnchor(args.demo_anchor,args.seed)])
        config['demo_anchor']=str(args.demo_anchor) if args.demo_anchor else None
        config['multi_map']=args.multi_map;config['split']=args.split
        if args.multi_map:config['map_manifest']=str(env.unwrapped.dataset/'manifest.json');config['training_maps']=env.unwrapped.entries
        config['pedestrian_laser_visible']=True
        config['reward_version']=4 if args.multi_map else 3
        config['learning_rate']=float(model.lr_schedule(1.))
        model.learn(total_timesteps=args.steps,callback=callback,reset_num_timesteps=not bool(args.resume))
        model.save(args.output/'policy')
        if args.multi_map:config['sampled_tasks']=env.unwrapped.task_history
        config['completed_steps']=model.num_timesteps
        config['added_steps']=model.num_timesteps-starting_steps
        config['estimated_stop_radius_m']=.12
        config['validation_only']=args.steps<10000
        (args.output/'config.json').write_text(json.dumps(config,indent=2))
        print('Saved',args.output/'policy.zip','; evaluate on held-out seeds before deployment')
    finally:
        env.close()


if __name__=='__main__': main()
