"""Learn a path-following demonstration policy, then save it for PPO optimization.
Demonstrations use laser-localized pose and known static map, never current true pose.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from stable_baselines3 import PPO
from navrl.physical_env import make_navigation_env
from navrl.multimap import MultiMapNavigationEnv
from navrl.geometry import ROOT
from evaluate_navigation import baseline


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--episodes',type=int,default=18)
    p.add_argument('--legacy-planar',action='store_true')
    p.add_argument('--multi-map',action='store_true')
    p.add_argument('--split',choices=['train','validation','test'],default='train')
    p.add_argument('--epochs',type=int,default=60)
    p.add_argument('--resume',type=Path)
    p.add_argument('--seed',type=int,default=100)
    p.add_argument('--learning-rate',type=float,default=1e-3)
    p.add_argument('--learner-mix',type=float,default=0.)
    p.add_argument('--output',type=Path,default=ROOT/'runs/goal_navigation')
    args=p.parse_args(); args.output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4); torch.manual_seed(42)
    rng=np.random.default_rng(42); observations=[]; actions=[]; outcomes=[]
    env=make_navigation_env(legacy_planar=args.legacy_planar,localizer='particle',dynamic=True,max_steps=1000)
    if args.multi_map:
        env.close();env=MultiMapNavigationEnv(split=args.split,localizer='particle',dynamic=True,max_steps=700)
    learner=PPO.load(args.resume,device='cpu') if args.resume else None
    try:
        for episode in range(args.episodes):
            task=episode%3
            options={'task':task,'initial_yaw':float([0.,np.pi/2,-np.pi/2,np.pi][(episode//3)%4])}
            obs,info=env.reset(seed=args.seed+episode,options={'map_index':episode%len(env.entries)} if args.multi_map else options)
            for step in range(1000):
                action=baseline(env)
                observations.append(obs.copy()); actions.append(action.copy())
                executed=learner.predict(obs,deterministic=True)[0] if learner is not None and rng.random()<args.learner_mix else action
                obs,_,terminated,truncated,info=env.step(executed)
                if terminated or truncated: break
            outcomes.append(dict(seed=args.seed+episode,task=task,steps=step+1,status=info['status'],
                                 goal_distance_m=info['true_goal_distance_m'],collisions=info['collision_events']))
            print('demonstration',outcomes[-1],flush=True)
        x=torch.as_tensor(np.array(observations),dtype=torch.float32)
        y=torch.as_tensor(np.array(actions),dtype=torch.float32)
        model=PPO('MlpPolicy',env,device='cpu',n_steps=256,batch_size=64,n_epochs=4,
                  learning_rate=1e-4,gamma=.995,ent_coef=.001,
                  policy_kwargs={'net_arch':dict(pi=[128,128],vf=[128,128])},seed=42,verbose=1)
        if learner is not None:
            model.policy.load_state_dict(learner.policy.state_dict())
        actor=list(model.policy.mlp_extractor.policy_net.parameters())+list(model.policy.action_net.parameters())+list(model.policy.features_extractor.parameters())
        optimizer=torch.optim.Adam(actor,lr=args.learning_rate)
        for epoch in range(args.epochs):
            losses=[]
            indices=rng.permutation(len(x))
            for start in range(0,len(x),256):
                index=indices[start:start+256]
                mean=model.policy.get_distribution(x[index]).distribution.mean
                loss=torch.nn.functional.mse_loss(mean,y[index])
                optimizer.zero_grad(); loss.backward(); optimizer.step(); losses.append(float(loss.detach()))
            if epoch%10==0: print('cloning epoch',epoch,'MSE',float(np.mean(losses)),flush=True)
        with torch.no_grad(): model.policy.log_std.fill_(-2.3)
        model.save(args.output/'warm_start')
        config=dict(scene_file=env.scene_file,simulation_backend=getattr(env,'physics_backend','legacy_planar'),localizer='particle',seed=42,training_method='demonstration warm start followed by PPO',
                    lidar=env.lidar.signature,ground_truth_in_observation=False,navigation_version=2,
                    collision_terminates=False,goal_radius_m=.30,demonstrations=outcomes,samples=len(x),epochs=args.epochs,
                    parent_policy=str(args.resume) if args.resume else None,learner_mix=args.learner_mix,
                    demonstration_seed=args.seed)
        config['pedestrian_laser_visible']=True
        config.update(multi_map=args.multi_map,split=args.split,sampled_tasks=env.task_history if args.multi_map else [])
        (args.output/'config.json').write_text(json.dumps(config,indent=2))
        np.savez_compressed(args.output/'demonstrations.npz',observations=np.array(observations),actions=np.array(actions))
        print('Saved warm-start actor; this checkpoint still requires independent evaluation and PPO training.')
    finally: env.close()


if __name__=='__main__': main()
