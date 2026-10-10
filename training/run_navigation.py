"""Interactive MuJoCo replay using laser-localized control."""
import argparse
import json
from pathlib import Path
import time
import torch
import mujoco.viewer
from stable_baselines3 import PPO
from navrl.physical_env import make_navigation_env
from navrl.multimap import MultiMapNavigationEnv
from evaluate_navigation import baseline
from navrl.visualization import NavigationOverlay


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--localizer',choices=['ros-amcl','particle'],default='ros-amcl')
    p.add_argument('--policy',type=Path)
    p.add_argument('--seed',type=int,default=1000)
    p.add_argument('--static-only',action='store_true')
    p.add_argument('--legacy-planar',action='store_true')
    p.add_argument('--multi-map',action='store_true')
    p.add_argument('--split',choices=['train','validation','test'],default='test')
    p.add_argument('--map-index',type=int,default=0)
    p.add_argument('--task',type=int,choices=[0,1,2])
    p.add_argument('--episodes',type=int,default=0,help='0 repeats; positive value stops after that many episodes')
    p.add_argument('--max-episode-steps',type=int,default=1000)
    args=p.parse_args(); torch.set_num_threads(4)
    env=make_navigation_env(legacy_planar=args.legacy_planar,localizer=args.localizer,dynamic=not args.static_only,max_steps=args.max_episode_steps)
    if args.multi_map:
        env.close();env=MultiMapNavigationEnv(split=args.split,localizer=args.localizer,dynamic=not args.static_only,max_steps=args.max_episode_steps)
        args.episodes=1
    try:
        model=PPO.load(args.policy,device='cpu') if args.policy else None
        if model and json.loads((args.policy.parent/'config.json').read_text())['lidar']!=env.lidar.signature:
            raise ValueError('Scanner profile does not match policy')
        obs,_=env.reset(seed=args.seed,options={'map_index':args.map_index} if args.multi_map else {'task':args.task} if args.task is not None else None); completed=0
        overlay=NavigationOverlay(env.model,env.lidar,env.path,env.goal)
        with mujoco.viewer.launch_passive(env.model,env.data) as viewer:
            viewer.cam.type=2
            viewer.cam.fixedcamid=0
            while viewer.is_running():
                started=time.monotonic()
                action=model.predict(obs,deterministic=True)[0] if model else baseline(env)
                obs,_,terminated,truncated,info=env.step(action)
                with viewer.lock():
                    viewer.user_scn.ngeom=0
                    overlay.path=env.path
                    overlay.draw(viewer.user_scn,env.data,env.truth,env.scan)
                cmd=info["commanded_velocity"]; actual=info["actual_velocity"]
                source="COLLISION BLOCKED" if info["contact_this_step"] and cmd[0]>.01 and actual[0]<.01 else "POLICY STOP" if cmd[0]<=.01 else "POLICY DRIVE"
                viewer.set_texts((None,None,"Policy v (m/s)\nActual v (m/s)\nPolicy w (rad/s)\nMotion source\nContacts\nPedestrians",f"{cmd[0]:.3f}\n{actual[0]:.3f}\n{cmd[1]:.3f}\n{source}\n{info['collision_events']}\n{', '.join(info.get('pedestrian_states',[]))}"))
                if info.get('decision'):
                    d=info['decision'];viewer.set_texts((None,None,'Decision\nReason\nRoute revision\nControl source',f"{d['state']}\n{d['reason']}\n{d['plan_revision']}\n{'supervisor_wait' if info.get('supervisor_override') else 'PPO'}"))
                viewer.sync()
                if terminated or truncated:
                    verified=info['status']=='success' and info['true_goal_distance_m']<=.30
                    print('Verified arrival:',verified,'goal distance:',info['true_goal_distance_m'],'contacts:',info['collision_events'],flush=True)
                    completed+=1
                    if args.episodes and completed>=args.episodes:
                        time.sleep(2); break
                    args.seed+=1; obs,_=env.reset(seed=args.seed,options={'map_index':args.map_index} if args.multi_map else {'task':args.task} if args.task is not None else None)
                    overlay=NavigationOverlay(env.model,env.lidar,env.path,env.goal)
                time.sleep(max(0,env.dt-(time.monotonic()-started)))
    finally: env.close()


if __name__=='__main__': main()
