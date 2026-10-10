"""Continuous waypoint missions: preserve physics, odometry, localization and columns."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
import torch
from stable_baselines3 import PPO
from navrl.geometry import ROOT
from navrl.scenarios import configure_column_task
from neural_navigation.env import LearnedNavigationEnv


def main():
    p=argparse.ArgumentParser();p.add_argument('--task',type=int,default=0);p.add_argument('--max-leg-steps',type=int,default=900);p.add_argument('--scene-dir',type=Path,default=ROOT/'assets/room_navigation_scripted');p.add_argument('--output-root',type=Path,default=ROOT/'runs/room_navigation_scripted');p.add_argument('--policy',type=Path,default=ROOT/'runs/full_model_navigation_v2/policy.zip');p.add_argument('--components',type=Path,default=ROOT/'runs/full_model_navigation_v2');p.add_argument('--revision',choices=['v2','v3'],default='v2');p.add_argument('--seed-base',type=int,default=27000);a=p.parse_args()
    torch.set_num_threads(2);scene=a.scene_dir;out=a.output_root/f'mission_{a.task}';out.mkdir(parents=True,exist_ok=True)
    bundle=ROOT/'runs/full_model_navigation_v2';weights={n:hashlib.sha256((bundle/n).read_bytes()).hexdigest() for n in ['policy.zip','localizer.pt','router.pt']}
    model=PPO.load(a.policy,device='cpu')
    if a.revision=='v3':
        from neural_navigation.room_env import RoomNavigationEnv
        env_class=RoomNavigationEnv
    else:env_class=LearnedNavigationEnv
    env=env_class(components=a.components,scene_dir=scene,dynamic=True,max_steps=4*a.max_leg_steps)
    try:
        obs,info=env.reset(seed=a.seed_base+a.task,options={'task':a.task});obs=configure_column_task(env,a.task)
        task=env.scene['tasks'][a.task];samples=[];legs=[]
        for index,goal in enumerate(task['waypoints']):
            if index:
                # Target transition only; no reset, pose correction or ground-truth control.
                obs=env.set_navigation_goal(goal)
            leg_start=env.elapsed
            for _ in range(a.max_leg_steps):
                action=model.predict(obs,deterministic=True)[0];obs,_,done,truncated,info=env.step(action)
                samples.append(dict(active_goal=list(goal),waypoint_index=index,waypoints_completed=len(legs),column_block_events=info.get('column_block_events',[]),
                    decision=info.get('decision'),supervisor_override=False,planned_path=env.path.tolist(),truth=info['ground_truth_pose'].tolist(),estimate=info['estimated_pose'].tolist(),
                    error_m=info['localization_error_m'],time_s=info['simulation_seconds'],qpos=env.data.qpos.tolist(),qvel=env.data.qvel.tolist(),
                    pedestrian_positions=info['pedestrian_positions'],pedestrian_states=info['pedestrian_states'],pedestrian_velocities=info['pedestrian_velocities'],
                    scan=[float(x) if np.isfinite(x) else None for x in env.scan],commanded_velocity=info['commanded_velocity'].tolist(),actual_velocity=info['actual_velocity'].tolist(),
                    contact=bool(info['contact_this_step']),motion_source='collision_blocked' if info['contact_this_step'] and abs(info['actual_velocity'][0])<.01 else 'policy_reverse' if info['commanded_velocity'][0]<-.01 else 'policy_stop' if abs(info['commanded_velocity'][0])<=.01 else 'policy_drive',collisions=info['collision_events']))
                if done or truncated:break
            verified=bool(done and info['true_goal_distance_m']<=.30)
            leg=dict(index=index+1,goal=goal,model_arrived=bool(done),verified_success=verified,true_goal_distance_m=info['true_goal_distance_m'],seconds=env.elapsed-leg_start)
            legs.append(leg);samples[-1]['waypoints_completed']=sum(l['verified_success'] for l in legs);print('mission',a.task,'leg',leg,flush=True)
            if not done or truncated:break
        outcome=dict(verified_success=len(legs)==len(task['waypoints']) and all(l['verified_success'] for l in legs),legs=legs,targets_total=len(task['waypoints']),
                     targets_reached=sum(l['verified_success'] for l in legs),true_goal_distance_m=info['true_goal_distance_m'],collision_events=info['collision_events'],
                     penetration_guard_activations=info['penetration_guard_activations'],steps=len(samples),seconds=env.elapsed,traveled_m=info['traveled_m'],
                     mean_localization_error_m=float(np.mean([s['error_m'] for s in samples])),max_localization_error_m=float(max(s['error_m'] for s in samples)),
                     column_block_events=info['column_block_events'],runtime_robot_amcl=False,runtime_robot_astar=False,localizer_reset_between_targets=False,parent_weight_sha256=weights,weight_sha256={'policy.zip':hashlib.sha256(a.policy.read_bytes()).hexdigest(),**{n:hashlib.sha256((a.components/n).read_bytes()).hexdigest() for n in ['localizer.pt','router.pt']}},policy_path=str(a.policy.resolve()),policy_sha256=hashlib.sha256(a.policy.read_bytes()).hexdigest(),policy_revision=a.revision)
        trace=dict(scene_directory=str(scene),scene_config=env.scene,scene_file='physical_scene.xml',task=a.task,seed=a.seed_base+a.task,controller='PPO',localizer='learned_attention',lidar=env.lidar.signature,
                   outcome=outcome,goal=env.goal.tolist(),waypoints=task['waypoints'],path=env.path.tolist(),samples=samples)
        (out/'metrics.json').write_text(json.dumps(outcome,indent=2));(out/'replay.json').write_text(json.dumps(trace))
        if weights!={n:hashlib.sha256((bundle/n).read_bytes()).hexdigest() for n in weights}:raise RuntimeError('Frozen weights changed')
        print('RESULT',json.dumps(outcome),flush=True)
    finally:env.close()

if __name__=='__main__':main()
