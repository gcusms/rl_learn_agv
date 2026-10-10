"""Evaluate baseline or PPO with localized pose; save metrics and a trace plot."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from stable_baselines3 import PPO
from navrl.physical_env import make_navigation_env
from navrl.multimap import MultiMapNavigationEnv, DecisionNavigationEnv
from navrl.geometry import ROOT, wrap
from navrl.map import path_features


def baseline(env):
    target,_,_=path_features(env.path,env.estimate[:2],getattr(env,'routing_map',env.map))
    delta=target-env.estimate[:2]
    heading=wrap(np.arctan2(delta[1],delta[0])-env.estimate[2])
    turn=np.clip(heading*1.8/.5,-1,1)
    speed=max(0.,np.cos(heading))*.9 if abs(heading)<.7 else 0.
    valid=np.isfinite(env.scan)&(env.scan>=env.lidar.range_min)&(env.scan<env.lidar.range_max)
    angles=env.lidar.angles[valid]; ranges=env.scan[valid]
    mount=env.lidar.mount; yaw=env.estimate[2]
    origin=env.estimate[:2]+np.array([np.cos(yaw)*mount[0]-np.sin(yaw)*mount[1],np.sin(yaw)*mount[0]+np.cos(yaw)*mount[1]])
    ends=origin+ranges[:,None]*np.column_stack([np.cos(angles+yaw+mount[3]),np.sin(angles+yaw+mount[3])])
    unmatched=env.map.lookup(ends,env.map.distance,outside=0.)>.18
    # Wait for unknown dynamic returns ahead; static walls are handled by the path geometry.
    if np.any(unmatched&(np.abs(angles)<.55)&(ranges<.60)):
        speed=0.
    return np.array([speed,turn],np.float32)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--localizer',choices=['ros-amcl','particle'],default='ros-amcl')
    p.add_argument('--policy',type=Path)
    p.add_argument('--episodes',type=int,default=3)
    p.add_argument('--max-episode-steps',type=int,default=800)
    p.add_argument('--seed',type=int,default=1000)
    p.add_argument('--static-only',action='store_true')
    p.add_argument('--legacy-planar',action='store_true')
    p.add_argument('--multi-map',action='store_true')
    p.add_argument('--dataset',type=Path)
    p.add_argument('--initial-prior-offset',type=float,nargs=3)
    p.add_argument('--neural',action='store_true',help='Learned localization and neural routing; requires --policy')
    p.add_argument('--components',type=Path)
    p.add_argument('--split',choices=['train','validation','test'],default='test')
    p.add_argument('--output',type=Path,default=ROOT/'runs/evaluation')
    p.add_argument('--save-frames',action='store_true')
    p.add_argument('--record-success',action='store_true')
    p.add_argument('--record-most-yields',action='store_true')
    p.add_argument('--record-most-stops',action='store_true',help='Select successful replay with the most non-contact stop-near-obstacle steps')
    p.add_argument('--task',type=int)
    p.add_argument('--scene-dir',type=Path)
    p.add_argument('--map-index',type=int)
    p.add_argument('--start',type=float,nargs=3,metavar=('X','Y','YAW'))
    p.add_argument('--goal',type=float,nargs=2,metavar=('X','Y'))
    p.add_argument('--scheduled-columns',action='store_true')
    p.add_argument('--obstacle-on-route',action='store_true',help='Explicit stress scenario: one solid pedestrian pauses on the initial planned route for 10 simulated seconds')
    p.add_argument('--max-trace-episodes',type=int,default=3)
    args=p.parse_args(); args.output.mkdir(parents=True,exist_ok=True); torch.set_num_threads(4)
    if args.neural:
        from neural_navigation.env import LearnedMultiMapEnv,LearnedNavigationEnv
        if args.policy is None:raise ValueError('--neural requires a learned --policy checkpoint')
        if args.multi_map and args.scene_dir:raise ValueError('Choose --scene-dir or --multi-map')
        args.localizer='learned_attention'
        if args.multi_map:env=LearnedMultiMapEnv(dataset=args.dataset,components=args.components,split=args.split,dynamic=not args.static_only,max_steps=args.max_episode_steps)
        else:env=LearnedNavigationEnv(components=args.components,scene_dir=args.scene_dir,dynamic=not args.static_only,max_steps=args.max_episode_steps)
    else:
        env=make_navigation_env(legacy_planar=args.legacy_planar,localizer=args.localizer,dynamic=not args.static_only,max_steps=args.max_episode_steps)
        if args.multi_map:
            env.close();env=MultiMapNavigationEnv(dataset=args.dataset,split=args.split,localizer=args.localizer,dynamic=not args.static_only,max_steps=args.max_episode_steps)
        if args.scene_dir:
            if args.multi_map:raise ValueError('Choose --scene-dir or --multi-map')
            env.close();env=DecisionNavigationEnv(scene_dir=args.scene_dir,localizer=args.localizer,dynamic=not args.static_only,max_steps=args.max_episode_steps)
    if (args.start is None)!=(args.goal is None):raise ValueError('--start and --goal must be specified together')
    traces=[]; outcomes=[]; success_recorded=False; best_stop_score=-1
    try:
        model=PPO.load(args.policy,device='cpu') if args.policy else None
        if model:
            config_path=args.policy.parent/'config.json'
            if not config_path.exists() and args.policy.parent.name=='checkpoints': config_path=args.policy.parent.parent/'config.json'
            config=json.loads(config_path.read_text())
            if config['lidar']!=env.lidar.signature: raise ValueError('Policy/scanner configuration mismatch')
        for episode in range(args.episodes):
            reset_options={'map_index':args.map_index if args.map_index is not None else episode%len(env.entries)} if args.multi_map else {'task':args.task if args.task is not None else episode%len(env.scene['tasks'])}
            if args.initial_prior_offset is not None:
                if not args.neural:raise ValueError('Coarse prior perturbation is available with --neural')
                reset_options['initial_prior_offset']=args.initial_prior_offset
            if args.start is not None:reset_options.update(start=args.start,goal=args.goal)
            obs,info=env.reset(seed=args.seed+episode,options=reset_options)
            if args.scheduled_columns:
                from navrl.scenarios import configure_column_task
                obs=configure_column_task(env,args.task if args.task is not None else episode%len(env.scene['tasks']))
            if args.obstacle_on_route:
                from navrl.scenarios import place_route_obstacle
                obs=place_route_obstacle(env)
            trace=[]; total_reward=0.
            for step in range(args.max_episode_steps):
                action=model.predict(obs,deterministic=True)[0] if model else baseline(env)
                obs,reward,terminated,truncated,info=env.step(action); total_reward+=reward
                trace.append(dict(column_block_events=info.get('column_block_events',[]),proposed_policy_action=info.get('proposed_policy_action',action.tolist()),decision=info.get('decision'),supervisor_override=info.get('supervisor_override',False),planned_path=env.path.tolist() if args.multi_map or args.scene_dir else None,truth=info['ground_truth_pose'].tolist(),estimate=info['estimated_pose'].tolist(),
                                  error_m=info['localization_error_m'],time_s=info['simulation_seconds'],
                                  qpos=env.data.qpos.tolist() if args.record_success else None,
                                  qvel=env.data.qvel.tolist() if args.record_success else None,
                                  pedestrian_positions=info.get('pedestrian_positions'),
                                  pedestrian_states=info.get('pedestrian_states'),
                                  pedestrian_velocities=info.get('pedestrian_velocities'),
                                  minimum_dynamic_gap_m=info.get('minimum_dynamic_gap_m'),
                                  mocap_pos=env.data.mocap_pos.tolist() if args.record_success else None,
                                  mocap_quat=env.data.mocap_quat.tolist() if args.record_success else None,
                                  scan=[float(x) if np.isfinite(x) else None for x in env.scan] if args.record_success else None,
                                  commanded_velocity=info['commanded_velocity'].tolist(),
                                  actual_velocity=info['actual_velocity'].tolist(),
                                  contact=bool(info['contact_this_step']),
                                  front_clearance_m=float(np.min(np.nan_to_num(env.scan[np.abs(env.lidar.angles)<.55],nan=env.lidar.range_max,posinf=env.lidar.range_max))),
                                  motion_source='supervisor_wait' if info.get('supervisor_override') else 'collision_blocked' if info['contact_this_step'] and info['commanded_velocity'][0]>.01 and info['actual_velocity'][0]<.01 else 'policy_stop' if info['commanded_velocity'][0]<=.01 else 'policy_drive',
                                  collisions=info['collision_events']))
                if args.save_frames and episode==0 and step%20==0:
                    from PIL import Image
                    Image.fromarray(env.render()).save(args.output/f'frame_{step:04d}.png')
                if terminated or truncated: break
            status=info['status']
            verified_success=status=='success' and info['true_goal_distance_m']<=.30
            outcomes.append(dict(column_block_events=info.get('column_block_events',[]),map_id=info.get('map_id'),seed=args.seed+episode,plan_revisions=info.get('decision',{}).get('plan_revision',0),supervisor_wait_steps=sum(t['supervisor_override'] for t in trace),episode=episode,status=status,verified_success=verified_success,steps=len(trace),
                policy_stop_steps=sum(t['motion_source']=='policy_stop' for t in trace),
                policy_stop_near_obstacle_steps=sum(t['motion_source']=='policy_stop' and t['front_clearance_m']<.8 and not t['contact'] for t in trace),
                collision_blocked_steps=sum(t['motion_source']=='collision_blocked' for t in trace),
                return_=total_reward,mean_localization_error_m=float(np.mean([t['error_m'] for t in trace])),
                max_localization_error_m=float(max(t['error_m'] for t in trace)),
                minimum_dynamic_gap_m=info.get('minimum_dynamic_gap_m'),
                pedestrian_yield_events=info.get('pedestrian_yield_events'),
                penetration_guard_activations=info.get('penetration_guard_activations'),
                native_contact_substeps=info.get('native_contact_substeps'),
                true_goal_distance_m=info['true_goal_distance_m'],traveled_m=info['traveled_m'],collision_events=info['collision_events'],collision_steps=info['collision_steps']))
            if episode<args.max_trace_episodes: traces.append(dict(map_id=info.get('map_id'),walls=env.scene['walls'],seed=args.seed+episode,task=args.task if args.task is not None else episode%len(env.scene['tasks']),goal=env.goal.tolist(),path=env.path.tolist(),samples=trace))
            stop_score=outcomes[-1]['pedestrian_yield_events'] if args.record_most_yields else outcomes[-1]['policy_stop_near_obstacle_steps']
            if verified_success and args.record_success and (not success_recorded or (args.record_most_stops or args.record_most_yields) and stop_score>best_stop_score):
                record=dict(scene_directory=str(env.scene_dir),scene_config=env.scene,map_id=info.get('map_id'),scene_file=env.scene_file,simulation_backend=getattr(env,'physics_backend','legacy_planar'),seed=args.seed+episode,task=args.task if args.task is not None else episode%len(env.scene['tasks']),
                            controller='PPO' if model else 'baseline',localizer=args.localizer,lidar=env.lidar.signature,
                            outcome=outcomes[-1],replay_selection='Successful episode with most pedestrian yielding events' if args.record_most_yields else 'Successful episode with most non-contact stop-near-obstacle steps' if args.record_most_stops else 'First successful episode',goal=env.goal.tolist(),path=env.path.tolist(),samples=trace)
                (args.output/'successful_replay.json').write_text(json.dumps(record))
                success_recorded=True; best_stop_score=stop_score
            if not verified_success:
                failure=dict(scene_file=env.scene_file,scene_directory=str(env.scene_dir),scene_config=env.scene,map_id=info.get('map_id'),seed=args.seed+episode,controller='PPO' if model else 'baseline',localizer=args.localizer,lidar=env.lidar.signature,outcome=outcomes[-1],goal=env.goal.tolist(),path=env.path.tolist(),samples=trace)
                (args.output/f'failed_replay_{args.seed+episode}.json').write_text(json.dumps(failure))
            print(outcomes[-1],flush=True)
        summary=dict(scheduled_column_tasks=args.scheduled_columns,initial_prior_offset=args.initial_prior_offset,fully_learned_robot_navigation=args.neural,runtime_amcl=not args.neural and args.localizer=='ros-amcl',runtime_robot_astar=not args.neural,stress_scenario='solid_pedestrian_pauses_on_initial_route_10s' if args.obstacle_on_route else None,multi_map=args.multi_map,split=args.split,sampled_tasks=env.task_history if args.multi_map else [],scene_file=env.scene_file,simulation_backend=getattr(env,'physics_backend','legacy_planar'),localizer=args.localizer,controller='PPO' if model else 'path-following baseline',
                     lidar=env.lidar.signature,episodes=outcomes,
                     success_rate=float(np.mean([r['verified_success'] for r in outcomes])),
                     success_criterion='Reached final goal within 0.30 m; collisions do not invalidate arrival',
                     collision_rate=float(np.mean([r['collision_events']>0 for r in outcomes])),
                     mean_collision_events=float(np.mean([r['collision_events'] for r in outcomes])),
                     evaluation_seed=args.seed,goal_radius_m=.30,
                     timeout_rate=float(np.mean([r['status']=='timeout' for r in outcomes])))
        (args.output/'metrics.json').write_text(json.dumps(summary,indent=2))
        (args.output/'trajectories.json').write_text(json.dumps(traces,indent=2))
        fig,axes=plt.subplots(1,2,figsize=(13,5))
        if not args.multi_map:axes[0].imshow(env.map.blocked,cmap='gray_r',origin='upper',extent=[env.map.origin[0],env.map.origin[0]+env.map.width*env.map.res,env.map.origin[1],env.map.origin[1]+env.map.height*env.map.res])
        for i,trace in enumerate(traces):
            samples=trace['samples']; truth=np.array([t['truth'] for t in samples]); est=np.array([t['estimate'] for t in samples])
            path=np.array(trace['path']); axes[0].plot(path[:,0],path[:,1],':',alpha=.4)
            axes[0].plot(truth[:,0],truth[:,1],label=f'Truth {i+1}')
            axes[0].plot(est[:,0],est[:,1],'--',label=f'Laser estimate {i+1}')
            axes[1].plot([t['time_s'] for t in samples],[t['error_m'] for t in samples],label=f'Episode {i+1}')
        axes[0].set(xlabel='x (m)',ylabel='y (m)',title='Map, planned paths and localization',aspect='equal'); axes[0].legend(fontsize=8)
        axes[1].set(xlabel='Simulated time (s)',ylabel='Localization position error (m)',title=f'{args.localizer}: diagnostic truth comparison'); axes[1].legend()
        fig.tight_layout(); fig.savefig(args.output/'evaluation.png',dpi=140); plt.close(fig)
        if args.multi_map:
            for i,tr in enumerate(traces):
                fig,ax=plt.subplots(figsize=(8,6))
                from matplotlib.patches import Rectangle
                for x0,y0,x1,y1 in tr['walls']:ax.add_patch(Rectangle((x0,y0),x1-x0,y1-y0,color='#465569'))
                poses=np.array([t['truth'] for t in tr['samples']]);ax.plot(poses[:,0],poses[:,1],color='orange',label='Actual trajectory')
                for t in tr['samples'][::25]:
                    path=np.array(t['planned_path']);ax.plot(path[:,0],path[:,1],color='cyan',alpha=.25)
                ax.plot(*tr['goal'],'go');ax.set(xlim=(0,10),ylim=(0,8),aspect='equal',title=tr['map_id']);ax.legend()
                fig.savefig(args.output/f'map_episode_{i:02d}.png',dpi=130);plt.close(fig)
        if traces:
            samples=traces[0]['samples']; t=np.array([x['time_s'] for x in samples])
            cmd=np.array([x['commanded_velocity'] for x in samples]); actual=np.array([x['actual_velocity'] for x in samples])
            fig,axes=plt.subplots(3,1,figsize=(11,8),sharex=True)
            axes[0].plot(t,cmd[:,0],label='Policy commanded forward velocity')
            axes[0].plot(t,actual[:,0],label='Actual forward velocity',alpha=.8); axes[0].set_ylabel('v (m/s)'); axes[0].legend()
            axes[1].plot(t,cmd[:,1],label='Policy commanded turn rate')
            axes[1].plot(t,actual[:,1],label='Actual turn rate',alpha=.8); axes[1].set_ylabel('w (rad/s)'); axes[1].legend()
            axes[2].step(t,[x['contact'] for x in samples],label='Physical contact',where='post')
            axes[2].step(t,[x['motion_source']=='policy_stop' for x in samples],label='Policy requests stop',where='post')
            axes[2].set(xlabel='Simulation time (s)',ylabel='Boolean',ylim=(-.1,1.2)); axes[2].legend()
            fig.suptitle('Commanded motion versus contact blocking: first evaluation episode')
            fig.tight_layout(); fig.savefig(args.output/'control_diagnostics.png',dpi=130); plt.close(fig)
    finally: env.close()


if __name__=='__main__': main()
