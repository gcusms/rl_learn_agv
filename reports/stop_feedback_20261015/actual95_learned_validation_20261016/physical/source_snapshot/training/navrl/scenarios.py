"""Explicit evaluation scene interventions, not policy inputs."""
import numpy as np
import mujoco


def place_route_obstacle(env):
    target_env=getattr(env,'active',env)
    candidates=[p for p in target_env.path if .9<np.linalg.norm(p-target_env.estimate[:2])<1.8 and target_env.wall_gap(p)>.14]
    peers=target_env._humans()[1:]
    candidates=[p for p in candidates if all(np.linalg.norm(p-o)>.7 for o in peers)]
    if not candidates:raise RuntimeError('No valid placement for route-blocking stress scenario')
    q,d=target_env.addresses[target_env.names[1]]
    target_env.data.qpos[q]=candidates[len(candidates)//2];target_env.data.qvel[d]=0.
    mujoco.mj_forward(target_env.model,target_env.data)
    target_env._new_route(0);target_env.walkers[0]['pause']=10.;target_env.walkers[0]['state']='pausing'
    target_env.scan=target_env._scan();target_env.estimate=target_env.localizer.update(target_env.odom,target_env.scan)
    obs=target_env._observation(target_env.scan,initialize=True);target_env.previous_remaining=target_env.remaining
    target_env.minimum_gap=target_env._minimum_gap()
    return obs


def configure_column_task(env,task_index=0):
    """World setup only: place solid columns and schedule approach-triggered release."""
    e=getattr(env,'active',env);profile=e.scene['tasks'][task_index].get('column_task')
    if not profile:raise ValueError('Selected scene task has no column schedule')
    for spec in profile['blocks']:
        i=spec['column_index'];point=np.asarray(spec['position'],float)
        if e.robot_gap(point)<.35 or e.wall_gap(point)<.03:raise ValueError('Unsafe initial blocking-column placement')
        q,d=e.addresses[e.names[i+1]];e.data.qpos[q]=point;e.data.qvel[d]=0.
    mujoco.mj_forward(e.model,e.data)
    if e._minimum_gap()<0.:raise ValueError('Scheduled columns overlap at initial placement')
    for spec in profile['blocks']:
        i=spec['column_index'];w=e.walkers[i];goal=np.asarray(spec['release_goal'],float)
        w['path']=e.human_map.plan(e._humans()[i],goal);w['goal']=goal;w['pause']=0.;w['wait']=0.
        w['block']=dict(column_index=i,phase='armed',hold_seconds=float(spec['hold_seconds']),trigger_distance_m=float(spec.get('trigger_distance_m',2.5)),group='paired_passage' if len(profile['blocks'])>1 else None,
                        trigger_timeout_s=60.,trigger_time_s=None,release_time_s=None,cleared_time_s=None)
        if spec.get('mode')=='approach_cross_return':
            block_position=np.asarray(spec['block_position'],float)
            if e.wall_gap(block_position)<.015 or not e.human_map.visible(np.asarray(spec['position']),block_position):
                raise ValueError('Scripted entry must follow a clear physical segment')
            w['speed']=float(spec.get('speed_m_s',.45))
            w['block'].update(mode='approach_cross_return',group='door_entry_return',block_position=block_position.tolist(),
                              park_position=spec['release_goal'],trigger_position=spec['trigger_position'],entry_timeout_s=10.,
                              entered_time_s=None,entry_aborted=False,trigger_timeout_s=None)
        w['state']='waiting_beside_door' if spec.get('mode')=='approach_cross_return' else 'staging_block'
    e.scan=e._scan();e.estimate=e.localizer.update(e.odom,e.scan)
    obs=e._observation(e.scan,initialize=True);e.previous_remaining=e.remaining;e.minimum_gap=e._minimum_gap()
    return obs
