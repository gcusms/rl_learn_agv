"""Independent training tasks, direct PPO actions and fixed reward potential."""
from pathlib import Path
import json
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from navrl.geometry import ROOT
from navrl.map import StaticMap
from neural_navigation.stress_env import StressNavigationEnv
from neural_navigation.training_distance import TrainingDistanceGraph


class NavigationCurriculum(gym.Env):
    def __init__(self,components,stage='static',route_mode='shortcut',max_steps=700,motion_profiles=False):
        if stage not in ('static','dynamic','robust'):raise ValueError('Unknown curriculum stage')
        self.components=Path(components);self.stage=stage;self.route_mode=route_mode;self.max_steps=max_steps
        self.motion_profiles=bool(motion_profiles)
        self.scenes=[ROOT/'assets/room_navigation_scripted']+[ROOT/f'assets/navigation_stage2_dataset/train_{i:02d}' for i in range(6)]
        self.maps={};self.graphs={};self.active=None
        self.action_space=spaces.Box(-1,1,(2,),np.float32)
        self.observation_space=spaces.Box(-1,1,(235,),np.float32)

    def reset(self,seed=None,options=None):
        super().reset(seed=seed);rng=self.np_random
        template=(options or {}).get('training_template')
        dynamic=self.stage!='static' and rng.random()<.35
        # Retain static tasks at every stage. Dynamic appearances start in the
        # known room so their physical trigger can be independently checked.
        scene=self.scenes[0] if dynamic else self.scenes[int(rng.integers(len(self.scenes)))]
        if template is not None:
            scene=Path(template['scene'])
            if scene not in self.scenes:raise ValueError('Failure templates must use existing training layouts')
            dynamic=template['interventions'].get('obstacle') in ('crossing','sudden_block')
        if scene not in self.maps:
            self.maps[scene]=StaticMap(scene/'decision_map.yaml',radius=.3)
            self.graphs[scene]=TrainingDistanceGraph(self.maps[scene])
        m=self.maps[scene];cells=np.argwhere(m.clearance>.65)
        for _ in range(500):
            if template is None:
                start,goal=[m.xy(cells[int(rng.integers(len(cells)))]) for _ in range(2)]
            else:
                start=np.asarray(template['options']['start'][:2])+rng.uniform(-.1,.1,2)
                goal=np.asarray(template['options']['goal'])+rng.uniform(-.1,.1,2)
                if not all(m.lookup(p,m.clearance,outside=0.)>.55 for p in (start,goal)):continue
            if np.linalg.norm(start-goal)<2.5:continue
            field=self.graphs[scene].field(goal)
            r,c=[int(v) for v in m.cells(start)]
            if field.reachable[r,c] and field.value(start)<=12.:break
        else:raise RuntimeError('No feasible independent curriculum task')
        case=dict(odom_scale=1.)
        if template is not None:
            case=json.loads(json.dumps(template['interventions']))
            if dynamic:case['obstacle_speed_m_s']=float(np.clip(case['obstacle_speed_m_s']*rng.uniform(.9,1.1),.08,.16))
        elif dynamic:case.update(obstacle=str(rng.choice(['crossing','sudden_block'])),obstacle_speed_m_s=float(rng.uniform(.08,.16)),hold_s=float(rng.uniform(2.,4.)))
        if template is None and dynamic and self.motion_profiles and case['obstacle']=='crossing':
            profile=str(rng.choice(['one_way','reverse_once','stop_go'],p=[.5,.25,.25]))
            if profile!='one_way':
                case['hold_s']=0.
                if profile=='reverse_once':case['reverse_after_event_s']=float(rng.uniform(2.5,5.))
                else:
                    begin=float(rng.uniform(1.5,3.));case['actor_pause_after_event_s']=[[begin,begin+float(rng.uniform(1.,3.))]]
        if self.stage=='robust' and rng.random()<.35:
            case.update(odom_scale=float(rng.uniform(.85,1.15)),yaw_odom_scale=float(rng.uniform(.9,1.1)),odom_bias_m_s=rng.uniform(-.005,.005,2).tolist(),extra_scan_dropout=.05)
            begin=float(rng.uniform(6.,20.));case['dropout_windows']=[[begin,begin+float(rng.uniform(.4,1.))]]
        if self.active is not None:self.active.close()
        self.active=StressNavigationEnv(case=case,components=self.components,scene_dir=scene,route_mode=self.route_mode,max_steps=self.max_steps)
        heading=float(rng.uniform(-np.pi,np.pi)) if template is None else float(template['options']['start'][2]+rng.uniform(-.25,.25))
        start_pose=[*start,heading]
        reset_options=dict(start=start_pose,goal=goal.tolist())
        if self.stage=='robust' and rng.random()<.2:
            reset_options['initial_prior_offset']=rng.uniform([-.15,-.15,-.08],[.15,.15,.08]).tolist()
        episode_seed=int(rng.integers(1,2**31-1))
        observation,info=self.active.reset(seed=episode_seed,options=reset_options)
        self.training_case_json=json.dumps(dict(scene=str(scene),seed=episode_seed,stage=self.stage,
                                               options=reset_options,interventions=case,
                                               failure_template_origin=template.get('origin') if template else None),default=float)
        self.field=field;self.previous_distance=field.value(self.active.truth[:2]);self.previous_action=np.zeros(2);self.reverse_seconds=0.
        self.case=case;self.scene=scene
        return observation,info

    def step(self,action):
        e=self.active;old_collisions=e.collision_events;old_guards=e.hard_guard_activations;old_travel=e.travel
        observation,_,done,truncated,info=e.step(action)
        distance=self.field.value(e.truth[:2]);progress=self.previous_distance-distance;self.previous_distance=distance
        command=e._velocity_command(np.clip(action,-1,1))
        self.reverse_seconds=self.reverse_seconds+e.dt if command[0]<-.001 else 0.
        finite=e.scan[np.isfinite(e.scan)&(e.scan>=e.lidar.range_min)]
        clearance=float(finite.min()) if len(finite) else e.lidar.range_max
        # Static potential stays fixed even when the runtime learned route
        # changes. Do not reuse the mutable path's apparent progress reward.
        reward=15*progress-.05-.1*(e.travel-old_travel)-.01*np.sum((np.asarray(action)-self.previous_action)**2)
        reward-=.05*max(0.,(.45-clearance)/.45)
        if self.reverse_seconds>1.5:reward-=.05
        self.previous_action=np.asarray(action).copy()
        failure=e.collision_events>old_collisions or e.hard_guard_activations>old_guards
        verified=bool(done and info['true_goal_distance_m']<=.2 and not failure)
        if failure:reward=min(reward,0.)-100.;done=True;truncated=False
        elif done:reward+=50. if verified else -50.
        elif truncated:reward-=5.
        info.update(mission_targets_reached=int(verified),verified_mission_success=verified,safety_termination=bool(failure),
                    fixed_geodesic_progress_m=float(progress),training_stage=self.stage,training_scene=str(self.scene),
                    training_case_json=self.training_case_json,
                    training_actions_overridden=False)
        return observation,float(reward),bool(done),bool(truncated),info

    def close(self):
        if self.active is not None:self.active.close()
