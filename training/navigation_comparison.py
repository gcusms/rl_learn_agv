"""Comparable four-goal missions with physical, approach-triggered column interference."""
import argparse,json,hashlib,shutil
from pathlib import Path
import numpy as np
import torch
from stable_baselines3 import PPO
from navrl.geometry import ROOT
from neural_navigation.waypoint_interference import build_scene,WaypointInterferenceEnv

SCENE=ROOT/'assets/navigation_comparison'
SCHEDULE=[dict(column=1,park=[6.4,1.1],block=[6.55,1.7]),dict(column=1,park=[8.9,5.3],block=[8.2,4.9]),dict(column=0,park=[3.6,6.8],block=[3.0,6.15]),dict(column=0,park=[.8,3.0],block=[1.45,3.0])]

def scene():
    if (SCENE/'arm_home.json').exists():return SCENE
    src=build_scene();SCENE.mkdir(exist_ok=True)
    for name in ['physical_scene.xml','physical_scene.json','decision_map.yaml','decision_map.pgm','arm_home.json']:shutil.copy2(src/name,SCENE/name)
    return SCENE

class ComparisonEnv(WaypointInterferenceEnv):
    def reset(self,**kwargs):
        self.schedule_enabled=False;self.actor_routes={};self.actor_destinations={};self.interferences=[];self.column_phases={};self.leg_phase='armed';self.hold_until=None
        obs,info=super().reset(**kwargs);self.schedule_enabled=True;self.stage(0);return obs,info
    def actor_target(self,index,point):
        point=np.asarray(point,float)
        if index in self.actor_destinations and np.linalg.norm(point-self.actor_destinations[index])<.01:return
        # This planner moves WORLD ACTORS only. Robot learned routing remains unchanged.
        self.actor_routes[index]=self.human_map.plan(self._humans()[index],point).tolist()
        self.actor_destinations[index]=point;self.column_phases[index]='staging'
    def stage(self,leg):
        self.leg_phase='armed';self.hold_until=None;self.actor_target(SCHEDULE[leg]['column'],SCHEDULE[leg]['park'])
    def select_goal(self,index):
        obs=super().select_goal(index);self.stage(index);return obs
    def _walker_velocity(self,index,h):
        if not self.schedule_enabled:return np.zeros(2)
        task=SCHEDULE[self.active_leg];active=task['column'];pos=self._humans()[index]
        # A pillar finishes its 8-second hold even if the robot has already reached
        # that goal and the next leg uses the other pillar.
        for event in self.interferences:
            if event['column']!=index+1 or 'entered_time_s' not in event or 'released_time_s' in event:continue
            if self.elapsed<event['entered_time_s']+8.:
                self.column_phases[index]='holding';return np.zeros(2)
            event['released_time_s']=self.elapsed
            previous_task=SCHEDULE[event['target']-1]
            if event['target']==self.active_leg+1:self.leg_phase='returning'
            destination=task['park'] if index==active and self.leg_phase=='armed' else previous_task['park']
            self.actor_target(index,destination)
        if index==active:
            if self.leg_phase=='armed' and np.linalg.norm(self.truth[:2]-self.goal)<3.2:
                self.leg_phase='entering';self.actor_target(index,task['block'])
                self.interferences.append(dict(target=self.active_leg+1,column=index+1,trigger_time_s=self.elapsed,block_position=task['block'],hold_seconds=8.))
            if self.leg_phase=='entering' and np.linalg.norm(pos-task['block'])<.09:
                self.leg_phase='holding';self.hold_until=self.elapsed+8.
                self.interferences[-1]['entered_time_s']=self.elapsed
            # Prevent an actor from endlessly attempting to occupy robot space.
            if self.leg_phase=='entering' and self.elapsed-self.interferences[-1]['trigger_time_s']>14.:
                self.leg_phase='returning';self.actor_target(index,task['park']);self.interferences[-1]['entry_aborted']=True
            if self.leg_phase=='holding' and self.elapsed>=self.hold_until:
                self.leg_phase='returning';self.actor_target(index,task['park']);self.interferences[-1]['released_time_s']=self.elapsed
            self.column_phases[index]=self.leg_phase
        route=self.actor_routes.get(index,[])
        while len(route)>1 and np.linalg.norm(pos-route[0])<.09:route.pop(0)
        if route:self.manual_targets[index]=np.asarray(route[0])
        return self._manual_velocity(index)

def evaluate(policy,output,seed=31002,max_leg_steps=900,controller=None,native_teb=False,env_class=None,configuration=None):
    torch.set_num_threads(2);output=Path(output);output.mkdir(parents=True,exist_ok=True)
    e=(env_class or ComparisonEnv)(scene_dir=scene(),components=ROOT/'runs/remote_room_100k',max_steps=100000)
    if native_teb:
        from navrl.ros_teb import NativeTebController
        from navrl.map import continuous_remaining
        def native_observation(scan,initialize=False):
            e.scan=scan;e.remaining=continuous_remaining(e.path,e.estimate[:2]);e.last_policy_observation=np.zeros(235,np.float32);return e.last_policy_observation
        e._observation=native_observation
        controller=NativeTebController(e)
    model=PPO.load(policy,device='cpu') if controller is None else None
    samples=[];legs=[];obs,info=e.reset(seed=seed,options={'task':0});last_time=0.;last_velocity=np.zeros(2);effort=0.;steps=0
    try:
        for leg in range(4):
            if leg:obs=e.select_goal(leg)
            reached=False
            for j in range(max_leg_steps):
                action=model.predict(obs,deterministic=True)[0] if controller is None else controller.action(e)
                obs,_,done,_,info=e.step(action);steps+=1
                # Arrival requires the simulator endpoint to agree with laser-localized arrival.
                reached=bool(done and info['true_goal_distance_m']<=.30)
                velocity=np.asarray(info['actual_velocity']);effort+=float(np.abs(velocity-last_velocity).sum());last_velocity=velocity
                samples.append(dict(time_s=e.elapsed,active_target=leg+1,waypoints_completed=len(legs),truth=e.truth.tolist(),estimate=e.estimate.tolist(),qpos=e.data.qpos.tolist(),qvel=e.data.qvel.tolist(),scan=[float(x) if np.isfinite(x) else None for x in e.scan],planned_path=e.path.tolist(),active_goal=e.goal.tolist(),commanded_velocity=np.asarray(info['commanded_velocity']).tolist(),actual_velocity=velocity.tolist(),column_phases=dict(e.column_phases),interferences=list(e.interferences),route_switch_reason=info.get('route_switch_reason'),collisions=e.collision_events))
                if steps%100==0:print('PROGRESS',steps,'pose',e.estimate.tolist(),'command',np.asarray(info['commanded_velocity']).tolist(),'path_start',e.path[0].tolist(),flush=True)
                if reached:break
            legs.append(dict(target=leg+1,verified_success=reached,seconds=e.elapsed-last_time));last_time=e.elapsed
            if reached:samples[-1]['waypoints_completed']=len(legs)
            print(Path(policy).stem if policy else 'Nav2 TEB','TARGET',legs[-1],flush=True)
            if not reached:break
        result=dict(verified_success=len(legs)==4 and all(x['verified_success'] for x in legs),targets_reached=sum(x['verified_success'] for x in legs),targets_total=4,simulation_seconds=e.elapsed,travel_m=e.travel,velocity_total_variation=effort,mean_velocity_variation_per_step=effort/max(steps,1),collision_events=e.collision_events,penetration_guard_activations=e.hard_guard_activations,legs=legs,interferences=e.interferences,seed=seed,max_leg_steps=max_leg_steps,configuration=configuration,policy_path=str(policy) if policy else None,policy_sha256=hashlib.sha256(Path(policy).read_bytes()).hexdigest() if policy else None,robot_action_source='PPO' if model else 'native_nav2_teb',runtime_robot_amcl=False if model else True,runtime_robot_astar=False if model else True)
        # After arrival, stop the mission and let physical pillars complete their
        # pending hold/retreat. This tail is not counted as navigation completion time.
        if result['verified_success']:
            tail_start=e.elapsed
            while e.elapsed-tail_start<24.:
                released=all('released_time_s' in v or v.get('entry_aborted') for v in e.interferences)
                parked=all(np.linalg.norm(e._humans()[i]-point)<.15 for i,point in e.actor_destinations.items())
                if released and parked:break
                _,_,_,_,tail_info=e.step(np.zeros(2))
                tail=dict(samples[-1]);tail.update(time_s=e.elapsed,waypoints_completed=4,truth=e.truth.tolist(),estimate=e.estimate.tolist(),qpos=e.data.qpos.tolist(),qvel=e.data.qvel.tolist(),scan=[float(x) if np.isfinite(x) else None for x in e.scan],commanded_velocity=[0.,0.],actual_velocity=np.asarray(tail_info['actual_velocity']).tolist(),column_phases=dict(e.column_phases),interferences=list(e.interferences),collisions=e.collision_events,mission_finished=True)
                samples.append(tail)
            result['post_arrival_tail_seconds']=e.elapsed-tail_start
            result['interferences']=e.interferences
        trace=dict(scene_directory=str(SCENE),scene_file='physical_scene.xml',waypoints=e.scene['tasks'][0]['waypoints'],samples=samples,outcome=result,protocol=dict(trigger_distance_to_goal_m=3.2,column_speed_max_m_s=.65,hold_s=8.,entry_timeout_s=14.,schedule=SCHEDULE,robot_speed_max_m_s=(configuration or {}).get('forward_limit',.2),robot_yaw_rate_max_rad_s=(configuration or {}).get('yaw_limit',.5),laser_fov_deg=219.44,laser_beams=220,dt_s=.2))
        (output/'trace.json').write_text(json.dumps(trace));(output/'metrics.json').write_text(json.dumps(result,indent=2));print('RESULT',json.dumps(result),flush=True)
        return result
    finally:
        if controller:controller.close()
        e.close()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--policy',type=Path);p.add_argument('--nav2-teb',action='store_true');p.add_argument('--output',type=Path,required=True);p.add_argument('--seed',type=int,default=31002);p.add_argument('--max-leg-steps',type=int,default=900);a=p.parse_args();evaluate(a.policy,a.output,a.seed,max_leg_steps=a.max_leg_steps,native_teb=a.nav2_teb)
