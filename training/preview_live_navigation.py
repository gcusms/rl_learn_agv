"""Actual local policy inference and physics with a mouse-controlled MuJoCo camera."""
import argparse,json,time,threading,hashlib
from pathlib import Path
import numpy as np
import torch
import mujoco
from stable_baselines3 import PPO
from navrl.geometry import ROOT
from navrl.visualization import NavigationOverlay
from neural_navigation.waypoint_interference import build_scene,WaypointInterferenceEnv

def main():
    p=argparse.ArgumentParser();p.add_argument('--policy',type=Path,default=ROOT/'runs/remote_room_100k/policy_50000.zip');p.add_argument('--components',type=Path,default=ROOT/'runs/remote_room_100k');p.add_argument('--seed',type=int,default=31002);p.add_argument('--headless',action='store_true');p.add_argument('--max-leg-steps',type=int,default=900);p.add_argument('--output',type=Path,default=ROOT/'runs/local_manual_two_live');a=p.parse_args()
    torch.set_num_threads(2);a.output.mkdir(exist_ok=True);scene=build_scene();env=WaypointInterferenceEnv(components=a.components,scene_dir=scene,max_steps=100000)
    model=PPO.load(a.policy,device='cpu');controls={'pause':False,'restart':False,'next':False,'camera':False};lock=threading.Lock();viewer=None;next_step_time=0.
    def key(code):
        with lock:
            if code==32:controls['pause']=not controls['pause']
            elif code in (82,114):controls['restart']=True
            elif code in (78,110):controls['next']=True
            elif code in (67,99):controls['camera']=True
    def camera():
        viewer.cam.type=mujoco.mjtCamera.mjCAMERA_FREE;viewer.cam.lookat[:]=[5,4,.4];viewer.cam.distance=12.8;viewer.cam.azimuth=90;viewer.cam.elevation=-65
    def reset():
        obs,info=env.reset(seed=a.seed,options={'task':0});return obs,info,0,0,[],[],False
    def save():
        result=dict(policy_path=str(a.policy.resolve()),policy_sha256=hashlib.sha256(a.policy.read_bytes()).hexdigest(),seed=a.seed,simulation_seconds=env.elapsed,active_target=leg+1,legs=legs,complete_success=len(legs)==4 and all(x['verified_success'] and not x['manually_skipped'] for x in legs),collision_events=env.collision_events,penetration_guard_activations=env.hard_guard_activations,manual_mode=env.manual_mode,manual_interventions=env.manual_events,column_block_events=info.get('column_block_events',[]),selected_column=(viewer.selected+1 if viewer and viewer.selected is not None else None),runtime_robot_amcl=False,runtime_robot_astar=False,supervisor_override=False,planning_fusion=info.get('planning_fusion'),route_switch_reason=info.get('route_switch_reason'),far_unconfirmed_cells=info.get('far_unconfirmed_cells'),cleared_dynamic_cells=info.get('cleared_dynamic_cells'))
        (a.output/'status.json').write_text(json.dumps(result,indent=2))
    try:
        obs,info,leg,steps,legs,samples,finished=reset();overlay=NavigationOverlay(env.model,env.lidar,env.path,env.goal)
        if not a.headless:
            from navrl.native_viewer import NativeManualNavigationViewer
            viewer=NativeManualNavigationViewer(env,key_callback=key);camera();viewer.sync()
        print('LIVE POLICY: 50k checkpoint (unless --policy changed). Mouse left-drag: rotate; right-drag: pan; wheel: zoom; Shift: horizontal adjustment. Double-click column to select; hold right mouse to drag; release right mouse to deselect. Obstacles are manual only. Space: pause; R: reset all; N: manually skip target; C: reset camera.',flush=True)
        while a.headless or viewer.is_running():
            started=time.monotonic()
            if viewer:viewer.poll_events()
            with lock:
                paused=controls['pause'];restart=controls['restart'];advance=controls['next'];reset_camera=controls['camera'];controls.update(restart=False,next=False,camera=False)
            if viewer and hasattr(viewer,'simulation_paused'):paused=viewer.simulation_paused
            if restart:
                obs,info,leg,steps,legs,samples,finished=reset();overlay.history.clear()
                if viewer:viewer.selected=None;viewer.dragging=False
                save()
            if reset_camera and viewer:camera()
            if not paused and not finished and (a.headless or started>=next_step_time):
                next_step_time=started+env.dt
                if not advance:
                    action=model.predict(obs,deterministic=True)[0];obs,_,done,truncated,info=env.step(action);steps+=1
                    samples.append(dict(time_s=env.elapsed,active_target=leg+1,truth=env.truth.tolist(),estimate=env.estimate.tolist(),qpos=env.data.qpos.tolist(),qvel=env.data.qvel.tolist(),scan=[float(x) if np.isfinite(x) else None for x in env.scan],planned_path=env.path.tolist(),action=np.asarray(action).tolist(),manual_mode=env.manual_mode,column_states=[w['state'] for w in env.walkers],column_block_events=info['column_block_events'],supervisor_override=False,route_switch_reason=info.get('route_switch_reason'),far_unconfirmed_cells=info.get('far_unconfirmed_cells'),cleared_dynamic_cells=info.get('cleared_dynamic_cells')))
                else:done=False;truncated=False
                if done or steps>=a.max_leg_steps or advance:
                    verified=bool(done and info['true_goal_distance_m']<=.30)
                    legs.append(dict(target=leg+1,verified_success=verified,manually_skipped=bool(advance),timeout=bool(steps>=a.max_leg_steps and not done),seconds=env.elapsed,manual_interventions=sum(x['active_target']==leg+1 for x in env.manual_events)))
                    print('TARGET RESULT',legs[-1],flush=True)
                    if verified and leg<3 or advance and leg<3:
                        leg+=1;steps=0;obs=env.select_goal(leg);info=env._info('running')
                    else:finished=True
                    save();(a.output/'replay.json').write_text(json.dumps(dict(scene_directory=str(scene),waypoints=env.scene['tasks'][0]['waypoints'],samples=samples,result=json.loads((a.output/'status.json').read_text()))))
                elif steps%10==0:save()
            elif advance and finished and leg<3:
                # Manual exploration is labeled and never counted as a successful mission.
                legs[-1]['manually_skipped']=True;leg+=1;steps=0;obs=env.select_goal(leg);info=env._info('running');finished=False;save()
            if a.headless:
                if finished:break
                continue
            with viewer.lock():
                viewer.user_scn.ngeom=0;overlay.path=env.path;overlay.goal=env.goal;overlay.draw(viewer.user_scn,env.data,env.truth,env.scan)
                for i,point in enumerate(env.scene['tasks'][0]['waypoints']):
                    sc=viewer.user_scn
                    if sc.ngeom>=sc.maxgeom:break
                    g=sc.geoms[sc.ngeom];sc.ngeom+=1;mujoco.mjv_initGeom(g,mujoco.mjtGeom.mjGEOM_SPHERE,np.array([.1,0,0]),np.array([*point,.12]),np.eye(3).reshape(-1),np.array([.7,.25,.9,.8],np.float32));g.label=str(i+1)
                for i,point in enumerate(env._humans()):
                    sc=viewer.user_scn
                    if sc.ngeom>=sc.maxgeom:break
                    g=sc.geoms[sc.ngeom];sc.ngeom+=1;mujoco.mjv_initGeom(g,mujoco.mjtGeom.mjGEOM_SPHERE,np.array([.07,0,0]),np.array([*point,1.78]),np.eye(3).reshape(-1),np.array([1.,1.,.1,1.],np.float32));g.label=f'C{i+1}'+(' SELECTED' if viewer.selected==i else '')
            chosen=viewer.selected
            phase='MANUAL '+('none selected' if chosen is None else f'C{chosen+1} '+env.walkers[chosen]['state']);state='FINISHED / TIMEOUT - R restart, N skip' if finished else 'PAUSED' if paused else 'LIVE inference'
            viewer.set_texts((None,None,f'Local PPO {a.policy.stem} | Learned laser pose + neural route\nTarget / interference\nTime / collisions / guards\nMap + laser route\nControls',f'{state}\n{leg+1}/4 | {phase}\n{env.elapsed:.1f}s / {env.collision_events} / {env.hard_guard_activations}\n{info.get("route_switch_reason", "initial")} | far pending: {info.get("far_unconfirmed_cells", 0)}\nDouble-click column | Right-drag | Release deselects | Space pause | R restart'))
            viewer.sync();time.sleep(max(0,1/60-(time.monotonic()-started)))
        save()
    finally:
        if viewer:viewer.close()
        env.close()

if __name__=='__main__':main()
