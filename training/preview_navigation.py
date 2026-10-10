"""Loop a recorded native MuJoCo evaluation in an interactive viewer."""
import argparse
import json
import threading
import time
from pathlib import Path
import numpy as np
import mujoco
import mujoco.viewer
from navrl.geometry import ROOT, LidarProfile
from navrl.visualization import NavigationOverlay


def main():
    p=argparse.ArgumentParser();p.add_argument('trace',type=Path);a=p.parse_args()
    trace=json.loads(a.trace.read_text());samples=trace['samples']
    model=mujoco.MjModel.from_xml_path(str(Path(trace.get('scene_directory',ROOT/'assets'))/trace['scene_file']))
    data=mujoco.MjData(model);lock=threading.Lock();controls={'paused':False,'restart':False}
    def key(code):
        with lock:
            if code==32:controls['paused']=not controls['paused']
            elif code in (82,114):controls['restart']=True
    overlay=NavigationOverlay(model,LidarProfile(),trace['path'],trace['goal']);index=0;goal_hold=0.
    with mujoco.viewer.launch_passive(model,data,key_callback=key) as display:
        display.cam.type=mujoco.mjtCamera.mjCAMERA_FIXED;display.cam.fixedcamid=0
        print('MuJoCo preview ready. Space: pause/resume; R: restart; close window: exit.',flush=True)
        while display.is_running():
            started=time.monotonic()
            with lock:
                if controls['restart']:
                    index=0;goal_hold=0.;overlay.history.clear();controls['restart']=False
                paused=controls['paused']
            s=samples[index];data.qpos[:]=s['qpos'];data.qvel[:]=s['qvel'];mujoco.mj_forward(model,data)
            overlay.path=np.asarray(s.get('planned_path') or trace['path'])
            overlay.goal=np.asarray(s.get('active_goal',trace['goal']))
            scan=np.array([np.inf if x is None else x for x in s['scan']])
            with display.lock():
                display.user_scn.ngeom=0;overlay.draw(display.user_scn,data,s['truth'],scan)
            decision=s.get('decision') or {};v=s['commanded_velocity'];actual=s['actual_velocity']
            display.set_texts((None,None,f"Recorded PPO + {trace['localizer']}\nSimulation time\nDecision\nReason\nRoute changes\nPolicy v / actual v\nMotion source\nControls",f"{trace.get('map_id','')} {'target '+str(s.get('waypoint_index',0)+1)+'/'+str(len(trace['waypoints'])) if trace.get('waypoints') else ''}\n{s['time_s']:.1f} s\n{decision.get('state','')}\n{decision.get('reason','')}\n{decision.get('plan_revision',0)}\n{v[0]:.3f} / {actual[0]:.3f} m/s\n{s.get('motion_source','')}\nSpace: pause | R: restart"))
            display.sync()
            if not paused:
                if index<len(samples)-1:index+=1
                else:
                    goal_hold+=.2
                    if goal_hold>=5.:index=0;goal_hold=0.;overlay.history.clear()
            time.sleep(max(0.,.2-(time.monotonic()-started)))


if __name__=='__main__':main()
