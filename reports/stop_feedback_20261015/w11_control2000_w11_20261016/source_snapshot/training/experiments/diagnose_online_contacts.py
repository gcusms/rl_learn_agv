"""Replay recorded commands to identify contacts; truth never selects commands."""
import argparse,json
from pathlib import Path
import numpy as np
import mujoco
from navrl.geometry import ROOT
from neural_navigation.stress_env import StressNavigationEnv
from experiments.probe_online_slam import relative_pose


def main():
    p=argparse.ArgumentParser();p.add_argument('--evaluation',type=Path,required=True)
    p.add_argument('--bundle',type=Path,required=True);a=p.parse_args()
    report=json.loads((a.evaluation/'RESULTS.json').read_text());cases=json.loads((a.evaluation/'cases.json').read_text());results=[]
    for row in report['rows']:
        if not row['collisions']:continue
        case=next(c for c in cases if c['id']==row['case_id'])
        trace=json.loads((a.evaluation/case['id']/'trace.json').read_text())
        env=StressNavigationEnv(case=case,components=a.bundle,scene_dir=ROOT/case['scene'],route_mode='shortcut',max_steps=1000)
        forward=mujoco.mj_forward;observed_contacts={}
        try:
            env.reset(seed=case['seed'],options=dict(start=case['start'],goal=case['goal'][:2]));reference=env.truth.copy()
            for _ in range(row['bootstrap_frames']):env.step(np.array([0.,1.]))
            for _ in range(5):env.step(np.zeros(2))
            def capture_forward(model,data):
                forward(model,data)
                for contact in data.contact:
                    pair={int(contact.geom1),int(contact.geom2)}
                    if not (pair&env.robot_geoms and (pair&env.wall_geoms or pair&set(env.human_geoms))):continue
                    names=tuple(mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_GEOM,int(g)) for g in [contact.geom1,contact.geom2])
                    observed_contacts[names]=min(observed_contacts.get(names,float('inf')),float(contact.dist))
            mujoco.mj_forward=capture_forward
            errors=[]
            for frame in trace:
                errors.append(float(np.linalg.norm(relative_pose(env.truth,reference)[:2]-np.array(frame['truth'])[:2])))
                env.step(np.asarray(frame['command'])/[.2,.5])
                if env.collision_events:break
            pairs=[dict(geometry=list(names),minimum_signed_distance_m=distance) for names,distance in observed_contacts.items()]
            result=dict(case_id=case['id'],command_replay_max_xy_difference_m=max(errors),
                reproduced_collision=bool(env.collision_events),contacts=pairs,
                robot_command_at_contact=trace[len(errors)-1]['command'],scope='independent physical contact labels, not policy inputs')
            results.append(result);print(json.dumps(result),flush=True)
        finally:mujoco.mj_forward=forward;env.close()
    (a.evaluation/'CONTACT_DIAGNOSIS.json').write_text(json.dumps(results,indent=2))


if __name__=='__main__':main()
