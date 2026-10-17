"""One-shot, proximity-triggered doorway blocking; no random destinations."""
import numpy as np


def velocity(env,index):
    w=env.walkers[index];b=w['block'];pos=env._humans()[index]
    group=[(j,o['block']) for j,o in enumerate(env.walkers) if o.get('block',{}).get('mode')=='approach_cross_return' and o['block']['group']==b['group']]
    if b['phase']=='armed':
        if np.linalg.norm(env.truth[:2]-np.asarray(b['trigger_position']))>b['trigger_distance_m']:
            w['state']='waiting_beside_door';return np.zeros(2)
        for j,other in group:
            other.update(phase='entering',trigger_time_s=float(env.elapsed))
    if b['phase']=='entering':
        positions=env._humans()
        if all(np.linalg.norm(positions[j]-np.asarray(other['block_position']))<.07 for j,other in group):
            for j,other in group:
                other.update(phase='holding',entered_time_s=float(env.elapsed),release_time_s=float(env.elapsed+other['hold_seconds']))
        elif env.elapsed-b['trigger_time_s']>=b['entry_timeout_s']:
            # If the robot prevents safe entry, retreat physically; never push through it.
            for j,other in group:
                other.update(phase='releasing',release_time_s=float(env.elapsed),entry_aborted=True)
    if b['phase']=='holding':
        if env.elapsed<b['release_time_s']:
            w['state']='temporary_block';return np.zeros(2)
        for j,other in group:
            if other['phase']=='holding':other['phase']='releasing'
    if b['phase']=='releasing' and np.linalg.norm(pos-np.asarray(b['park_position']))<.07:
        b.update(phase='cleared',cleared_time_s=float(env.elapsed))
    if b['phase']=='cleared':
        w['state']='parked_beside_door';return np.zeros(2)
    target=np.asarray(b['block_position'] if b['phase']=='entering' else b['park_position'])
    delta=target-pos;norm=np.linalg.norm(delta)
    if norm<.045:
        w['state']='waiting_for_partner' if b['phase']=='entering' else 'parking';return np.zeros(2)
    desired=delta/max(norm,1e-8)*min(w['speed'],2*norm)
    future=pos+desired*.25
    closing=np.dot(desired,env.truth[:2]-pos)>0
    if closing and (env.robot_gap(pos)<.35 or env.robot_gap(future)<.14):
        w['state']='yielding';return np.zeros(2)
    if env.wall_gap(future)<.015 or not env.human_map.visible(pos,future) or any(j!=index and np.linalg.norm(future-other)<.60 for j,other in enumerate(env._humans())):
        w['state']='blocked_on_script';return np.zeros(2)
    w['state']='moving_into_door' if b['phase']=='entering' else 'returning_beside_door'
    return desired
