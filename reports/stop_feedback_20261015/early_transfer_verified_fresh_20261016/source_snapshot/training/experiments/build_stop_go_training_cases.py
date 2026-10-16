"""Explicitly labelled variations of a known development failure family."""
import argparse
import copy
import json
from pathlib import Path


def build(cases):
    original=next(c for c in cases if c['id']=='motion_profile_015')
    previous=next(c for c in cases if c['id']=='train_failure_w11_e7')
    result=[]
    for i,(speed,yaw,dx,pause) in enumerate(((.9,-.06,-.04,3.8),(1.1,.06,.04,4.2))):
        case=copy.deepcopy(original)
        case.update(id=f'train_failure_stop_go_variation_{i}',seed=2010015+i,
            category='training_rehearsal',pilot_category='known_development_failure_family',
            obstacle_speed_m_s=original['obstacle_speed_m_s']*speed,
            actor_pause_after_event_s=[[2.,pause]])
        case['start'][0]+=dx;case['start'][2]+=yaw;result.append(case)
    result.append(copy.deepcopy(previous))
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--cases',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():p.error('Fresh output required')
    a.output.write_text(json.dumps(build(json.loads(a.cases.read_text())),indent=2)+'\n')
