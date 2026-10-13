"""Four connected rooms, numbered mission goals and solid doorway columns."""
import argparse, json, shutil, xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
from PIL import Image
from navrl.geometry import ROOT
from navrl.arm_home import apply_home_pose
from navrl.map import StaticMap


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=ROOT/'assets/room_navigation_scripted');args=p.parse_args()
    src=ROOT/'assets/column_navigation_tasks_v2/columns_00';out=args.output;out.mkdir(parents=True,exist_ok=True)
    tree=ET.parse(src/'physical_scene.xml');root=tree.getroot();world=root.find('worldbody')
    for geom in list(world.findall('geom')):
        if geom.get('name','').startswith('wall_'):world.remove(geom)
    walls=[[0,0,10,.2],[0,7.8,10,8],[0,.2,.2,7.8],[9.8,.2,10,7.8],
           [4.9,.2,5.1,3.1],[4.9,4.9,5.1,7.8],
           [.2,3.9,1.6,4.1],[3.4,3.9,4.9,4.1],
           [5.1,3.9,6.6,4.1],[8.4,3.9,9.8,4.1]]
    for i,(x0,y0,x1,y1) in enumerate(walls):
        ET.SubElement(world,'geom',name=f'wall_{i}',type='box',pos=f'{(x0+x1)/2} {(y0+y1)/2} .5',size=f'{(x1-x0)/2} {(y1-y0)/2} .5',rgba='.35 .4 .45 1')
    # Decorative room floor patches are below the laser and have no contacts.
    for i,(x0,y0,x1,y1) in enumerate([[.2,.2,4.9,3.9],[.2,4.1,4.9,7.8],[5.1,.2,9.8,3.9],[5.1,4.1,9.8,7.8]]):
        ET.SubElement(world,'geom',name=f'room_floor_{i}',type='box',pos=f'{(x0+x1)/2} {(y0+y1)/2} -.015',size=f'{(x1-x0)/2} {(y1-y0)/2} .01',rgba=['.88 .91 .95 1','.92 .90 .84 1','.86 .93 .89 1','.93 .87 .89 1'][i],contype='0',conaffinity='0')
    home=apply_home_pose(root);ET.indent(root);tree.write(out/'physical_scene.xml',encoding='unicode')
    shutil.copy2(src/'decision_map.yaml',out/'decision_map.yaml')
    yy,xx=np.mgrid[0:160,0:200];x=(xx+.5)*.05;y=8-(yy+.5)*.05;occupied=np.zeros(x.shape,bool)
    for x0,y0,x1,y1 in walls:occupied|=(x>=x0)&(x<x1)&(y>=y0)&(y<y1)
    Image.fromarray(np.where(occupied,0,254).astype(np.uint8)).save(out/'decision_map.pgm')
    config=json.loads((src/'physical_scene.json').read_text());config.update(walls=walls,map_id='four_rooms',arm_home=home,
        rooms=[dict(name=n,bounds=b) for n,b in zip(['Room A','Room B','Room C','Room D'],[[.2,.2,4.9,3.9],[.2,4.1,4.9,7.8],[5.1,.2,9.8,3.9],[5.1,4.1,9.8,7.8]])],
        door_width_m=1.8,column_behavior='Wait beside doorway, enter when robot is within 2.2 m, hold 12 s after entry, return and park; no random roaming',task_source='Externally specified mission waypoints; robot receives only current target, known map and sensors',tasks=[])
    orders=[[[1.2,6.6],[8.6,6.6],[8.6,1.2],[1.2,1.2]],[[8.6,6.6],[1.2,6.6],[8.6,1.2],[1.2,1.2]],[[8.6,1.2],[8.6,6.6],[1.2,6.6],[1.2,1.2]]]
    m=StaticMap(out/'decision_map.yaml',radius=.30)
    for i,points in enumerate(orders):
        start=[1.2,1.2,1.57 if i==0 else 0.]
        previous=np.array(start[:2]);lengths=[]
        for point in points:
            path=m.plan(previous,np.array(point));lengths.append(float(np.linalg.norm(np.diff(path,axis=0),axis=1).sum()));previous=np.array(point)
        blocks=[dict(column_index=j,mode='approach_cross_return',position=[1.25 if j==0 else 3.75,5.2],block_position=[2.14+.72*j,4.],release_goal=[1.25 if j==0 else 3.75,5.2],trigger_position=[2.5,4.],hold_seconds=12,trigger_distance_m=2.2,speed_m_s=.45) for j in range(2)]
        config['tasks'].append(dict(name=f'four_room_mission_{i+1}',start=start,goal=points[0],waypoints=points,offline_leg_lengths_m=lengths,column_task=dict(kind='approach_cross_return',blocks=blocks)))
    (out/'physical_scene.json').write_text(json.dumps(config,indent=2));(out/'arm_home.json').write_text(json.dumps(home,indent=2))
    print('Room scene:',out,'home:',home['joint_positions_rad'],'FK error:',home['cad_fk_error_m'],flush=True)

if __name__=='__main__':main()
