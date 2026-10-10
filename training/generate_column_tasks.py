"""Construct long-distance test tasks; offline geometry is only world/task setup."""
import argparse
from pathlib import Path
from PIL import Image
import copy,json,shutil,xml.etree.ElementTree as ET
import numpy as np
from navrl.geometry import ROOT
from navrl.map import StaticMap


def dense(path):
    points=[]
    for a,b in zip(path[:-1],path[1:]):
        n=max(1,int(np.ceil(np.linalg.norm(b-a)/.10)))
        points.extend(a+(b-a)*j/n for j in range(n))
    return np.asarray(points+[path[-1]])


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=ROOT/'assets/column_navigation_tasks_v2');args=parser.parse_args()
    out=args.output;out.mkdir(exist_ok=True);rng=np.random.default_rng(25000)
    entries=[]
    for i in range(3):
        src=ROOT/f'assets/navigation_dataset_v2/test_{i:02d}';folder=out/f'columns_{i:02d}';folder.mkdir(exist_ok=True)
        for f in ['decision_map.pgm','decision_map.yaml']:shutil.copy2(src/f,folder/f)
        tree=ET.parse(src/'physical_scene.xml');root=tree.getroot()
        config=json.loads((src/'physical_scene.json').read_text())
        walls=config['walls']
        if i==2:
            lower=walls[4];top=[lower[0],lower[3]+1.7,lower[2],7.8]
            j=len(walls);walls.append(top)
            ET.SubElement(root.find('worldbody'),'geom',name=f'wall_{j}',type='box',pos=f'{(top[0]+top[2])/2} {(top[1]+top[3])/2} .5',size=f'{(top[2]-top[0])/2} {(top[3]-top[1])/2} .5',rgba='.35 .4 .45 1')
            yy,xx=np.mgrid[0:160,0:200];x=(xx+.5)*.05;y=8-(yy+.5)*.05;occupied=np.zeros(x.shape,bool)
            for x0,y0,x1,y1 in walls:occupied|=(x>=x0)&(x<x1)&(y>=y0)&(y<y1)
            Image.fromarray(np.where(occupied,0,254).astype(np.uint8)).save(folder/'decision_map.pgm')
        if i==1:
            gap=np.array([(walls[4][2]+walls[5][0])/2,(walls[4][1]+walls[4][3])/2]);gap_normal=np.array([1.,0.]);cross_axis=1
        else:
            top=walls[5] if i==0 else walls[-1]
            gap=np.array([(walls[4][0]+walls[4][2])/2,(walls[4][3]+top[1])/2]);gap_normal=np.array([0.,1.]);cross_axis=0
        for j,o in enumerate(config['dynamic_obstacles']):
            body=root.find(".//body[@name='%s']"%o['name'])
            for g in list(body.findall('geom')):
                if g.get('name')!=o['name']+'_geom':body.remove(g)
                else:g.set('rgba','0.95 0.42 0.10 1' if j==0 else '0.16 0.48 0.86 1')
            o['kind']='moving_solid_column';o['height_m']=1.7
        config.update(map_id=folder.name,dynamic_obstacle_kind='columns',tasks=[],column_behavior='Solid acceleration-limited moving columns; approach-triggered blocking then physical release',task_source='Offline scene/task geometry; no routes supplied to robot inference')
        m=StaticMap(folder/'decision_map.yaml',radius=.30);cells=np.argwhere(m.clearance>.75)
        for kind in ['bypass','temporary_chokepoint']:
            for attempt in range(1500):
                a,b=[m.xy(cells[int(rng.integers(len(cells)))]) for _ in range(2)]
                if not 6.5<=np.linalg.norm(a-b)<=9.5:continue
                if kind=='temporary_chokepoint' and (a[cross_axis]-gap[cross_axis])*(b[cross_axis]-gap[cross_axis])>=0:continue
                try:path=m.plan(a,b)
                except ValueError:continue
                length=float(np.linalg.norm(np.diff(path,axis=0),axis=1).sum())
                if not 7.<=length<=15.:continue
                pts=dense(path);candidates=[]
                for k in range(int(len(pts)*.30),int(len(pts)*.65)):
                    point=pts[k];tangent=pts[min(k+2,len(pts)-1)]-pts[max(0,k-2)]
                    tangent/=max(np.linalg.norm(tangent),1e-6);normal=np.array([-tangent[1],tangent[0]])
                    if np.linalg.norm(point-a)<2.5 or np.linalg.norm(point-b)<2.:continue
                    if kind=='bypass':
                        if m.lookup(point)>.95:candidates.append((float(m.lookup(point)),point,[point],normal))
                    else:
                        # Center paired columns within the local passage rather than against a wall.
                        for shift in np.linspace(-.5,.5,11):
                            center=point+shift*normal;pair=[center-.36*normal,center+.36*normal]
                            clearance=float(m.lookup(center))
                            if .65<clearance<1.05 and all(m.lookup(p)>.32 for p in pair):
                                candidates.append((-clearance,center,pair,normal))
                if kind=='temporary_chokepoint':
                    pair=[gap-.36*gap_normal,gap+.36*gap_normal]
                    candidates=[(0.,gap,pair,gap_normal)] if all(m.lookup(p)>.32 for p in pair) and np.linalg.norm(a-gap)>2.5 and np.linalg.norm(b-gap)>2. else []
                if not candidates:continue
                _,center,positions,normal=max(candidates,key=lambda c:c[0]);blocks=[];used=[]
                for j,point in enumerate(positions):
                    release_cells=np.argwhere((m.clearance>.65))
                    choices=[m.xy(c) for c in release_cells]
                    choices.sort(key=lambda p:np.linalg.norm(p-(point+normal*(1.4 if j==0 else -1.4))))
                    release=None
                    for g in choices:
                        if not 1.0<np.linalg.norm(g-point)<3.:continue
                        if np.linalg.norm(g-a)<.8 or any(np.linalg.norm(g-o)<.75 for o in used):continue
                        try:m.plan(point,g)
                        except ValueError:continue
                        release=g;break
                    if release is None:break
                    used.append(release)
                    blocks.append(dict(column_index=j,position=point.tolist(),release_goal=release.tolist(),hold_seconds=12 if kind=='bypass' else 18,trigger_distance_m=2.5 if kind=='bypass' else 1.5))
                if len(blocks)!=len(positions):continue
                heading=float(np.arctan2(*(path[1]-path[0])[::-1]))
                config['tasks'].append(dict(name=kind,start=[*a,heading],goal=b.tolist(),straight_distance_m=float(np.linalg.norm(a-b)),offline_reference_length_m=length,column_task=dict(kind=kind,blocks=blocks)))
                break
            else:raise RuntimeError('Unable to construct safe long-distance '+kind)
        ET.indent(root);tree.write(folder/'physical_scene.xml',encoding='unicode')
        (folder/'physical_scene.json').write_text(json.dumps(config,indent=2))
        entries.append(dict(directory=str(folder),map_id=folder.name,tasks=config['tasks']))
    (out/'manifest.json').write_text(json.dumps(dict(entries=entries,model_weights='Frozen full_model_navigation_v2; no retraining for these tasks',maps='Same known layouts as v2 test; new long-distance tasks and column schedules'),indent=2))
    for e in entries:
        print(e['map_id'],[(t['name'],round(t['straight_distance_m'],2),round(t['offline_reference_length_m'],2)) for t in e['tasks']],flush=True)


if __name__=='__main__':main()
