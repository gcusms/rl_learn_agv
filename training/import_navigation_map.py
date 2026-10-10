"""Extrude a ROS occupancy map into a paired native MuJoCo navigation scene."""
import argparse
import copy
import json
import shutil
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import yaml
from navrl.geometry import ROOT
from navrl.map import StaticMap


def main():
    p=argparse.ArgumentParser();p.add_argument('map_yaml',type=Path);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);source=a.map_yaml.resolve()
    cfg=yaml.safe_load(source.read_text());m=StaticMap(source,radius=.30)
    shutil.copyfile(source.parent/cfg['image'],a.output/'decision_map.pgm')
    cfg['image']='decision_map.pgm';(a.output/'decision_map.yaml').write_text(yaml.safe_dump(cfg))
    # Merge identical horizontal occupancy runs on consecutive rows into rectangles.
    active={};rectangles=[]
    for row in range(m.height+1):
        occupied=m.blocked[row] if row<m.height else np.zeros(m.width,bool)
        edges=np.diff(np.r_[False,occupied,False].astype(int));runs=set(zip(np.flatnonzero(edges==1),np.flatnonzero(edges==-1)))
        for run,(begin,last) in list(active.items()):
            if run not in runs:rectangles.append((begin,row,*run));del active[run]
        for run in runs:
            if run not in active:active[run]=(row,row)
    walls=[]
    for r0,r1,c0,c1 in rectangles:
        walls.append([float(m.origin[0]+c0*m.res),float(m.origin[1]+(m.height-r1)*m.res),float(m.origin[0]+c1*m.res),float(m.origin[1]+(m.height-r0)*m.res)])
    model=ET.parse(ROOT/'assets/physical_scene.xml').getroot();world=model.find('worldbody')
    for g in list(world.findall('geom')):
        if g.get('name','').startswith('wall_'):world.remove(g)
    for i,(x0,y0,x1,y1) in enumerate(walls):
        ET.SubElement(world,'geom',name=f'wall_{i}',type='box',pos=f'{(x0+x1)/2} {(y0+y1)/2} .5',size=f'{(x1-x0)/2} {(y1-y0)/2} .5',rgba='.35 .4 .45 1')
    w,h=m.width*m.res,m.height*m.res;cx,cy=m.origin[:2]+[w/2,h/2]
    world.find('camera').set('pos',f'{cx} {cy} {max(w,h)*1.4}')
    world.find('light').set('pos',f'{cx} {cy} {max(w,h)}')
    ET.indent(model);ET.ElementTree(model).write(a.output/'physical_scene.xml',encoding='unicode')
    cells=np.argwhere(m.clearance>.60)
    if len(cells)<2:raise ValueError('Map has insufficient safe spawn space')
    rng=np.random.default_rng(45)
    for _ in range(500):
        start,goal=[m.xy(cells[int(rng.integers(len(cells)))]) for _ in range(2)]
        if np.linalg.norm(start-goal)<2:continue
        try:m.plan(start,goal);break
        except ValueError:continue
    else:raise ValueError('Map has no sampled reachable navigation task of at least 2 m')
    c=json.loads((ROOT/'assets/physical_scene.json').read_text())
    c.update(map_id=a.output.name,source_map=str(source),size_m=[w,h],origin=m.origin.tolist(),resolution_m=m.res,walls=walls,tasks=[dict(start=[*start,0],goal=goal.tolist())],map_import_assumptions='Occupied and unknown cells extruded to 1 m high static walls; floor level, axis-aligned map; map image is not a reconstruction of object heights')
    (a.output/'physical_scene.json').write_text(json.dumps(c,indent=2))
    print('Imported',len(walls),'wall rectangles to',a.output)


if __name__=='__main__':main()
