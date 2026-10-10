"""Disjoint map layouts for training, validation and held-out evaluation."""
import argparse
from pathlib import Path
import copy
import json
import hashlib
import xml.etree.ElementTree as ET
import numpy as np
from PIL import Image
from navrl.geometry import ROOT
from navrl.map import StaticMap


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=ROOT/'assets/navigation_dataset');p.add_argument('--test-offset',type=int,default=33000);a=p.parse_args()
    base=ROOT/'assets';output=a.output; output.mkdir(exist_ok=True)
    template=ET.parse(base/'physical_scene.xml').getroot()
    config=json.loads((base/'physical_scene.json').read_text())
    manifest={'version':1,'size_m':[10,8],'splits':{},'sampling':'random free start/goal >= 2 m apart; randomized full-circle initial heading'}
    for split,count,offset in [('train',6,11000),('validation',2,22000),('test',3,a.test_offset)]:
        manifest['splits'][split]=[]
        for i in range(count):
            rng=np.random.default_rng(offset+i); folder=output/f'{split}_{i:02d}';folder.mkdir(exist_ok=True)
            walls=[[0,0,10,.2],[0,7.8,10,8],[0,.2,.2,7.8],[9.8,.2,10,7.8]]
            kind=i%3
            if kind==0:
                x=float(rng.uniform(3.2,6.5)); y=float(rng.uniform(2.1,5.7))
                walls += [[x-.1,.2,x+.1,y-.85],[x-.1,y+.85,x+.1,7.8]]
            elif kind==1:
                y=float(rng.uniform(3,5)); x=float(rng.uniform(2.6,7.4))
                walls += [[.2,y-.1,x-.85,y+.1],[x+.85,y-.1,9.8,y+.1]]
            else:
                walls += [[3,.2,3.2,float(rng.uniform(3.6,4.8))],[6.5,float(rng.uniform(2.7,3.7)),6.7,7.8]]
            for j in range(3):
                x,y=rng.uniform([1,1],[8.5,6.5]);w,h=rng.uniform(.35,.7,2)
                walls.append([float(x),float(y),float(x+w),float(y+h)])
            yy,xx=np.mgrid[0:160,0:200]; x=(xx+.5)*.05;y=8-(yy+.5)*.05
            occupied=np.zeros(x.shape,bool)
            for x0,y0,x1,y1 in walls:occupied|=(x>=x0)&(x<x1)&(y>=y0)&(y<y1)
            Image.fromarray(np.where(occupied,0,254).astype(np.uint8)).save(folder/'decision_map.pgm')
            (folder/'decision_map.yaml').write_text((base/'decision_map.yaml').read_text())
            model=copy.deepcopy(template);model.set('model',f'agv_{split}_{i:02d}');world=model.find('worldbody')
            for g in list(world.findall('geom')):
                if g.get('name','').startswith('wall_'):world.remove(g)
            for j,(x0,y0,x1,y1) in enumerate(walls):
                ET.SubElement(world,'geom',name=f'wall_{j}',type='box',pos=f'{(x0+x1)/2} {(y0+y1)/2} .5',size=f'{(x1-x0)/2} {(y1-y0)/2} .5',rgba='.35 .4 .45 1')
            ET.indent(model);ET.ElementTree(model).write(folder/'physical_scene.xml',encoding='unicode')
            m=StaticMap(folder/'decision_map.yaml',radius=.30);cells=np.argwhere(m.clearance>.65)
            for attempt in range(500):
                a,b=[m.xy(cells[int(rng.integers(len(cells)))]) for _ in range(2)]
                if np.linalg.norm(a-b)<2:continue
                try:m.plan(a,b);break
                except ValueError:continue
            else:raise RuntimeError('Invalid dataset layout')
            c=copy.deepcopy(config);c.update(walls=walls,map_id=folder.name,split=split,generation_seed=offset+i,tasks=[dict(start=[*a,0],goal=b.tolist())])
            (folder/'physical_scene.json').write_text(json.dumps(c,indent=2))
            manifest['splits'][split].append(dict(id=folder.name,directory=str(folder),seed=offset+i,sha256=hashlib.sha256((folder/'decision_map.pgm').read_bytes()).hexdigest()))
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print('Generated 6 training, 2 validation and 3 unseen test maps:',output)


if __name__=='__main__':main()
