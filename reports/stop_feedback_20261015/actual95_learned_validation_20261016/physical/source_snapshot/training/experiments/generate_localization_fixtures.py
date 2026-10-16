"""Fresh geometry fixtures; open map padding is not a physical boundary wall."""
import argparse
import copy
import json
import os
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
from PIL import Image
from navrl.geometry import ROOT


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    base=ROOT/'assets';template=ET.parse(base/'physical_scene.xml').getroot()
    config=json.loads((base/'physical_scene.json').read_text())
    boundary=[[0,0,10,.2],[0,7.8,10,8],[0,.2,.2,7.8],[9.8,.2,10,7.8]]
    layouts={
        'informative_closed':boundary+[[2,2,2.4,2.6],[7,5,7.6,5.4]],
        'single_long_wall':[[.8,3.9,9.2,4.1]],
        'parallel_corridor':[[.8,2.2,9.2,2.4],[.8,5.6,9.2,5.8]],
        'sparse_asymmetric':[[2,2,2.25,2.35],[7,5,7.45,5.25],[8,1.5,8.2,1.8]],
        'repeated_columns':[[x,2.5,x+.3,2.8] for x in (1.,3.,5.,7.,9.)]+[[x,5.2,x+.3,5.5] for x in (1.,3.,5.,7.,9.)],
        'feature_to_open':[[.3,.3,.5,5.],[.5,.3,4.,.5],[3.8,.5,4.,3.5],[1.8,3.,2.1,3.4]]}
    rows=[]
    for name,walls in layouts.items():
        folder=a.output/name;folder.mkdir();model=copy.deepcopy(template);world=model.find('worldbody')
        for geom in list(world.findall('geom')):
            if geom.get('name','').startswith('wall_'):world.remove(geom)
        for asset in model.findall('./asset/*'):
            filename=asset.get('file')
            if filename:asset.set('file',os.path.relpath((base/filename).resolve(),folder.resolve()))
        for i,(x0,y0,x1,y1) in enumerate(walls):
            ET.SubElement(world,'geom',name=f'wall_{i}',type='box',pos=f'{(x0+x1)/2} {(y0+y1)/2} .5',
                size=f'{(x1-x0)/2} {(y1-y0)/2} .5',rgba='.35 .4 .45 1')
        ET.indent(model);ET.ElementTree(model).write(folder/'physical_scene.xml',encoding='unicode')
        yy,xx=np.mgrid[0:160,0:200];x=(xx+.5)*.05;y=8-(yy+.5)*.05;occupied=np.zeros(x.shape,bool)
        for x0,y0,x1,y1 in walls:occupied|=(x>=x0)&(x<x1)&(y>=y0)&(y<y1)
        image=np.full(x.shape,254,np.uint8);image[[0,-1],:]=127;image[:,[0,-1]]=127;image[occupied]=0
        Image.fromarray(image).save(folder/'decision_map.pgm')
        (folder/'decision_map.yaml').write_text('image: decision_map.pgm\nresolution: 0.05\norigin: [0, 0, 0]\nnegate: 0\noccupied_thresh: 0.65\nfree_thresh: 0.25\n')
        c=copy.deepcopy(config);c.update(walls=walls,map_id=name,split='development_unseen_geometry')
        (folder/'physical_scene.json').write_text(json.dumps(c,indent=2))
        rows.append(str(folder.resolve().relative_to(ROOT)))
    (a.output/'scenes.json').write_text(json.dumps(rows,indent=2));print(json.dumps(rows,indent=2))


if __name__=='__main__':main()
