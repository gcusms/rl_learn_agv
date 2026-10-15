"""Narrow L junctions, with and without a wider corner, retaining native robot."""
import json,os
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
from PIL import Image


def generate(output,hardware):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    template=Path('training/assets/room_navigation_scripted');manifest=[]
    for extra in (.15,.125,.10):
        width=hardware['robot_footprint_m'][1]+extra
        for buffer in (False,True):
            directory=output/f'corner_{round(extra*1000)}_{"buffer" if buffer else "tight"}';directory.mkdir()
            rectangles=[[.2,.2,2.2,1.2],[1.2-width/2,1.1,1.2+width/2,2.8+width/2],
                        [1.2-width/2,2.8-width/2,4.8,2.8+width/2],[4.6,1.8,5.8,3.8]]
            if buffer:rectangles.append([.7,2.3,1.7,3.3])
            # Tile the exact complement of free rectangles. Same wall boxes
            # feed physics and conservative map rasterization.
            xs=sorted({0.,6.}|{v for r in rectangles for v in (r[0],r[2])})
            ys=sorted({0.,4.}|{v for r in rectangles for v in (r[1],r[3])})
            walls=[]
            for x0,x1 in zip(xs[:-1],xs[1:]):
                for y0,y1 in zip(ys[:-1],ys[1:]):
                    x,y=(x0+x1)/2,(y0+y1)/2
                    if not any(a<=x<=c and b<=y<=d for a,b,c,d in rectangles):walls.append([x0,y0,x1,y1])
            model=ET.parse(template/'physical_scene.xml').getroot();world=model.find('worldbody')
            for child in list(world):
                if child.tag=='geom' and child.get('name')!='floor':world.remove(child)
            for mesh in model.findall('.//mesh'):mesh.set('file',os.path.relpath((template/mesh.get('file')).resolve(),directory))
            for i,(x0,y0,x1,y1) in enumerate(walls):
                ET.SubElement(world,'geom',name=f'wall_{i}',type='box',pos=f'{(x0+x1)/2} {(y0+y1)/2} .6',
                    size=f'{(x1-x0)/2} {(y1-y0)/2} .6',rgba='.6 .65 .7 1')
            ET.SubElement(model.find('asset'),'texture',name='junction_floor',type='2d',builtin='checker',width='256',height='256',rgb1='.25 .35 .5',rgb2='.55 .6 .65')
            ET.SubElement(model.find('asset'),'material',name='junction_mat',texture='junction_floor',texrepeat='12 8',texuniform='true')
            floor=world.find('geom[@name="floor"]');floor.set('material','junction_mat');floor.set('rgba','1 1 1 1')
            world.find('camera').set('pos','3 2 8');ET.indent(model)
            ET.ElementTree(model).write(directory/'physical_scene.xml',encoding='unicode')
            config=json.loads((template/'physical_scene.json').read_text());config.update(size_m=[6,4],resolution_m=.02,walls=walls,
                tasks=[dict(name='junction',start=[1.2,.7,np.pi/2],goal=[5.2,2.8])],map_id=directory.name)
            (directory/'physical_scene.json').write_text(json.dumps(config,indent=2))
            res=.02;x=(np.arange(300)+.5)*res;y=(200-np.arange(200)-.5)*res;blocked=np.zeros((200,300),bool)
            for a,b,c,d in walls:blocked|=((y[:,None]+res/2>b)&(y[:,None]-res/2<d)&(x[None,:]+res/2>a)&(x[None,:]-res/2<c))
            Image.fromarray(np.where(blocked,0,254).astype(np.uint8)).save(directory/'decision_map.pgm')
            (directory/'decision_map.yaml').write_text('image: decision_map.pgm\nresolution: 0.02\norigin: [0, 0, 0]\nnegate: 0\noccupied_thresh: 0.65\nfree_thresh: 0.196\n')
            manifest.append(dict(scene=str(directory),extra_width_m=extra,corner_buffer=buffer,
                start=[1.2,.7,np.pi/2],goal=[5.2,2.8],pivot=[1.2,2.8]))
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2));return manifest
