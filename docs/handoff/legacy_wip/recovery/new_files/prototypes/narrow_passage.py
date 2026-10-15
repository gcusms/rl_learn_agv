"""Deterministic MuJoCo narrow-lane fixtures and continuous static body checks.

No commands or graph planner here. Geometry uses the configured real footprint;
unknown map cells remain occupied. Fixtures deliberately retain wide entrances.
"""
import json
import math
import os
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
from PIL import Image
from neural_navigation.static_footprint import static_footprint_clear


def swept_segment_clear(m, a, b, footprint, margin, max_spacing=.01):
    a,b=np.asarray(a,float),np.asarray(b,float)
    distance=float(np.linalg.norm(b-a))
    if not np.isfinite(np.r_[a,b]).all() or distance<1e-8:return False
    yaw=math.atan2(b[1]-a[1],b[0]-a[0])
    if not np.isfinite(max_spacing) or max_spacing<=0:raise ValueError('Positive finite sampling spacing required')
    count=max(1,math.ceil(distance/max_spacing))
    # Extra isotropic allowance covers translation between samples.
    return all(static_footprint_clear(m,np.r_[p,yaw],footprint,margin+distance/count/2)
               for p in np.linspace(a,b,count+1))


def rotation_clear(m, point, before, after, footprint, margin):
    angle=math.atan2(math.sin(after-before),math.cos(after-before))
    count=max(1,math.ceil(abs(angle)/math.radians(2)))
    # Maximum displacement of any expanded-body corner between sampled angles.
    corner=np.linalg.norm(np.asarray(footprint)/2+margin)
    allowance=corner*abs(angle)/count/2
    return all(static_footprint_clear(m,np.r_[point,before+angle*i/count],footprint,margin+allowance)
               for i in range(count+1))


def generate(output, hardware):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    template=Path('training/assets/room_navigation_scripted')
    length,width=hardware['robot_footprint_m'];manifest=[]
    for extra in (.15,.125,.10):
        directory=output/f'gap_{round(extra*1000):03d}';directory.mkdir()
        span=width+extra;bottom=2-span/2;top=2+span/2
        walls=[[0,0,6,.2],[0,3.8,6,4],[0,.2,.2,3.8],[5.8,.2,6,3.8],
               [2,.2,4,bottom],[2,top,4,3.8]]
        model=ET.parse(template/'physical_scene.xml').getroot();world=model.find('worldbody')
        for child in list(world):
            if child.tag=='geom' and child.get('name')!='floor':world.remove(child)
        for mesh in model.findall('.//mesh'):
            source=(template/mesh.get('file')).resolve()
            mesh.set('file',os.path.relpath(source,directory))
        for i,(x0,y0,x1,y1) in enumerate(walls):
            ET.SubElement(world,'geom',name=f'wall_{i}',type='box',pos=f'{(x0+x1)/2} {(y0+y1)/2} .6',
                          size=f'{(x1-x0)/2} {(y1-y0)/2} .6',rgba='.6 .65 .7 1')
        ET.SubElement(model.find('asset'),'texture',name='narrow_floor_tex',type='2d',builtin='checker',width='256',height='256',rgb1='.25 .35 .5',rgb2='.55 .6 .65')
        ET.SubElement(model.find('asset'),'material',name='narrow_floor_mat',texture='narrow_floor_tex',texrepeat='12 8',texuniform='true')
        floor=world.find('geom[@name="floor"]');floor.set('material','narrow_floor_mat');floor.set('rgba','1 1 1 1')
        model.find('worldbody/camera').set('pos','3 2 8')
        ET.indent(model);ET.ElementTree(model).write(directory/'physical_scene.xml',encoding='unicode')
        config=json.loads((template/'physical_scene.json').read_text());config.update(size_m=[6,4],resolution_m=.02,walls=walls,
            tasks=[dict(name='straight_narrow',start=[1,2,0],goal=[5,2])],map_id=directory.name)
        for item in config['dynamic_obstacles']:item['spawn_m']=[20,20]
        (directory/'physical_scene.json').write_text(json.dumps(config,indent=2))
        res=.02;xs=(np.arange(300)+.5)*res;ys=(200-np.arange(200)-.5)*res
        blocked=np.zeros((200,300),bool)
        for x0,y0,x1,y1 in walls:
            # Any cell intersecting a solid wall is occupied, never optimistic rounding.
            blocked |= ((ys[:,None]+res/2>y0)&(ys[:,None]-res/2<y1)&
                        (xs[None,:]+res/2>x0)&(xs[None,:]-res/2<x1))
        Image.fromarray(np.where(blocked,0,254).astype(np.uint8)).save(directory/'decision_map.pgm')
        (directory/'decision_map.yaml').write_text('image: decision_map.pgm\nresolution: 0.02\norigin: [0, 0, 0]\nnegate: 0\noccupied_thresh: 0.65\nfree_thresh: 0.196\n')
        manifest.append(dict(scene=str(directory),extra_width_m=extra,physical_corridor_width_m=span,
                             passage_bounds=[2,bottom,4,top],start=[1,2,0],goal=[5,2]))
    (output/'manifest.json').write_text(json.dumps(dict(hardware=hardware,scenes=manifest,training_data=False),indent=2))
    return manifest
