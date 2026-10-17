"""Display-only geometry. Never contributes to collision or laser observations."""
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
import mujoco
from .geometry import WORKSPACE, rpy_matrix


def quat(rpy):
    q=np.zeros(4); mujoco.mju_mat2Quat(q,rpy_matrix(rpy).reshape(-1)); return q


def attach_urdf_visuals(model, robot):
    source=WORKSPACE/'agv/simulations/agv_description'
    urdf=ET.parse(source/'urdf/agv_chassis.urdf').getroot()
    asset=ET.SubElement(model,'asset')
    materials={m.get('name'):m.find('color').get('rgba') for m in urdf.findall('material')}
    links={l.get('name'):l for l in urdf.findall('link')}
    edges={}
    for j in urdf.findall('joint'): edges.setdefault(j.find('parent').get('link'),[]).append(j)
    count=0
    def populate(name,body):
        nonlocal count
        for i,v in enumerate(links[name].findall('visual')):
            mesh=v.find('geometry/mesh')
            if mesh is None: continue
            filename=source/mesh.get('filename').split('package://agv_description/')[1]
            meshname=f'urdf_{name}_{i}'
            ET.SubElement(asset,'mesh',name=meshname,file=str(filename),scale=mesh.get('scale','1 1 1'))
            origin=v.find('origin'); pos='0 0 0' if origin is None else origin.get('xyz','0 0 0')
            rpy='0 0 0' if origin is None else origin.get('rpy','0 0 0')
            mat=v.find('material'); rgba=materials.get(mat.get('name'),'.4 .4 .4 1') if mat is not None else '.4 .4 .4 1'
            ET.SubElement(body,'geom',name=meshname,type='mesh',mesh=meshname,pos=pos,
                          quat=' '.join(map(str,quat(list(map(float,rpy.split()))))),rgba=rgba,
                          group='2',contype='0',conaffinity='0',density='0')
            count+=1
        for j in edges.get(name,[]):
            o=j.find('origin'); pos=o.get('xyz','0 0 0') if o is not None else '0 0 0'
            rpy=o.get('rpy','0 0 0') if o is not None else '0 0 0'
            child=j.find('child').get('link')
            b=ET.SubElement(body,'body',name='urdf_'+child,pos=pos,quat=' '.join(map(str,quat(list(map(float,rpy.split()))))))
            populate(child,b)
    populate('base_link',robot)
    return count


class NavigationOverlay:
    def __init__(self,model,lidar,path,goal):
        self.model=model; self.lidar=lidar; self.path=np.asarray(path); self.goal=np.asarray(goal)
        self.history=[]
        self.site=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_SITE,'lidar')

    def draw(self,scene,data,pose,scan):
        # Viewer user_scn is separate; renderer scene already contains world geometry.
        def geom(kind,size,pos,rgba):
            if scene.ngeom>=scene.maxgeom: return None
            g=scene.geoms[scene.ngeom]; scene.ngeom+=1
            mujoco.mjv_initGeom(g,kind,np.asarray(size,dtype=float),np.asarray(pos,dtype=float),np.eye(3).reshape(-1),np.asarray(rgba,dtype=np.float32))
            g.category=mujoco.mjtCatBit.mjCAT_DECOR; return g
        def line(a,b,width,color):
            g=geom(mujoco.mjtGeom.mjGEOM_LINE,[0,0,0],[0,0,0],color)
            if g is not None: mujoco.mjv_connector(g,mujoco.mjtGeom.mjGEOM_LINE,width,np.asarray(a,float),np.asarray(b,float))
        for a,b in zip(self.path[:-1],self.path[1:]): line([*a,.035],[*b,.035],5,[.1,.75,.95,1])
        xy=np.asarray(pose[:2])
        if not self.history or np.linalg.norm(xy-self.history[-1])>.025: self.history.append(xy.copy())
        tail=self.history[-1000:]
        for a,b in zip(tail[:-1],tail[1:]): line([*a,.045],[*b,.045],3,[1,.55,.08,1])
        geom(mujoco.mjtGeom.mjGEOM_CYLINDER,[.15,.025,0],[*self.goal,.03],[.1,.85,.25,1])
        origin=data.site_xpos[self.site].copy()
        angles=self.lidar.angles+pose[2]+self.lidar.mount[3]
        for i,(a,d) in enumerate(zip(angles,scan)):
            if not np.isfinite(d): continue
            end=origin+np.array([np.cos(a)*d,np.sin(a)*d,0])
            line(origin,end,1,[.12,.85,.3,.32])
            geom(mujoco.mjtGeom.mjGEOM_SPHERE,[.016,0,0],end,[1,.12,.15,1])
        # Sector boundaries show finite FOV; no fabricated rear beams.
        for a in angles[[0,-1]]:
            line(origin,origin+np.array([np.cos(a)*.9,np.sin(a)*.9,0]),2,[.95,.75,.1,1])
