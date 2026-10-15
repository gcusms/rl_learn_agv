"""Two manual-only physical columns with live robot navigation."""
import copy,json,shutil,xml.etree.ElementTree as ET
import numpy as np
from navrl.geometry import ROOT
from neural_navigation.room_env import RoomNavigationEnv

SCENE=ROOT/'assets/room_navigation_manual_two'

def build_scene():
    src=ROOT/'assets/room_navigation_v3_validation';SCENE.mkdir(exist_ok=True)
    for name in ['decision_map.yaml','decision_map.pgm','arm_home.json']:
        shutil.copy2(src/name,SCENE/name)
    config=json.loads((src/'physical_scene.json').read_text());task=copy.deepcopy(config['tasks'][2])
    tree=ET.parse(src/'physical_scene.xml');world=tree.getroot().find('worldbody')
    template=copy.deepcopy(world.find("body[@name='crossing_obstacle']"))
    for item in config['dynamic_obstacles']:world.remove(world.find("body[@name='%s']"%item['name']))
    for geom in list(world.findall('geom')):
        if geom.get('name','').startswith('room_floor_'):world.remove(geom)
    asset=tree.getroot().find('asset')
    ET.SubElement(asset,'texture',name='manual_ground_texture',type='2d',builtin='checker',rgb1='.2 .3 .4',rgb2='.1 .2 .3',width='512',height='512')
    ET.SubElement(asset,'material',name='manual_ground',texture='manual_ground_texture',texrepeat='1 1',texuniform='true',reflectance='.05')
    floor=world.find("geom[@name='floor']");floor.attrib.pop('rgba',None);floor.set('material','manual_ground')
    # A plane is infinite in MuJoCo; use a finite slab bounded by the outer walls.
    width,height=config['size_m'];floor.set('type','box');floor.set('pos',f'{width/2} {height/2} -.02');floor.set('size',f'{width/2} {height/2} .02')
    config['dynamic_obstacles']=[];blocks=[]
    for i,park in enumerate([[3.6,6.8],[6.4,1.1]]):
        name=f'manual_column_{i+1}';body=copy.deepcopy(template);body.set('name',name)
        for child in body:
            child.set('name',child.get('name').replace('crossing_obstacle',name))
            if child.tag=='geom':child.set('rgba',['.95 .42 .1 1','.16 .48 .86 1'][i])
        world.append(body)
        config['dynamic_obstacles'].append(dict(name=name,radius_m=.25,mass_kg=70,spawn_m=park,kind='moving_solid_column',height_m=1.7,random_reachable_goals=False))
        blocks.append(dict(column_index=i,position=park))
    task.update(name='manual_two_columns',column_task=dict(kind='manual_only',blocks=blocks));config['tasks']=[task]
    config.update(floor_style='Blue-gray checker, finite slab inside outer wall bounds',floor_bounds_m=[0,0,*config['size_m']],column_behavior='Two manually controlled solid columns; parked above/below, away from doors. No automatic interference.',pedestrian_behavior='Manual movement targets, bounded velocity and solid contacts',validation_note='Interactive manual interference; not fixed-protocol success-rate evaluation')
    ET.indent(tree);tree.write(SCENE/'physical_scene.xml',encoding='unicode');(SCENE/'physical_scene.json').write_text(json.dumps(config,indent=2))
    return SCENE

class WaypointInterferenceEnv(RoomNavigationEnv):
    def _new_route(self,index,away=False):
        # Suppress random walking even during the base reset; scripts own world actors.
        pos=self._humans()[index];self.walkers[index]['path']=np.array([pos,pos]);self.walkers[index]['goal']=pos

    def reset(self,**kwargs):
        self.active_leg=0;self.manual_mode=True;self.manual_targets={};self.manual_events=[]
        obs,info=super().reset(**kwargs)
        for i,item in enumerate(self.scene['dynamic_obstacles']):
            q,d=self.addresses[item['name']];self.data.qpos[q]=item['spawn_m'];self.data.qvel[d]=0.
            self.walkers[i]['block']=dict(column_index=i,phase='manual',trigger_time_s=None)
        import mujoco
        mujoco.mj_forward(self.model,self.data)
        self.manual_targets={i:p.copy() for i,p in enumerate(self._humans())}
        self.scan=self._scan();self.estimate=self.localizer.update(self.odom,self.scan)
        obs=self._observation(self.scan,initialize=True);self.previous_remaining=self.remaining;self.minimum_gap=self._minimum_gap()
        return obs,self._info('running')

    def set_manual_mode(self,enabled):
        # This interactive scenario has no automatic obstacle controller.
        self.manual_mode=True

    def drag_column(self,index,target):
        point=np.asarray(target,float)
        if index not in range(len(self.walkers)) or point.shape!=(2,) or not np.isfinite(point).all():raise ValueError('Invalid column drag')
        if not self.manual_mode:self.set_manual_mode(True)
        point=np.clip(point,[.28,.28],np.array(self.scene['size_m'])-.28)
        self.manual_targets[index]=point.copy()
        self.manual_events.append(dict(time_s=float(self.elapsed),active_target=self.active_leg+1,column_index=index,target=point.tolist()))

    def _manual_velocity(self,index):
        pos=self._humans()[index];target=self.manual_targets.get(index,pos)
        delta=target-pos;norm=np.linalg.norm(delta)
        self.walkers[index]['state']='manual_hold' if norm<.04 else 'manual_drag'
        if norm<.04:return np.zeros(2)
        desired=delta/norm*min(.65,2*norm);future=pos+desired*.25
        if self.wall_gap(future)<.02 or not self.human_map.visible(pos,future) or any(j!=index and np.linalg.norm(future-p)<.60 for j,p in enumerate(self._humans())):
            self.walkers[index]['state']='manual_blocked';return np.zeros(2)
        if np.dot(desired,self.truth[:2]-pos)>0 and (self.robot_gap(pos)<.3 or self.robot_gap(future)<.14):
            self.walkers[index]['state']='manual_yielding';return np.zeros(2)
        return desired

    def _walker_velocity(self,index,h):
        if not self.manual_targets:return np.zeros(2)
        return self._manual_velocity(index)

    def select_goal(self,index):
        self.active_leg=index
        return self.set_navigation_goal(self.scene['tasks'][0]['waypoints'][index])
