"""Build MuJoCo dynamic planar bodies and solid human collision envelopes."""
import json
import xml.etree.ElementTree as ET
from navrl.geometry import ROOT


def main():
    model=ET.parse(ROOT/'assets/decision_scene.xml').getroot()
    model.set('model','agv_solid_pedestrian_navigation')
    option=model.find('option'); option.set('timestep','.005'); option.set('iterations','80'); option.set('integrator','implicitfast')
    default=ET.SubElement(model,'default')
    ET.SubElement(default,'geom',solref='.008 1',solimp='.99 .999 .001',margin='.015',friction='.7 .02 .002')
    world=model.find('worldbody');world.find('geom[@name="floor"]').set('contype','0');world.find('geom[@name="floor"]').set('conaffinity','0')
    for name in ['robot','crossing_obstacle','corridor_obstacle']:
        body=world.find(f'body[@name="{name}"]');body.attrib.pop('mocap',None)
        for axis,vec in [('x','1 0 0'),('y','0 1 0')]:
            ET.SubElement(body,'joint',name=f'{name}_{axis}',type='slide',axis=vec,damping='0')
        if name=='robot':
            ET.SubElement(body,'joint',name='robot_yaw',type='hinge',axis='0 0 1',damping='0')
            body.find('geom[@name="robot_footprint"]').set('mass','8')
            # Meshes use their MuJoCo convex contact hulls as well as the chassis box.
            for g in body.findall('.//geom'):
                if g.get('type')=='mesh':g.set('contype','1');g.set('conaffinity','1')
        else:
            body.set('pos','0 0 .85')
            g=body.find('geom');g.set('size','.25 .85');g.set('mass','70');g.set('rgba','0 0 0 .001')
            color='.85 .28 .12 1' if name=='crossing_obstacle' else '.2 .45 .85 1'
            def visual(kind,pos,size,rgba,**extra):
                ET.SubElement(body,'geom',type=kind,pos=pos,size=size,rgba=rgba,group='2',contype='0',conaffinity='0',density='0',**extra)
            visual('ellipsoid','0 0 .35','.17 .22 .28',color)
            visual('sphere','0 0 .71','.14','.86 .65 .45 1')
            for side in [-1,1]:
                visual('capsule',f'0 {side*.105} -.39','.065 .34','.15 .18 .22 1')
                visual('capsule',f'0 {side*.20} .28','.04 .22',color)
                visual('ellipsoid',f'.04 {side*.105} -.79','.13 .07 .05','.08 .09 .1 1')
    ET.indent(model);ET.ElementTree(model).write(ROOT/'assets/physical_scene.xml',encoding='unicode')
    config=json.loads((ROOT/'assets/scene.json').read_text())
    config['dynamic_obstacles']=[dict(name=name,radius_m=.25,mass_kg=70,spawn_m=spawn,nominal_speed_range_m_s=[.5,.85],acceleration_limit_per_axis_m_s2=1.8,random_reachable_goals=True) for name,spawn in [('crossing_obstacle',[2.5,3.5]),('corridor_obstacle',[4.8,2.])]]
    config['robot_model']='Native planar MuJoCo dynamic body; URDF convex contact geometry; force servo; synthetic chassis mass 8 kg'
    config.update(scene_file='physical_scene.xml',simulation_backend='MuJoCo mj_step, constrained planar rigid bodies',
                  pedestrian_behavior='random reachable waypoints; acceleration-limited yielding and side stepping; native solid contacts')
    (ROOT/'assets/physical_scene.json').write_text(json.dumps(config,indent=2))
    print('Built native MuJoCo solid robot and pedestrian scene.')


if __name__=='__main__':main()
