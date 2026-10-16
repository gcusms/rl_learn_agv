"""Park the imported CAD arm using the repository's home XYZ/pitch, offline only."""
import ast
import json
import math
import xml.etree.ElementTree as ET
import numpy as np
import yaml
import mujoco
from scipy.optimize import root
from scipy.spatial.transform import Rotation
from .geometry import WORKSPACE, rpy_matrix

SOURCE = WORKSPACE/'agv/simulations/agv_description'
HOME_FILE = WORKSPACE/'agv/driver/ros_arm_controller/config/init_pose.yaml'


def home_configuration():
    cfg=yaml.safe_load((SOURCE/'config/arm_sync.yaml').read_text())
    home=yaml.safe_load(HOME_FILE.read_text())['home_pose']
    c=-math.radians(home['pitch']); length=cfg['tool_length_m']
    def fk(q):
        a,b=q[0],sum(q)
        return np.array([.0348*np.cos(a)+.22535*np.sin(a)+.027316*np.cos(b)-.14239*np.sin(b)+length*np.cos(c),
                         .110449-.0348*np.sin(a)+.22535*np.cos(a)-.027316*np.sin(b)-.14239*np.cos(b)-length*np.sin(c)])
    target=np.array([math.hypot(home['x'],home['y']),home['z']])/1000
    result=root(lambda q:fk(q)-target,cfg['preview_radians'][1:3])
    if not result.success or np.linalg.norm(fk(result.x)-target)>1e-6:raise ValueError('CAD home IK failed')
    a,b=result.x
    positions=[-math.atan2(home['y'],home['x']),float(a),float(b),float(c-a-b),math.radians(home['roll']),0.]
    # Cross-check with the actual read-only ROS bridge's independent validator.
    source=ast.parse((SOURCE/'agv_description/arm_state_bridge.py').read_text())
    validator=next(n for n in source.body if isinstance(n,ast.FunctionDef) and n.name=='validate_home_solution')
    namespace={'math':math};exec(compile(ast.Module(body=[validator],type_ignores=[]),'<source CAD validator>','exec'),namespace)
    namespace['validate_home_solution'](positions,home,cfg)
    names=['joint1','joint2','joint3','joint4','joint5','gripper_base_joint']
    joints=dict(zip(names,positions))
    urdf=ET.parse(SOURCE/'urdf/agv_chassis.urdf').getroot()
    for j in urdf.findall('joint'):
        if j.get('name') in joints:
            limit=j.find('limit');q=joints[j.get('name')]
            if limit is not None and not float(limit.get('lower',-math.inf))<=q<=float(limit.get('upper',math.inf)):
                raise ValueError('Home outside URDF limit: '+j.get('name'))
    return dict(source=str(HOME_FILE),mapping_source=str(SOURCE/'config/arm_sync.yaml'),home_pose=home,
                joint_positions_rad=joints,method='Offline CAD IK, checked against source arm_state_bridge.validate_home_solution; no hardware commands',
                cad_fk_error_m=float(np.linalg.norm(fk(result.x)-target)),gripper='Open linkage q=0 for claw=-95 degrees',
                launch_note='controller.launch.py -> odom_publisher.launch.py -> ros_robot_controller.launch.py -> arm_controller; auto_home_on_startup defaults false in current source')


def apply_home_pose(model_xml):
    """Apply origin rotation followed by joint-axis rotation to fixed imported bodies."""
    cfg=home_configuration(); positions=cfg['joint_positions_rad']
    for j in ET.parse(SOURCE/'urdf/agv_chassis.urdf').getroot().findall('joint'):
        mimic=j.find('mimic');name=j.get('name')
        if mimic is not None:
            q=positions.get(mimic.get('joint'),0)*float(mimic.get('multiplier',1))+float(mimic.get('offset',0))
        elif name in positions:q=positions[name]
        else:continue
        body=model_xml.find(".//body[@name='urdf_%s']"%j.find('child').get('link'))
        if body is None:raise ValueError('Missing imported arm body '+name)
        o=j.find('origin');rpy=np.fromstring(o.get('rpy','0 0 0'),sep=' ')
        axis=np.fromstring(j.find('axis').get('xyz'),sep=' ');axis/=np.linalg.norm(axis)
        if j.get('type')=='prismatic':
            xyz=np.fromstring(o.get('xyz','0 0 0'),sep=' ')+rpy_matrix(rpy)@(q*axis)
            body.set('pos',' '.join(map(str,xyz)));continue
        rotation=rpy_matrix(rpy)@Rotation.from_rotvec(q*axis).as_matrix();quat=np.zeros(4)
        mujoco.mju_mat2Quat(quat,rotation.reshape(-1));body.set('quat',' '.join(map(str,quat)))
    return cfg
