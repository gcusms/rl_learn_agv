import math
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent


def wrap(angle):
    return (angle + np.pi) % (2*np.pi) - np.pi


def integrate(pose, v, w, dt):
    x,y,yaw = pose
    if abs(w) < 1e-9:
        return np.array([x + v*dt*np.cos(yaw), y + v*dt*np.sin(yaw), wrap(yaw+w*dt)])
    new_yaw = yaw+w*dt
    return np.array([x+v/w*(np.sin(new_yaw)-np.sin(yaw)),
                     y-v/w*(np.cos(new_yaw)-np.cos(yaw)), wrap(new_yaw)])


def rpy_matrix(rpy):
    r,p,y = rpy
    cr,sr,cp,sp,cy,sy = np.cos(r),np.sin(r),np.cos(p),np.sin(p),np.cos(y),np.sin(y)
    return np.array([[cy*cp,cy*sp*sr-sy*cr,cy*sp*cr+sy*sr],
                     [sy*cp,sy*sp*sr+cy*cr,sy*sp*cr-cy*sr],[-sp,cp*sr,cp*cr]])


def lidar_mount():
    robot = ET.parse(WORKSPACE/'agv/simulations/agv_description/urdf/agv_chassis.urdf').getroot()
    edges = {j.find('child').get('link'): j for j in robot.findall('joint')}
    chain=[]; child='lidar_frame'
    while child != 'base_footprint':
        joint=edges[child]
        if joint.get('type') != 'fixed':
            raise ValueError('LiDAR mount must have a fixed URDF chain')
        origin=joint.find('origin')
        tf=np.eye(4)
        if origin is not None:
            tf[:3,3]=[float(v) for v in origin.get('xyz','0 0 0').split()]
            tf[:3,:3]=rpy_matrix([float(v) for v in origin.get('rpy','0 0 0').split()])
        chain.append(tf); child=joint.find('parent').get('link')
    result=np.eye(4)
    for tf in reversed(chain): result=result@tf
    if not np.allclose(result[2,:3], [0,0,1], atol=1e-6):
        raise ValueError('This planar scanner requires a horizontal LiDAR mount')
    return np.array([*result[:3,3], math.atan2(result[1,0], result[0,0])])


class LidarProfile:
    def __init__(self, path=None):
        self.path=Path(path) if path else ROOT/'config/lidar.yaml'
        self.config=yaml.safe_load(self.path.read_text())
        source=WORKSPACE/self.config['filter_source']
        filt=yaml.safe_load(source.read_text())['/**']['ros__parameters']['filter1']
        if filt['type'] != 'laser_filters/LaserScanSectorFilter' or not filt['params']['clear_inside']:
            raise ValueError('Expected SCLIDAR rear-sector clearing configuration')
        params=filt['params']
        # Retained sector crosses zero: [rear_end, 2pi) union [0, rear_start].
        self.angle_min=float(params['angle_max'])-2*np.pi
        self.angle_max=float(params['angle_min'])
        self.beams=int(self.config['beams'])
        self.angles=np.linspace(self.angle_min,self.angle_max,self.beams)
        self.range_min=float(self.config['range_min_m'])
        self.range_max=float(self.config['range_max_m'])
        self.mount=lidar_mount()
        self.noise=float(self.config['noise_std_m'])
        self.dropout=float(self.config['dropout_probability'])

    @property
    def signature(self):
        return dict(angle_min=self.angle_min, angle_max=self.angle_max, beams=self.beams,
                    range_min=self.range_min, range_max=self.range_max, mount=self.mount.tolist(),
                    source=self.config['filter_source'])
