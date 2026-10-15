"""Bounded pose-recovery evidence on frozen scans and stationary MuJoCo rays.

No ROS nodes, model training or motor output. Physical pose enters only the
sensor simulator and evaluation labels, never the recovery API. All maps and
source trajectories have already been used for development; not a held-out
generalization or full navigation acceptance test.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from navrl.geometry import ROOT,wrap
from navrl.map import StaticMap
from neural_navigation.experiment_budget import ExperimentBudget
from neural_navigation.fast_static_localizer import FastStaticLocalizer,temporal_contract
from neural_navigation.global_relocalizer import GlobalRelocalizer,StoppedPoseRecovery,contract
from neural_navigation.sensors import LidarSpec


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def error(pose,truth):
    return dict(xy_m=float(np.linalg.norm(np.asarray(pose)[:2]-truth[:2])),yaw_rad=float(abs(wrap(pose[2]-truth[2]))))


class StationaryLaser:
    def __init__(self,scene,lidar,strip_visuals=True):
        import mujoco
        self.mj=mujoco
        if strip_visuals:
            import xml.etree.ElementTree as ET
            root=ET.parse(scene/'physical_scene.xml').getroot()
            # This fixture only calls mj_forward/mj_multiRay with group 0.
            # Remove the excluded visual group and its unused mesh assets;
            # preserve walls, the physical body and the unmapped cylinder.
            # This is NOT a replacement for the full dynamics/visual viewer.
            for parent in root.iter():
                for geom in list(parent.findall('geom')):
                    if geom.get('group')=='2':parent.remove(geom)
            remaining={geom.get('mesh') for geom in root.iter('geom') if geom.get('mesh')}
            asset=root.find('asset')
            if asset is not None:
                for mesh in list(asset.findall('mesh')):
                    if mesh.get('name') not in remaining:asset.remove(mesh)
                    elif mesh.get('file'):mesh.set('file',str((scene/mesh.get('file')).resolve()))
            self.model=mujoco.MjModel.from_xml_string(ET.tostring(root,encoding='unicode'))
        else:self.model=mujoco.MjModel.from_xml_path(str(scene/'physical_scene.xml'))
        self.data=mujoco.MjData(self.model);self.lidar=lidar
        self.robot=mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_BODY,'robot')
        joints=[mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_JOINT,'robot_'+a) for a in ['x','y','yaw']]
        self.robot_q=[self.model.jnt_qposadr[j] for j in joints]
        meta=json.loads((scene/'physical_scene.json').read_text())
        self.obstacles=[]
        for obstacle in meta['dynamic_obstacles']:
            name=obstacle['name'];bid=mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_BODY,name)
            q=[self.model.jnt_qposadr[mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_JOINT,name+'_'+a)] for a in ['x','y']]
            self.obstacles.append((bid,q))

    def scan(self,pose,rng,mode,static_map):
        self.mj.mj_resetData(self.model,self.data)
        self.data.qpos[self.robot_q]=pose-np.r_[self.model.body_pos[self.robot,:2],0.]
        for index,(bid,q) in enumerate(self.obstacles):
            self.data.qpos[q]=[30.+index*2,30.]
        obstacle=None
        if mode=='unmapped' and self.obstacles:
            c,s=np.cos(pose[2]),np.sin(pose[2]);rotation=np.array([[c,s],[-s,c]])
            for body in [[1.,.15],[1.,-.15],[1.3,0.],[.8,.3]]:
                point=pose[:2]+np.array(body)@rotation
                if static_map.lookup(point,static_map.clearance,outside=0.)>.32:
                    bid,q=self.obstacles[0];self.data.qpos[q]=point-self.model.body_pos[bid,:2]
                    obstacle=point.tolist();break
            if obstacle is None:raise RuntimeError('Unmapped-obstacle case has no physical placement')
        self.mj.mj_forward(self.model,self.data)
        spec=self.lidar;c,s=np.cos(pose[2]),np.sin(pose[2])
        origin=np.r_[pose[:2]+spec.mount[:2]@np.array([[c,s],[-s,c]]),spec.mount[2]]
        angle=spec.angles+pose[2]+spec.mount[3]
        vector=np.c_[np.cos(angle),np.sin(angle),np.zeros(spec.beams)].reshape(-1)
        ids=np.zeros(spec.beams,np.int32);values=np.zeros(spec.beams)
        self.mj.mj_multiRay(self.model,self.data,origin,vector,np.array([1,0,0,0,0,0],np.uint8),1,self.robot,ids,values,spec.beams,10.)
        invalid=(values<spec.range_min)|(values>=10.)
        values+=rng.normal(0.,.01,spec.beams);values=np.clip(values,spec.range_min,10.)
        invalid|=rng.random(spec.beams)<.08
        if mode=='sector':invalid|=(spec.angles>.35)&(spec.angles<.65)
        if mode=='blank':invalid[:]=True
        values[invalid]=np.nan
        return values,obstacle


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--cpu-cores',type=float,default=.35)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    budget=ExperimentBudget(args.cpu_cores,memory_mib=2048,directory=args.output)
    cases=json.loads((args.cache/'TASKS.json').read_text());matchers={};simulators={}
    report=dict(completed=False,deployment_approved=False,
        scope='global-pose proposals and stopped confirmation; known development maps, NOT end-to-end navigation, real accuracy or unknown-map SLAM',
        sensor_only_simulator='MuJoCo forward/rays, excluded group-2 visuals and unused meshes removed; physical laser geometry retained, no physics stepping',
        contract=contract(),source_sha256={str(path.relative_to(ROOT)):digest(path) for path in
            [Path(__file__),ROOT/'neural_navigation/global_relocalizer.py']},snapshots=[],stationary=[])

    def matcher_for(case):
        scene=case['scene'];meta=json.loads((args.cache/(case['id']+'.json')).read_text())
        if scene not in matchers:
            m=StaticMap(ROOT/scene/'decision_map.yaml',radius=.3)
            spec=LidarSpec(meta['lidar'])
            matchers[scene]=GlobalRelocalizer(m,spec)
        return matchers[scene]

    def save():
        report['resource']=budget.statistics()
        (args.output/'RESULTS.json').write_text(json.dumps(report,indent=2)+'\n')

    # This is an exactly frozen input check. No re-simulated scan is substituted
    # for the historical snapshot, including new objects and sensor failures.
    for case in cases:
        data=np.load(args.cache/(case['id']+'.npz'));matcher=matcher_for(case)
        began=time.perf_counter();job=matcher.start(data['scan'][0])
        while not job.advance():budget.tick()
        row=dict(id=case['id'],mode=case['mode'],cache_sha256=digest(args.cache/(case['id']+'.npz')),
            result=job.result,wall_seconds=time.perf_counter()-began)
        if job.result['candidate_valid']:row['error']=error(job.result['pose'],data['truth'][0])
        report['snapshots'].append(row);budget.tick();save()
        print(json.dumps(dict(stage='snapshot',id=case['id'],valid=job.result['candidate_valid'],reason=job.result['reason'])),flush=True)

    # Independent noisy ray measurements at a stationary physical pose permit
    # testing confirmation. The old tracking baseline starts at a stale map
    # pose; both approaches receive the same arbitrary odom origin and scans.
    selected=[case for case in cases if case['id'] in ['short_06','short_07','short_10','short_11']]
    selected += [case for case in cases if case['id'] in ['short_00','short_12']]
    for case in selected:
        matcher=matcher_for(case);scene=case['scene']
        if scene not in simulators:simulators[scene]=StationaryLaser(ROOT/scene,matcher.lidar)
        simulator=simulators[scene];data=np.load(args.cache/(case['id']+'.npz'));truth=data['truth'][0].copy()
        for mode in ['normal','sector','unmapped','blank']:
            seed=218000+len(report['stationary']);rng=np.random.default_rng(seed)
            scans=[];obstacles=[]
            for _ in range(8):
                scan,obstacle=simulator.scan(truth,rng,mode,matcher.map);scans.append(scan);obstacles.append(obstacle)
            scans=np.array(scans);odom=np.array([15.,-6.,1.2]);velocity=np.zeros(2)
            baseline=FastStaticLocalizer(matcher.map,matcher.lidar,temporal_contract())
            baseline.reset([1.2,1.2,0.],None)
            for i,scan in enumerate(scans):baseline.update(odom,scan,i*.2)
            recovery=StoppedPoseRecovery(matcher);recovery.begin(scans[0],odom,velocity,0.)
            began=time.perf_counter()
            while recovery.state=='searching':recovery.advance();budget.tick()
            search_wall=time.perf_counter()-began
            for i,scan in enumerate(scans[1:]):
                recovery.observe(scan,odom,velocity,search_wall+(i+1)*.2)
                if recovery.state=='ready':break
                budget.tick()
            diag=recovery.diagnostics();row=dict(id=case['id']+'_'+mode,mode=mode,seed=seed,
                physical_xml_sha256=digest(ROOT/scene/'physical_scene.xml'),unmapped_obstacle_xy=obstacles[0],
                stationary_truth=truth.tolist(),odometry_origin=odom.tolist(),old_seed=[1.2,1.2,0.],
                baseline_error=error(baseline.pose,truth),recovery=diag,search_wall_seconds=search_wall)
            if recovery.state=='ready':
                row['error']=error(recovery.pose,truth)
                row['incorrect_accept']=row['error']['xy_m']>.15 or row['error']['yaw_rad']>.10
            else:row['incorrect_accept']=False
            name=row['id'];np.savez_compressed(args.output/(name+'.npz'),scan=scans,truth=truth,odom=odom)
            report['stationary'].append(row);save()
            print(json.dumps(dict(stage='stationary',id=name,state=recovery.state,reason=recovery.reason,error=row.get('error'))),flush=True)
    valid=[row for row in report['snapshots'] if 'error' in row]
    ready=[row for row in report['stationary'] if 'error' in row]
    report['summary']=dict(snapshot_total=len(report['snapshots']),snapshot_proposed=len(valid),
        stationary_total=len(report['stationary']),stationary_confirmed=len(ready),
        incorrect_snapshot_proposals=sum(row['error']['xy_m']>.15 or row['error']['yaw_rad']>.10 for row in valid),
        incorrect_stationary_accepts=sum(row['incorrect_accept'] for row in report['stationary']),
        confirmed_xy_max_m=max([row['error']['xy_m'] for row in ready],default=None),
        confirmed_yaw_max_rad=max([row['error']['yaw_rad'] for row in ready],default=None))
    report['completed']=True;save();print(json.dumps(report['summary']),flush=True)


if __name__=='__main__':main()
