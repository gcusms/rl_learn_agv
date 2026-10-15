"""Paired localizer replay and causal measured-motion ablation.

Truth is evaluator-only in the frozen simulation cache. The robot recording
has no independent pose truth and cannot produce navigation success scores.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import time
import numpy as np

from navrl.geometry import ROOT,wrap
from navrl.map import StaticMap
from neural_navigation.experiment_budget import ExperimentBudget
from neural_navigation.fast_static_localizer import FastStaticLocalizer,temporal_contract
from neural_navigation.verified_static_localizer import VerifiedStaticLocalizer,contract as verified_contract
from neural_navigation.measured_odometry import MeasuredOdometry,contract as motion_contract
from neural_navigation.sensors import LidarSpec
from experiments.benchmark_static_localization import metrics


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run_localizers(m,lidar,data,budget):
    output={}
    for name,build in [('temporal',lambda:FastStaticLocalizer(m,lidar,temporal_contract())),
                       ('ray_consistent',lambda:VerifiedStaticLocalizer(m,lidar))]:
        loc=build();loc.reset(data['prior'],None);loc._field(np.array([data['prior'][:2]]))
        poses=[];stds=[];cpus=[];walls=[];reasons=Counter();excluded=[];ranks=[];accepted=[]
        for odom,scan,t in zip(data['odom'],data['scan'],data['t']):
            start=time.perf_counter();cpu=time.process_time()
            pose=loc.update(odom,scan,t)
            cpus.append(time.process_time()-cpu);walls.append(time.perf_counter()-start)
            poses.append(pose);stds.append(loc.std.copy());reasons[loc.correction_reason]+=1
            excluded.append(getattr(loc,'ray_excluded_static_hits',0))
            ranks.append(loc.observable_rank);accepted.append(loc.scan_update_accepted)
            budget.tick()
        output[name]=dict(poses=np.array(poses),stds=np.array(stds),cpu=np.array(cpus),wall=np.array(walls),
                          reasons=dict(reasons),excluded=np.array(excluded),rank=np.array(ranks),accepted=np.array(accepted))
    return output


def synthetic(args,budget,result):
    cache=args.cache;rows=[];validation={k:[] for k in ['temporal','ray_consistent']}
    for case in json.loads((cache/'TASKS.json').read_text()):
        z=np.load(cache/(case['id']+'.npz'));meta=json.loads((cache/(case['id']+'.json')).read_text())
        lidar=LidarSpec(meta['lidar']);lidar.range_max=10.
        m=StaticMap(ROOT/case['scene']/'decision_map.yaml',radius=.3)
        scan=z['scan'].copy();scan[scan>=10.]=np.inf
        data=dict(prior=z['prior'],scan=scan,odom=z['odom'],t=z['t'])
        outputs=run_localizers(m,lidar,data,budget)
        row=dict(id=case['id'],split=case['split'],mode=case['mode'],cache_sha256=digest(cache/(case['id']+'.npz')),algorithms={})
        arrays={}
        for name,o in outputs.items():
            row['algorithms'][name]=dict(metrics(z['truth'],o['poses'],o['stds'],o['wall'],o['cpu']),
                reasons=o['reasons'],excluded_static_hits_total=int(o['excluded'].sum()))
            for field in ('poses','stds','cpu','excluded','rank','accepted'):arrays[name+'_'+field]=o[field]
            if case['split']=='validation':
                error=np.linalg.norm(o['poses'][:,:2]-z['truth'][:,:2],axis=1)
                yaw=abs(wrap(o['poses'][:,2]-z['truth'][:,2]))
                validation[name].append(dict(error=error,yaw=yaw,cpu=o['cpu'],final=float(error[-1])))
        np.savez_compressed(args.output/(case['id']+'.npz'),**arrays)
        rows.append(row);result['simulation_cases']=rows
        (args.output/'RESULTS.json').write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(dict(case=case['id'],p95={k:round(v['xy_p95_m'],4) for k,v in row['algorithms'].items()})),flush=True)
    result['simulation_validation']={}
    for name,batches in validation.items():
        errors=np.concatenate([b['error'] for b in batches]);yaw=np.concatenate([b['yaw'] for b in batches])
        cpu=np.concatenate([b['cpu'] for b in batches])
        result['simulation_validation'][name]=dict(trajectories=len(batches),frames=len(errors),
            xy_p95_m=float(np.percentile(errors,95)),yaw_p95_rad=float(np.percentile(yaw,95)),
            mean_final_xy_m=float(np.mean([b['final'] for b in batches])),cpu_p95_ms=float(np.percentile(cpu,95)*1000),
            cpu_max_ms=float(cpu.max()*1000))


def recorded_motion(args,budget):
    sensors=np.load(args.analysis/'sensors.npz');wheel=sensors['/wheel/odom'];imu=sensors['/imu']
    feedback=json.loads((args.analysis/'feedback.json').read_text())
    feedback_times=np.array([row['t'] for row in feedback]);odom=MeasuredOdometry()
    states=[];valid=[];reasons=Counter();gyros=[];bias=[];timings=[]
    for row in wheel:
        timestamp=row[1];ii=np.searchsorted(imu[:,1],timestamp,side='right')-1
        fi=np.searchsorted(feedback_times,row[0],side='right')-1
        if ii<0 or fi<0:
            # Startup has not acquired both streams; no missing interval is
            # bridged because the frontend has not anchored any pose yet.
            continue
        f=feedback[fi]['data']
        fresh=bool(f['fresh'] and f['measurement_valid'] and row[0]-feedback_times[fi]<=.15)
        start=time.process_time()
        state=odom.update(timestamp,row[5:8],timestamp,imu[ii,2],imu[ii,1],wheel_valid=fresh)
        timings.append(time.process_time()-start)
        states.append([timestamp,*state['pose'],*state['velocity']]);valid.append(state['valid'])
        gyros.append(imu[ii,2]);bias.append(state['gyro_bias_rad_s']);reasons[state['reason']]+=1
        budget.tick()
    states=np.array(states);valid=np.array(valid,bool)
    np.savez_compressed(args.output/'measured_motion.npz',states=states,valid=valid,gyro_z=gyros,bias=bias,cpu=timings)
    report=dict(frames=len(states),valid_frames=int(valid.sum()),reasons=dict(reasons),
        gyro_bias_final_rad_s=float(bias[-1]),bias_updates=odom.bias_updates,
        cpu_p95_ms=float(np.percentile(timings,95)*1000),cpu_max_ms=float(max(timings)*1000),
        uses_future_measurements=False, wheel_or_command_pose_used=False,
        accuracy_measured=False, reference_pose_available=False)
    return states,valid,report


def recorded(args,budget,result):
    motion,valid,motion_report=recorded_motion(args,budget);result['recorded_motion']=motion_report
    rows=[json.loads(line)['data'] for line in (args.record/'decisions.jsonl').open()]
    start=json.loads((args.record/'session.json').read_text())['started_unix']
    run,=(args.record/'snapshots/runs').iterdir()
    m=StaticMap(run/'prepared_map/map.yaml',radius=.3)
    lidar=LidarSpec(json.loads(args.hardware.read_text())['lidar'])
    missions=sorted({row['mission_id'] for row in rows});record_rows=[]
    for mission in missions:
        group=[row for row in rows if row['mission_id']==mission]
        for source in ['recorded_ekf','measured_wheel_gyro']:
            original=np.array([row['inputs']['odom'] for row in group],float)
            odometry=original.copy();skip=None
            if source=='measured_wheel_gyro':
                t=np.array([row['scan_stamp_ns']/1e9-start for row in group])
                index=np.searchsorted(motion[:,0],t,side='right')-1
                index=np.maximum(index,0);age=t-motion[index,0]
                if not np.all(valid[index]&(age>=-1e-6)&(age<=motion_contract()['output_max_age_s'])):
                    skip='motion_not_causally_available_for_all_frames'
                else:odometry=motion[index,1:4]
            row=dict(mission=mission,source=source,frames=len(group),skip=skip,algorithms={})
            if skip is None:
                data=dict(prior=np.array(group[0]['initial'],float),odom=odometry,
                    scan=np.array([r['inputs']['scan'] for r in group],float),
                    t=np.array([r['result']['timestamp'] for r in group]))
                outputs=run_localizers(m,lidar,data,budget);arrays={}
                for name,o in outputs.items():
                    row['algorithms'][name]=dict(estimated_displacement_m=float(np.linalg.norm(o['poses'][-1,:2]-o['poses'][0,:2])),
                        estimated_travel_m=float(np.linalg.norm(np.diff(o['poses'][:,:2],axis=0),axis=1).sum()),
                        accepted_scan_frames=int(o['accepted'].sum()),reasons=o['reasons'],
                        excluded_static_hits_total=int(o['excluded'].sum()),
                        cpu_p95_ms=float(np.percentile(o['cpu'],95)*1000))
                    for field in ('poses','stds','cpu','excluded','rank','accepted'):arrays[name+'_'+field]=o[field]
                row['odometry_displacement_m']=float(np.linalg.norm(odometry[-1,:2]-odometry[0,:2]))
                np.savez_compressed(args.output/(f'real_{mission:02d}_{source}.npz'),**arrays)
            record_rows.append(row)
        print(json.dumps(dict(mission=mission,completed=True)),flush=True)
        result['recorded_localization']=record_rows
        (args.output/'RESULTS.json').write_text(json.dumps(result,indent=2)+'\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache',type=Path,required=True);parser.add_argument('--analysis',type=Path,required=True)
    parser.add_argument('--record',type=Path,required=True);parser.add_argument('--hardware',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--cpu-cores',type=float,default=.35)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    budget=ExperimentBudget(args.cpu_cores,memory_mib=1024,directory=args.output)
    sources=[Path(__file__),ROOT/'neural_navigation/fast_static_localizer.py',
             ROOT/'neural_navigation/verified_static_localizer.py',ROOT/'neural_navigation/measured_odometry.py']
    result=dict(completed=False,deployment_approved=False,
        scope='fixed sensor replay; no closed-loop navigation, no real absolute pose truth',
        baseline_contract=temporal_contract(),candidate_contract=verified_contract(),motion_contract=motion_contract(),
        source_sha256={str(p.relative_to(ROOT)):digest(p) for p in sources},
        record_sha256=digest(args.record/'decisions.jsonl'),hardware_sha256=digest(args.hardware))
    synthetic(args,budget,result)
    recorded(args,budget,result)
    result['completed']=True;result['resource']=budget.statistics()
    (args.output/'RESULTS.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(validation=result['simulation_validation'],motion=result['recorded_motion'],resource=result['resource'])),flush=True)


if __name__=='__main__':main()
