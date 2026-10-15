"""Frozen physical sensor replay: original, v5, fast tracking and native AMCL.

Generate scans ONCE with MuJoCo and keep truth only in evaluator NPZ. Native
AMCL's binary input intentionally has no truth. No motor/ROS publishers.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess
import time
import numpy as np
from navrl.geometry import ROOT,wrap,integrate
from navrl.map import StaticMap
from neural_navigation.sensors import LidarSpec
from neural_navigation.experiment_budget import ExperimentBudget


def fixtures():
    cases=[]
    modes=['normal_feedback','small_initial_offset','unmapped_obstacle','missing_sector','command_odometry','command_and_unmapped']
    for index in range(24):
        cases.append(dict(id=f'short_{index:02d}',split='development' if index<6 else 'validation',
            mode=modes[index%6],seed=117000+index,frames=61,
            scene='assets/room_navigation_scripted' if index//6%2==0 else 'assets/navigation_stage2_dataset/train_00'))
    for index,name in enumerate(['single_long_wall','parallel_corridor','sparse_asymmetric','feature_to_open']):
        cases.append(dict(id='long_'+name,split='validation',mode='long_drift',seed=117210+index,frames=465,
            scene='assets/localization_geometry_20261015/'+name))
    return cases


def generate(args):
    import torch
    from neural_navigation.stress_env import StressNavigationEnv
    from experiments.probe_online_slam import OdometryProfile
    torch.set_num_threads(1);args.output.mkdir(parents=True,exist_ok=False)
    cases=fixtures()
    if args.ids:cases=[c for c in cases if c['id'] in args.ids.split(',')]
    for case in cases:case['seed']+=args.seed_offset
    (args.output/'TASKS.json').write_text(json.dumps(cases,indent=2))
    budget=ExperimentBudget(args.cpu_cores,directory=args.output)
    for case in cases:
        rng=np.random.default_rng(case['seed']);scene=ROOT/case['scene'];mode=case['mode']
        settings=dict(odom_scale=1.,sensor_max_range_m=10.)
        if 'unmapped' in mode:settings.update(obstacle='sudden_block',trigger_travel_m=0.,hold_s=100.)
        if mode=='long_drift':settings.update(extra_scan_dropout=.08)
        env=StressNavigationEnv(case=settings,components=args.components,scene_dir=scene,route_mode='shortcut',max_steps=case['frames']+1)
        try:
            if mode=='long_drift':start=np.array([1.2,6.5,0.])
            else:
                m=env.static_map;cells=np.argwhere(m.clearance>.9)
                start=np.r_[m.xy(cells[int(rng.integers(len(cells)))]),rng.uniform(-np.pi,np.pi)]
            env.reset(seed=case['seed'],options=dict(start=start.tolist(),goal=start[:2].tolist()))
            prior=start.copy()
            if mode=='small_initial_offset':prior+=[.06,-.04,.04]
            profile=OdometryProfile(env.observed_odometry.copy(),1.08,1.025);command_odom=np.zeros(3)
            scans=[];odometry=[];truth=[];timestamps=[];velocities=[]
            for frame in range(case['frames']):
                scan=env.scan.copy()
                if mode=='missing_sector':
                    scan[rng.random(len(scan))<.08]=np.nan
                    scan[abs(env.lidar.angles)<.18]=np.nan
                    if 4.<=env.elapsed<4.6:scan[:]=np.nan
                odom=command_odom.copy() if mode.startswith('command') else env.observed_odometry.copy()
                if mode=='long_drift':odom=profile.update(env.observed_odometry,env.elapsed)
                scans.append(scan);odometry.append(odom);truth.append(env.truth.copy());timestamps.append(env.elapsed);velocities.append(env.measured_velocity.copy())
                if frame==case['frames']-1:break
                if mode=='long_drift':command=np.array([.15,0.]) if frame<200 or frame>=264 else np.array([0.,.25])
                else:command=np.array([.08,.15]) if frame<24 else np.array([0.,.3 if case['seed']%2==0 else -.3])
                env.step(command/[.2,.5]);command_odom=integrate(command_odom,*command,env.dt);budget.tick()
                if env.collision_events or env.hard_guard_activations:break
            np.savez_compressed(args.output/(case['id']+'.npz'),scan=scans,odom=odometry,truth=truth,t=timestamps,velocity=velocities,prior=prior)
            meta=dict(case=case,frames=len(scans),seconds=env.elapsed,travel_m=env.travel,
                collision_events=env.collision_events,physical_guards=env.hard_guard_activations,intervention=env.intervention,
                lidar=env.lidar.signature,physical_xml_sha256=digest(scene/'physical_scene.xml'),map_sha256=digest(scene/'decision_map.pgm'))
            (args.output/(case['id']+'.json')).write_text(json.dumps(meta,indent=2));print(case['id'],len(scans),flush=True)
            (args.output/'progress.json').write_text(json.dumps(dict(completed_cases=len(list(args.output.glob('*.npz'))),total_cases=len(cases),resource=budget.statistics()),indent=2))
        finally:env.close()


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def native_input(path,m,lidar,data):
    occupancy=np.where(m.occupied,1,np.where(m.unknown,0,-1)).astype('i1')
    with path.open('wb') as out:
        out.write(struct.pack('<4i',m.width,m.height,len(data['t']),lidar.beams))
        values=[m.res,*m.origin[:2],lidar.angle_min,lidar.angle_max,lidar.range_min,10.,*lidar.mount[[0,1,3]],*data['prior']]
        out.write(struct.pack('<'+'d'*len(values),*values));out.write(np.flipud(occupancy).tobytes())
        for t,odom,scan in zip(data['t'],data['odom'],data['scan']):out.write(np.r_[t,odom,scan].astype('<f8').tobytes())


def metrics(truth,est,std,timings,cpu_timings):
    xy=np.linalg.norm(est[:,:2]-truth[:,:2],axis=1);yaw=abs(wrap(est[:,2]-truth[:,2]))
    # A diagnostic threshold, not a claim of calibrated probabilistic coverage.
    trusted=(np.max(std[:,:2],axis=1)<.1)&(std[:,2]<.1)
    return dict(frames=len(truth),xy_rmse_m=float(np.sqrt(np.mean(xy**2))),xy_p95_m=float(np.quantile(xy,.95)),
        final_xy_m=float(xy[-1]),yaw_p95_rad=float(np.quantile(yaw,.95)),
        erroneous_trusted_fraction=float(np.mean(trusted&((xy>.2)|(yaw>.2)))),
        update_mean_ms=float(np.mean(timings)*1000),update_p95_ms=float(np.quantile(timings,.95)*1000),
        update_p99_ms=float(np.quantile(timings,.99)*1000),update_max_ms=float(max(timings)*1000),
        cpu_mean_ms=float(np.mean(cpu_timings)*1000),cpu_p95_ms=float(np.quantile(cpu_timings,.95)*1000))


def replay(args):
    import torch
    from neural_navigation.models import LearnedLocalizer
    from neural_navigation.odom_anchored_localizer import OdometryAnchoredLocalizer,precision_contract
    from neural_navigation.fast_static_localizer import FastStaticLocalizer,contract
    torch.set_num_threads(1);args.output.mkdir(parents=True,exist_ok=False)
    budget=ExperimentBudget(args.cpu_cores,directory=args.output);cases=json.loads((args.cache/'TASKS.json').read_text())
    if args.ids:cases=[c for c in cases if c['id'] in args.ids.split(',')]
    source=Path(__file__).with_name('amcl_offline.cpp');executable=args.output/'amcl_offline'
    subprocess.run(['g++','-O3','-std=c++17',str(source),'-I/opt/ros/humble/include','-L/opt/ros/humble/lib','-Wl,-rpath,/opt/ros/humble/lib',
        '-lsensors_lib','-lmotions_lib','-lpf_lib','-lmap_lib','-o',str(executable)],check=True)
    report=dict(completed=False,scope='paired physical sensor tracking; supplied initial map pose, NOT global localization, navigation or real-world accuracy',
        truth='independent MuJoCo rigid-body pose; never supplied to any localizer',candidate_contract=contract(),
        comparison='Nav2 1.1.19 native libraries; omni model; same initial Gaussian [.12,.12,.08], identical 60 selected beam indices and 10m range; compact 300-1000 and budget 100-300 particles, gated and forced updates',
        amcl=dict(min_particles=300,max_particles=1000,alphas=[.1]*5,laser='likelihood_field_prob',sigma_hit=.12,z_hit=.9,z_rand=.1,
            skip_distance=.5,skip_threshold=.3,skip_error_threshold=.9,kld_error=.05,kld_z=.99),
        timing='single-thread local algorithm wall time, IO and quota sleeps excluded; startup measured separately; no ROS transport or 50ms polling wait',
        source_sha256={str(p.relative_to(ROOT)):digest(p) for p in [source,Path(__file__),ROOT/'neural_navigation/fast_static_localizer.py']},
        deployment_approved=False,cases=[])
    for case in cases:
        data=np.load(args.cache/(case['id']+'.npz'));metadata=json.loads((args.cache/(case['id']+'.json')).read_text())
        m=StaticMap(ROOT/case['scene']/'decision_map.yaml',radius=.3)
        lidar=LidarSpec(metadata['lidar']);lidar.range_max=10.
        # Finite 10m+ returns carry no localizable hit for any comparator.
        scan=data['scan'].copy();scan[scan>=10.]=np.inf
        algorithms={
            'legacy':lambda:LearnedLocalizer(m,lidar,args.components/'localizer.pt'),
            'v5':lambda:OdometryAnchoredLocalizer(m,lidar,args.components/'localizer.pt',precision_contract()),
            'fast_static':lambda:FastStaticLocalizer(m,lidar)}
        result=dict(id=case['id'],split=case['split'],mode=case['mode'],frames=len(scan),cache_sha256=digest(args.cache/(case['id']+'.npz')),
            physical_events=dict(collisions=metadata['collision_events'],guards=metadata['physical_guards']),algorithms={})
        estimates={}
        for name,build in algorithms.items():
            start=time.perf_counter();loc=build();loc.reset(data['prior'],None)
            # Fields are initialized before timing updates; preprocessing cost is
            # still reported as startup, just like native AMCL's cspace map.
            if name=='fast_static':loc._field(np.array([data['prior'][:2]]))
            startup=time.perf_counter()-start;poses=[];stds=[];durations=[];cpu_durations=[];reasons=[]
            for odom,frame,t in zip(data['odom'],scan,data['t']):
                start=time.perf_counter();cpu_start=time.process_time();pose=loc.update(odom,frame) if name=='legacy' else loc.update(odom,frame,t)
                cpu_durations.append(time.process_time()-cpu_start)
                durations.append(time.perf_counter()-start);poses.append(pose.copy());stds.append(loc.std.copy());reasons.append(getattr(loc,'correction_reason','legacy'));budget.tick()
            estimates[name]=np.array(poses);result['algorithms'][name]=dict(metrics(data['truth'],np.array(poses),np.array(stds),durations,cpu_durations),startup_ms=startup*1000)
            np.savez_compressed(args.output/(case['id']+'_'+name+'.npz'),estimate=poses,std=stds,seconds=durations,cpu_seconds=cpu_durations,reason=reasons)
        binary=args.output/(case['id']+'.bin');native_input(binary,m,lidar,dict(t=data['t'],odom=data['odom'],scan=scan,prior=data['prior']))
        configs=[('standard',False,300,1000,0.),('skip',True,300,1000,0.),
            ('budget_gated',True,100,300,.02)]
        for tag,skip,minimum,maximum,gate in configs:
            for seed in args.amcl_seeds.split(','):
                label=f'amcl_{tag}_{seed}';csv=args.output/(case['id']+'_'+label+'.csv')
                env=dict(os.environ,RL_EXPERIMENT_CPU_CORE=str(args.cpu_cores))
                run=subprocess.run([str(executable),str(binary),str(csv),str(minimum),str(maximum),str(int(skip)),seed,str(gate)],env=env,capture_output=True,text=True,check=True)
                values=np.genfromtxt(csv,delimiter=',',names=True);est=np.c_[values['x'],values['y'],values['yaw']];std=np.c_[values['std_x'],values['std_y'],values['std_yaw']]
                result['algorithms'][label]=dict(metrics(data['truth'],est,std,values['seconds'],values['cpu_seconds']),native_peak_rss_kib=int(run.stdout.strip().split('=')[-1]),
                    mean_particles=float(np.mean(values['particles'])))
                estimates[label]=est
        binary.unlink();report['cases'].append(result);report['completed']=len(report['cases'])==len(cases);report['resource']=budget.statistics()
        (args.output/'RESULTS.json').write_text(json.dumps(report,indent=2));print(json.dumps(result),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['generate','replay']);p.add_argument('--components',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--cache',type=Path);p.add_argument('--cpu-cores',type=float,default=.35)
    p.add_argument('--ids');p.add_argument('--amcl-seeds',default='11,29,47')
    p.add_argument('--seed-offset',type=int,default=0,help='New physical sensor realization, not an unseen-map claim')
    a=p.parse_args()
    if a.output.exists():p.error('Fresh output required')
    if a.mode=='generate':generate(a)
    elif a.cache is None:p.error('Replay requires --cache')
    else:replay(a)


if __name__=='__main__':main()
