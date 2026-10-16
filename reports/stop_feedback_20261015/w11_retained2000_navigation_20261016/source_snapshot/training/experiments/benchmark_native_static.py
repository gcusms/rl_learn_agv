"""Evaluate optional compiled math against its NumPy/SciPy reference on ALL scans."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from navrl.geometry import ROOT,wrap
from navrl.map import StaticMap
from neural_navigation.sensors import LidarSpec
from neural_navigation.fast_static_localizer import FastStaticLocalizer,native_contract
from neural_navigation.experiment_budget import ExperimentBudget


def main():
    p=argparse.ArgumentParser();p.add_argument('--cache',type=Path,required=True);p.add_argument('--kernel',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--cpu-cores',type=float,default=.15);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False);budget=ExperimentBudget(a.cpu_cores,directory=a.output)
    cases=json.loads((a.cache/'TASKS.json').read_text());rows=[]
    for case in cases:
        data=np.load(a.cache/(case['id']+'.npz'));meta=json.loads((a.cache/(case['id']+'.json')).read_text())
        lidar=LidarSpec(meta['lidar']);lidar.range_max=10.;m=StaticMap(ROOT/case['scene']/'decision_map.yaml',radius=.3)
        scan=data['scan'].copy();scan[scan>=10.]=np.inf;results={};outputs={}
        for label,loc in [('python',FastStaticLocalizer(m,lidar)),('native',FastStaticLocalizer(m,lidar,native_contract(),a.kernel.resolve()))]:
            start=time.perf_counter();loc.reset(data['prior'],None);loc._field(np.array([data['prior'][:2]]));startup=time.perf_counter()-start
            pose=[];std=[];durations=[];cpu=[]
            for frame_index,(odom,frame,t) in enumerate(zip(data['odom'],scan,data['t'])):
                start=time.perf_counter();ct=time.process_time();estimate=loc.update(odom,frame,t)
                cpu.append(time.process_time()-ct);durations.append(time.perf_counter()-start);pose.append(estimate);std.append(loc.std.copy());budget.tick()
                if frame_index%100==0:print(json.dumps(dict(progress=case['id'],algorithm=label,frame=frame_index,resource=budget.statistics())),flush=True)
            pose=np.array(pose);std=np.array(std);error=np.linalg.norm(pose[:,:2]-data['truth'][:,:2],axis=1)
            results[label]=dict(xy_p95_m=float(np.quantile(error,.95)),xy_rmse_m=float(np.sqrt(np.mean(error**2))),
                final_xy_m=float(error[-1]),yaw_p95_rad=float(np.quantile(abs(wrap(pose[:,2]-data['truth'][:,2])),.95)),
                cpu_mean_ms=float(np.mean(cpu)*1000),cpu_p95_ms=float(np.quantile(cpu,.95)*1000),
                update_mean_ms=float(np.mean(durations)*1000),update_p95_ms=float(np.quantile(durations,.95)*1000),
                update_p99_ms=float(np.quantile(durations,.99)*1000),startup_ms=startup*1000)
            outputs[label]=(pose,std);np.savez_compressed(a.output/(case['id']+'_'+label+'.npz'),estimate=pose,std=std,seconds=durations,cpu_seconds=cpu)
        delta=outputs['python'][0]-outputs['native'][0];delta[:,2]=wrap(delta[:,2]);pose_delta=float(abs(delta).max());cov_delta=float(abs(outputs['python'][1]-outputs['native'][1]).max())
        row=dict(id=case['id'],split=case['split'],mode=case['mode'],frames=len(scan),algorithms=results,
            maximum_pose_difference=pose_delta,maximum_std_difference=cov_delta,parity=pose_delta<1e-6 and cov_delta<1e-6)
        rows.append(row);report=dict(completed=len(rows)==len(cases),rows=rows,all_parity=all(r['parity'] for r in rows),
            kernel_sha256=hashlib.sha256(a.kernel.read_bytes()).hexdigest(),resource=budget.statistics(),
            scope='same frozen MuJoCo scans, reference and compiled math; known-map local tracking only')
        (a.output/'RESULTS.json').write_text(json.dumps(report,indent=2));print(json.dumps(row),flush=True)
    if not report['all_parity']:raise RuntimeError('Native/reference parity failed')


if __name__=='__main__':main()
