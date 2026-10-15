"""Paired V1/V2 tracking, including covariance and native/reference parity.

Reuse frozen actual scans. No truth is passed into a localizer. Previously seen
failure fixtures are regression cases, not a new unseen-map acceptance set.
"""
import argparse
import json
from pathlib import Path
import time
import numpy as np
from navrl.geometry import ROOT,wrap
from navrl.map import StaticMap
from neural_navigation.sensors import LidarSpec
from neural_navigation.fast_static_localizer import FastStaticLocalizer,native_contract,sparse_contract,temporal_contract
from neural_navigation.experiment_budget import ExperimentBudget
from experiments.localization_replay_metrics import digest,metrics


def main():
    p=argparse.ArgumentParser();p.add_argument('--cache',type=Path,required=True)
    p.add_argument('--kernel',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--cpu-cores',type=float,default=.35)
    p.add_argument('--temporal',action='store_true');a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False);budget=ExperimentBudget(a.cpu_cores,directory=a.output)
    rows=[];cases=json.loads((a.cache/'TASKS.json').read_text())
    report=dict(completed=False,deployment_approved=False,scope='known-map initial-pose tracking; frozen paired sensor replay',
        candidate_contract=temporal_contract(True) if a.temporal else sparse_contract(True),kernel_sha256=digest(a.kernel),
        source_sha256={str(path.relative_to(ROOT)):digest(path) for path in
            [Path(__file__).resolve(),ROOT/'neural_navigation/fast_static_localizer.py',ROOT/'neural_navigation/static_tracking_kernel.cpp']},cases=rows)
    for case in cases:
        data=np.load(a.cache/(case['id']+'.npz'));meta=json.loads((a.cache/(case['id']+'.json')).read_text())
        lidar=LidarSpec(meta['lidar']);lidar.range_max=10.
        m=StaticMap(ROOT/case['scene']/'decision_map.yaml',radius=.3)
        scan=data['scan'].copy();scan[scan>=10.]=np.inf;estimates={};stds={}
        row=dict(id=case['id'],split=case['split'],mode=case['mode'],frames=len(scan),
            cache_sha256=digest(a.cache/(case['id']+'.npz')),algorithms={})
        candidate=temporal_contract if a.temporal else sparse_contract
        for label,config in [('v1',native_contract()),('v2_python',candidate()),('v2_native',candidate(True))]:
            loc=FastStaticLocalizer(m,lidar,config,a.kernel.resolve());loc.reset(data['prior'],None)
            loc._field(np.array([data['prior'][:2]]));poses=[];std=[];cpu=[];wall=[];quality=[]
            for odom,frame,t in zip(data['odom'],scan,data['t']):
                start=time.perf_counter();ct=time.process_time();pose=loc.update(odom,frame,t)
                cpu.append(time.process_time()-ct);wall.append(time.perf_counter()-start)
                poses.append(pose);std.append(loc.std.copy())
                quality.append([loc.valid_static_hits,loc.observable_rank,int(loc.scan_update_accepted),*loc.observable_eigenvalues])
                budget.tick()
            est=np.asarray(poses);std=np.asarray(std);estimates[label]=est;stds[label]=std
            row['algorithms'][label]=metrics(data['truth'],est,std,wall,cpu)
            np.savez_compressed(a.output/(case['id']+'_'+label+'.npz'),estimate=est,std=std,
                cpu_seconds=cpu,seconds=wall,quality=quality)
        delta=estimates['v2_native']-estimates['v2_python'];delta[:,2]=wrap(delta[:,2])
        row['maximum_pose_difference']=float(abs(delta).max())
        row['maximum_std_difference']=float(abs(stds['v2_native']-stds['v2_python']).max())
        if max(row['maximum_pose_difference'],row['maximum_std_difference'])>1e-6:raise RuntimeError('Native/reference parity failure')
        rows.append(row);report.update(completed=len(rows)==len(cases),resource=budget.statistics())
        (a.output/'RESULTS.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(dict(id=case['id'],p95={k:round(v['xy_p95_m'],4) for k,v in row['algorithms'].items()})),flush=True)


if __name__=='__main__':main()
