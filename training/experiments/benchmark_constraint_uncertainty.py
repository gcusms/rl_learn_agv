"""Replay a separate uncertainty budget; assert every pose stays unchanged."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from navrl.geometry import ROOT
from navrl.map import StaticMap
from neural_navigation.sensors import LidarSpec
from neural_navigation.fast_static_localizer import FastStaticLocalizer,temporal_contract
from neural_navigation.weak_geometry_uncertainty import contract
from neural_navigation.experiment_budget import ExperimentBudget
from experiments.localization_replay_metrics import metrics


def main():
    p=argparse.ArgumentParser();p.add_argument('--cache',type=Path,required=True)
    p.add_argument('--reference',type=Path,required=True);p.add_argument('--bundle',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--cpu-cores',type=float,default=.1)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);budget=ExperimentBudget(a.cpu_cores,directory=a.output)
    report=dict(completed=False,pose_accuracy_changed=False,uncertainty_calibrated=False,cases=[])
    for case in json.loads((a.cache/'TASKS.json').read_text()):
        data=np.load(a.cache/(case['id']+'.npz'));meta=json.loads((a.cache/(case['id']+'.json')).read_text())
        lidar=LidarSpec(meta['lidar']);lidar.range_max=10.;scan=data['scan'].copy();scan[scan>=10.]=np.inf
        m=StaticMap(ROOT/case['scene']/'decision_map.yaml',radius=.3)
        loc=FastStaticLocalizer(m,lidar,temporal_contract(True),(a.bundle/'localization_kernel.so').resolve(),uncertainty=contract())
        loc.reset(data['prior'],None);poses=[];std=[];floors=[];times=[];cpu=[]
        for odom,frame,t in zip(data['odom'],scan,data['t']):
            start=time.perf_counter();ct=time.process_time();poses.append(loc.update(odom,frame,t))
            cpu.append(time.process_time()-ct);times.append(time.perf_counter()-start)
            std.append(loc.std.copy());floors.append(loc.weak_geometry_uncertainty.floor.copy());budget.tick()
        poses=np.asarray(poses);std=np.asarray(std)
        previous=np.load(a.reference/(case['id']+'_v2_native.npz'))
        delta=float(abs(poses-previous['estimate']).max())
        if delta>1e-12:raise AssertionError('An uncertainty-only change moved the pose')
        row=dict(id=case['id'],split=case['split'],mode=case['mode'],frames=len(poses),maximum_pose_difference=delta,
            original=metrics(data['truth'],previous['estimate'],previous['std'],times,cpu),
            candidate=metrics(data['truth'],poses,std,times,cpu),
            original_guard_uncertain_fraction=float(np.mean((previous['std'][:,:2].max(axis=1)>.2)|(previous['std'][:,2]>.25))),
            candidate_guard_uncertain_fraction=float(np.mean((std[:,:2].max(axis=1)>.2)|(std[:,2]>.25))))
        np.savez_compressed(a.output/(case['id']+'.npz'),estimate=poses,std=std,budget=floors)
        report['cases'].append(row);report.update(resource=budget.statistics())
        (a.output/'RESULTS.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps({k:row[k] for k in ['id','maximum_pose_difference','original_guard_uncertain_fraction','candidate_guard_uncertain_fraction']}),flush=True)
    report.update(completed=True,contract=contract(),bundle_manifest_sha256=hashlib.sha256((a.bundle/'manifest.json').read_bytes()).hexdigest())
    (a.output/'RESULTS.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
