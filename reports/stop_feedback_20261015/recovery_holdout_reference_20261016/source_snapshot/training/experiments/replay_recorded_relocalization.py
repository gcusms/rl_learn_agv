"""Shadow pose-recovery proposals on a real recording; no pose truth or motion.

The core timestamp/received_timestamp are relative to the run epoch, whereas
scan_stamp_ns/decision_stamp_ns are ROS epoch stamps. Never subtract across
these clock domains. Recorded motion cannot be changed by this offline replay.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from navrl.map import StaticMap
from neural_navigation.experiment_budget import ExperimentBudget
from neural_navigation.global_relocalizer import GlobalRelocalizer,StoppedPoseRecovery
from neural_navigation.sensors import LidarSpec
from experiments.benchmark_global_relocalization import digest


def sensor_row(row):
    timestamp=float(row['result']['timestamp'])
    decision_age=(row['decision_stamp_ns']-row['scan_stamp_ns'])/1e9
    received=max(float(row['inputs']['received_timestamp']),timestamp+decision_age)
    return (np.array(row['inputs']['scan'],float),np.array(row['inputs']['odom'],float),
        np.array(row['inputs']['velocity'],float),timestamp,received)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--record',type=Path,required=True);parser.add_argument('--hardware',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--cpu-cores',type=float,default=.35)
    parser.add_argument('--missions',default='1,7,10,13,19,22')
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    rows=[json.loads(line)['data'] for line in (args.record/'decisions.jsonl').open()]
    runs=list((args.record/'snapshots/runs').iterdir())
    if len(runs)!=1:raise ValueError('Explicit single-run recording required')
    map_path=runs[0]/'prepared_map/map.yaml';m=StaticMap(map_path,radius=.3)
    spec=LidarSpec(json.loads(args.hardware.read_text())['lidar']);matcher=GlobalRelocalizer(m,spec)
    budget=ExperimentBudget(args.cpu_cores,memory_mib=1024,directory=args.output)
    result=dict(completed=False,deployment_approved=False,independent_pose_truth_available=False,
        scope='real fixed-input shadow; estimates and eligibility only, no accuracy/success or actual stationary guarantee',
        limitations=['Recorded commands cannot be changed; recovery does not produce a new closed-loop robot trajectory.',
            'Reported velocity uses the recorded odometry chain, not independently measured stationary truth.',
            'The initial/reset poses and original model estimates are not localization ground truth.'],
        record_sha256=digest(args.record/'decisions.jsonl'),map_yaml_sha256=digest(map_path),hardware_sha256=digest(args.hardware),
        source_sha256={path.name:digest(path) for path in [Path(__file__),Path(__file__).parents[1]/'neural_navigation/global_relocalizer.py']},cases=[])
    for mission in map(int,args.missions.split(',')):
        group=[row for row in rows if row['mission_id']==mission];chosen=None
        for index,row in enumerate(group):
            scan,odom,velocity,t,received=sensor_row(row)
            if (abs(velocity[0])<=matcher.config['stopped_linear_m_s'] and abs(velocity[1])<=matcher.config['stopped_angular_rad_s']
                and -.05<=received-t<=matcher.config['maximum_scan_age_s'] and row['inputs']['scan_valid']):
                chosen=index;break
        if chosen is None:
            result['cases'].append(dict(mission=mission,available=False,reason='no_fresh_reportedly_stopped_frame'))
        else:
            row=group[chosen];values=sensor_row(row);recovery=StoppedPoseRecovery(matcher)
            recovery.begin(*values)
            while recovery.state=='searching':recovery.advance();budget.tick()
            for later in group[chosen+1:]:
                values=sensor_row(later)
                if values[3]-sensor_row(row)[3]>2.:break
                recovery.observe(*values)
                if recovery.state in ('ready','held'):break
                budget.tick()
            result['cases'].append(dict(mission=mission,available=True,inference_id=row['inference_id'],
                first_scan_stamp_ns=row['scan_stamp_ns'],first_total_age_s=sensor_row(row)[4]-sensor_row(row)[3],
                recovery=recovery.diagnostics()))
        result['resource']=budget.statistics()
        (args.output/'RESULTS.json').write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(dict(mission=mission,available=chosen is not None,
            state=None if chosen is None else recovery.state,reason=None if chosen is None else recovery.reason)),flush=True)
    result['completed']=True;result['resource']=budget.statistics()
    (args.output/'RESULTS.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
