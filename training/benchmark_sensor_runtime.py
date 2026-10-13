"""Benchmark the complete sensor core, actor and stop guard without a simulator.

Input is a recorded JSONL sensor stream; this is an open-loop software benchmark,
not an obstacle-avoidance success test or proof of real-board performance.
"""
import argparse
import json
import resource
import sys
import time
from collections import Counter
from pathlib import Path
import numpy as np
import torch
from neural_navigation.runtime import SensorNavigationRuntime
from neural_navigation.sensors import OdometryBuffer
from neural_navigation.scan_alignment import NativeScanAdapter


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--bundle',type=Path,required=True)
    parser.add_argument('--map',type=Path,required=True)
    parser.add_argument('--initial-pose',nargs=3,type=float,required=True)
    parser.add_argument('--goal',nargs=2,type=float,required=True)
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--route-mode',choices=['raw','shortcut'],default='shortcut')
    parser.add_argument('--native-scan',action='store_true')
    args=parser.parse_args()
    if args.output.exists():parser.error('Use a fresh report path')
    torch.set_num_threads(1)
    frames=[json.loads(line) for line in args.input.read_text().splitlines() if line.strip()]
    if not frames:parser.error('The sensor recording is empty')
    started=time.perf_counter()
    runtime=SensorNavigationRuntime(args.bundle,args.map,args.initial_pose,args.goal,args.route_mode)
    adapter=NativeScanAdapter(runtime.core.lidar,OdometryBuffer()) if args.native_scan else None
    startup=time.perf_counter()-started
    times=[];reasons=Counter();first_command=None
    for frame in frames:
        started=time.perf_counter()
        if adapter:
            for row in frame['odometry_samples']:adapter.odometry.append(row['timestamp'],row['pose'],row['velocity'])
            n=frame['native_scan'];prepared=adapter.prepare([np.nan if v is None else v for v in n['ranges']],n['angle_min'],n['angle_increment'],
                                                          n['timestamp'],n['time_increment'],n['range_min'],n['range_max'])
            if not prepared.valid:raise ValueError(prepared.reason)
            result=runtime.step(prepared.scan,prepared.pose,prepared.velocity,prepared.timestamp,frame.get('received_timestamp'))
        else:
            scan=np.array([np.nan if value is None else value for value in frame['scan']])
            result=runtime.step(scan,frame['odom'],frame['velocity'],frame['timestamp'],frame.get('received_timestamp'))
        times.append(1000*(time.perf_counter()-started))
        if result['safety_override']:reasons[result['stop_reason']]+=1
        if first_command is None:first_command=result['command']
    forbidden=[name for name in ['mujoco','stable_baselines3','gymnasium','rclpy'] if name in sys.modules]
    report=dict(frames=len(frames),route_mode=args.route_mode,native_scan_adapter=bool(adapter),startup_seconds=startup,
                pipeline_ms={name:float(np.percentile(times,q)) for name,q in [('p50',50),('p95',95),('p99',99),('max',100)]},
                peak_process_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
                safety_stop_counts=dict(reasons),safety_override_fraction=sum(reasons.values())/len(frames),
                first_command=first_command,forbidden_runtime_imports=forbidden,
                diagnostics=runtime.core.diagnostics(),
                includes='Localizer, laser-map fusion, route inference/extraction/simplification, observation, actor and emergency guard'+('; native scan resampling, odometry interpolation and rolling-scan compensation' if adapter else ''),
                excludes='Sensor transport and JSON IO'+('' if adapter else '; native scan resampling and odometry timestamp interpolation'),
                limitation='Host CPU open-loop replay of synthetic sensors; not real Orin timing, closed-loop safety or localization validation')
    if forbidden:raise RuntimeError(f'Training/simulator imports leaked into runtime: {forbidden}')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False))
    print(json.dumps(report,indent=2,allow_nan=False))


if __name__=='__main__':main()
