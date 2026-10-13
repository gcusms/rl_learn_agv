"""Read synchronized JSONL sensor frames; emit commands without driving hardware."""
import argparse
import json
import sys
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
    parser.add_argument('--input',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--route-mode',choices=['raw','shortcut'],default='raw')
    parser.add_argument('--native-scan',action='store_true',help='Frames contain native_scan and timestamped odometry_samples; apply optional rolling-scan compensation')
    args=parser.parse_args()
    if args.output.exists():parser.error('Use a fresh output file')
    torch.set_num_threads(1)
    runtime=SensorNavigationRuntime(args.bundle,args.map,args.initial_pose,args.goal,args.route_mode)
    odometry=OdometryBuffer();adapter=NativeScanAdapter(runtime.core.lidar,odometry) if args.native_scan else None
    stream=args.input.open() if args.input else sys.stdin
    args.output.parent.mkdir(parents=True,exist_ok=True)
    try:
        with args.output.open('w') as output:
            for line in stream:
                try:
                    frame=json.loads(line)
                    if 'goal' in frame:runtime.core.set_goal(frame['goal'])
                    if adapter:
                        for row in frame['odometry_samples']:odometry.append(row['timestamp'],row['pose'],row['velocity'])
                        native=frame['native_scan']
                        prepared=adapter.prepare([np.nan if v is None else v for v in native['ranges']],
                                                 native['angle_min'],native['angle_increment'],native['timestamp'],
                                                 native['time_increment'],native['range_min'],native['range_max'])
                        if not prepared.valid:raise ValueError(prepared.reason)
                        result=runtime.step(prepared.scan,prepared.pose,prepared.velocity,prepared.timestamp,frame.get('received_timestamp'),
                                            rear_clearance_known=frame.get('rear_clearance_known',False))
                        result['scan_alignment']=dict(compensated=prepared.compensated,angular_coverage=prepared.coverage,
                                                      reference_timestamp=prepared.timestamp,odometry_calibrated=False)
                    else:
                        scan=np.array([np.nan if v is None else v for v in frame['scan']])
                        result=runtime.step(scan,frame['odom'],frame['velocity'],frame['timestamp'],
                                            frame.get('received_timestamp'),frame.get('scan_valid',True),
                                            frame.get('rear_clearance_known',False))
                except (ValueError,KeyError,RuntimeError) as error:
                    # The emitted stop is an executed command too. Do not leave
                    # the policy's history containing the previous moving action.
                    runtime.core.record_action(np.zeros(2));runtime.guard.reverse_started=None
                    if runtime.context is not None:
                        runtime.context.record(dict(timestamp=runtime.core.elapsed,
                            proposed_action=runtime.context.last_proposal,command=[0.,0.],
                            safety_override=True,stop_reason='invalid_sensor_frame'))
                    result=dict(command=[0.,0.],safety_override=True,stop_reason='invalid_sensor_frame',error=str(error))
                output.write(json.dumps(result,allow_nan=False)+'\n');output.flush()
    finally:
        if args.input:stream.close()


if __name__=='__main__':main()
