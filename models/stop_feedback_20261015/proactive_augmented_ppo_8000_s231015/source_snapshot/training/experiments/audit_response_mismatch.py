"""Offline displacement scoring of instantaneous versus finite-response motion.

Predictors use the command and measured velocity encoded in the actual actor
input. True relative displacement is only the scoring target. This diagnostic
uses simulation force-response settings, not calibrated chassis parameters.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from navrl.geometry import integrate,wrap


def finite_response(velocity,command,dt):
    pose=np.zeros(3);v=np.asarray(velocity,float).copy()
    count=max(1,int(np.ceil(dt/.005)));h=dt/count
    for _ in range(count):
        acceleration=np.clip((command-v)*20.,[-.5,-1.],[.5,1.])
        nxt=v+acceleration*h
        pose=integrate(pose,*((v+nxt)/2),h);v=nxt
    return pose


def corners(pose,half):
    points=np.array([[half[0],half[1]],[half[0],-half[1]],[-half[0],half[1]],[-half[0],-half[1]]])
    c,s=np.cos(pose[2]),np.sin(pose[2])
    return points@np.array([[c,s],[-s,c]])+pose[:2]


def audit(trace,footprint):
    rows=trace['samples'];records=[];half=np.asarray(footprint)/2
    for before,after in zip(rows,rows[1:]):
        dt=after['timestamp']-before['timestamp']
        if not .05<=dt<=.3:raise ValueError('Expected ordinary adjacent control periods')
        observation=np.asarray(before['policy_observation'],float)
        if len(observation)<235:raise ValueError('Recorded native velocity features required')
        velocity=observation[220:222]*np.array([.2,.5]);command=np.asarray(before['command'])
        instantaneous=integrate(np.zeros(3),*command,dt)
        response=finite_response(velocity,command,dt)
        first=np.asarray(before['truth']);last=np.asarray(after['truth'])
        c,s=np.cos(first[2]),np.sin(first[2])
        displacement=(last[:2]-first[:2])@np.array([[c,-s],[s,c]])
        actual=np.r_[displacement,wrap(last[2]-first[2])]
        target=corners(actual,half)
        error=lambda pose:float(np.max(np.linalg.norm(corners(pose,half)-target,axis=1)))
        records.append(dict(timestamp=before['timestamp'],command=command.tolist(),
            observed_velocity=velocity.tolist(),large_velocity_switch=bool(
                abs(command[1]-velocity[1])>.25 or abs(command[0]-velocity[0])>.08),
            instantaneous_corner_error_m=error(instantaneous),response_corner_error_m=error(response)))
    switched=[row for row in records if row['large_velocity_switch']]
    def summary(selected):
        result={'frames':len(selected)}
        for name in ('instantaneous_corner_error_m','response_corner_error_m'):
            values=[row[name] for row in selected]
            result[name]=dict(mean=float(np.mean(values)),p95=float(np.quantile(values,.95)),max=float(np.max(values))) if values else None
        return result
    return dict(case_id=trace['case']['id'],outcome=trace['result'],
        all_frames=summary(records),large_switch_frames=summary(switched),records=records)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--trace',type=Path,action='append',required=True)
    parser.add_argument('--components',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():parser.error('Fresh response audit required')
    hardware=json.loads((args.components/'hardware.json').read_text())
    results=[audit(json.loads(path.read_text()),hardware['robot_footprint_m']) for path in args.trace]
    result=dict(cases=results,trace_sha256={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in args.trace},
        prediction_inputs='recorded measured velocity and commanded velocity only',
        truth_used_only_for_relative_displacement_scoring=True,commands_issued=False,runtime_changed=False,
        diagnostic_response_settings={'gain_s_inv':20.,'linear_acceleration_m_s2':.5,'angular_acceleration_rad_s2':1.},
        limitation='Simulation force-response approximation, ignores lateral slip and actuator scale variation. '
            'Velocity input is clipped to policy feature range. Not a real-hardware calibration or future safety certificate.')
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2))
    print(json.dumps([{key:row[key] for key in ('case_id','all_frames','large_switch_frames')} for row in results],indent=2))


if __name__=='__main__':main()
