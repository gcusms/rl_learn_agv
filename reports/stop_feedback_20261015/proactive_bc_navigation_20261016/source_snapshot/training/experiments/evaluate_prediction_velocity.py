"""Evaluation labels for the explicitly inspected 015 obstacle/track match.

This is an offline diagnostic, not a policy, general association benchmark,
or calibrated uncertainty metric. Sensor velocities and physical labels must
be compared in the same odometry axes, which differ from world axes here.
"""
import hashlib
import json
from pathlib import Path
import numpy as np


def main():
    root=Path('reports/localization_mapping_20261015/blind_repair_20261015')
    trace_path=root.parent/'anticipatory_ppo_20261015_132151/paired_12/motion_profile_015_constraints_before_ppo.npz'
    shadow_path=root/'motion_shadow_v2_v3.json'
    z=np.load(trace_path);timestamps=z['labels'][:,0]
    shadow=json.loads(shadow_path.read_text())
    if len(shadow['rows'])!=len(timestamps):raise ValueError('Shadow/trace length mismatch')
    world_velocity=np.gradient(z['human_positions'][:,0],timestamps,axis=0)
    yaw=float(z['labels'][0,3]-z['odometry'][0,2]);c,s=np.cos(yaw),np.sin(yaw)
    odometry_velocity=world_velocity@np.array([[c,-s],[s,c]])
    rows=[]
    for index,record in enumerate(shadow['rows']):
        stamp=float(timestamps[index])
        if not np.isclose(stamp,record['time_s'],rtol=0,atol=1e-8):raise ValueError('Shadow clock mismatch')
        if not 7.3<stamp<8.9:continue
        row=dict(time_s=stamp,physical_velocity_world_m_s=world_velocity[index].tolist(),
            physical_velocity_in_odometry_frame_m_s=odometry_velocity[index].tolist())
        for name in ('cumulative','registered'):
            matches=[track for track in record[name]['tracks'] if track['id']==4]
            if len(matches)!=1:raise ValueError('Explicit evaluation match disappeared')
            track=matches[0];predicted=np.asarray(track['velocity_m_s'])
            row[name]=dict(velocity_odometry_frame_m_s=predicted.tolist(),
                error_m_s=float(np.linalg.norm(predicted-odometry_velocity[index])),
                source=track['source'],contour_uncertainty_m=track['contour_uncertainty_m'])
        rows.append(row)
    report=dict(scope='Evaluation-only comparison on frozen original 015 sensor history; physical labels never enter policy',
        matching='Explicit obstacle 0 / sensor track 4 in inspected 7.4-8.8 s segment; not a general association metric',
        truth_velocity_method='Centered physical finite differences rotated into odometry axes by initial simulator frame transform',
        world_from_odometry_yaw_rad=yaw,
        trace_sha256=hashlib.sha256(trace_path.read_bytes()).hexdigest(),
        shadow_sha256=hashlib.sha256(shadow_path.read_bytes()).hexdigest(),rows=rows)
    output=root/'prediction_velocity_error.json';output.write_text(json.dumps(report,indent=2)+'\n')
    print(output)


if __name__=='__main__':main()
