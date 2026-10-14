"""Fixed-input replay: corrections compared to odometry, NOT localization truth."""
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
from navrl.geometry import wrap
from navrl.map import StaticMap
from neural_navigation.sensors import LidarSpec
from neural_navigation.models import LearnedLocalizer
from neural_navigation.odom_anchored_localizer import OdometryAnchoredLocalizer,contract,turn_response_contract,precision_contract


def main():
    p=argparse.ArgumentParser();p.add_argument('--record',type=Path,required=True);p.add_argument('--bundle',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--turn-response-v4',action='store_true',help='Opt-in bounded yaw response; leave v3 default unchanged')
    p.add_argument('--precision-v5',action='store_true',help='Reduced residual deadband and bounded startup refinement')
    a=p.parse_args();configuration=precision_contract() if a.precision_v5 else turn_response_contract() if a.turn_response_v4 else contract()
    torch.set_num_threads(1);a.output.mkdir(parents=True,exist_ok=True)
    rows=[json.loads(l)['data'] for l in (a.record/'decisions.jsonl').open()]
    run=next((a.record/'snapshots/runs').iterdir())
    m=StaticMap(run/'prepared_map/map.yaml',radius=.3);lidar=LidarSpec(json.loads((a.bundle/'hardware.json').read_text())['lidar'])
    previous=None;poses=[];old=[];reasons=[];timings=[];accepted=[]
    baseline=LearnedLocalizer(m,lidar,a.bundle/'localizer.pt')
    candidate=OdometryAnchoredLocalizer(m,lidar,a.bundle/'localizer.pt',configuration)
    for r in rows:
        if r['mission_id']!=previous:
            baseline.reset(r['initial'],None);candidate.reset(r['initial'],None);previous=r['mission_id']
        scan=np.array(r['inputs']['scan'],float);odom=np.array(r['inputs']['odom']);t=r['scan_stamp_ns']/1e9
        old.append(baseline.update(odom,scan));start=time.perf_counter()
        poses.append(candidate.update(odom,scan,t));timings.append((time.perf_counter()-start)*1000)
        reasons.append(candidate.correction_reason);accepted.append(candidate.scan_update_accepted)
    poses=np.array(poses);old=np.array(old);recorded=np.array([r['result']['diagnostics']['estimated_pose'] for r in rows]);missions=[]
    for mid in sorted(set(r['mission_id'] for r in rows)):
        ix=np.flatnonzero([r['mission_id']==mid for r in rows]);od=np.array([rows[i]['inputs']['odom'] for i in ix])
        def distance(x):return float(np.linalg.norm(x[-1,:2]-x[0,:2]))
        missions.append(dict(mission=mid,frames=len(ix),recorded_displacement_m=distance(recorded[ix]),baseline_replay_displacement_m=distance(old[ix]),candidate_displacement_m=distance(poses[ix]),input_odometry_displacement_m=distance(od),candidate_map_corrections=sum(reasons[i]=='accepted_static_correction' for i in ix)))
    result=dict(scope='fixed recorded scan and odometry; no independent pose truth; not closed-loop navigation',contract=configuration,frames=len(rows),candidate_accepted_frames=int(sum(accepted)),reasons={s:reasons.count(s) for s in sorted(set(reasons))},candidate_localizer_ms=dict(p50=float(np.percentile(timings,50)),p95=float(np.percentile(timings,95)),p99=float(np.percentile(timings,99)),maximum=float(max(timings))),missions=missions)
    (a.output/'replay.json').write_text(json.dumps(result,indent=2));np.savez_compressed(a.output/'replay.npz',candidate=poses,baseline=old,recorded=recorded)
    print(json.dumps({k:v for k,v in result.items() if k!='missions'},indent=2));print([r for r in missions if r['mission'] in (13,19)])

if __name__=='__main__':main()
