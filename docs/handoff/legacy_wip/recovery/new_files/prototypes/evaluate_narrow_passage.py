"""Same actor, paired native MuJoCo narrow-passage development evaluation."""
import argparse,json,time,hashlib
from pathlib import Path
import numpy as np
import torch
from navrl.map import StaticMap
from neural_navigation.static_footprint import static_footprint_clear
from prototypes.narrow_passage import generate,swept_segment_clear
from prototypes.narrow_runtime import NarrowRuntime
from experiments.time_consistent_tracking import TimeConsistentFreshRuntime
import experiments.evaluate_quality_regression as evaluation


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--bundle',type=Path,default=Path('models/recovery_20261012/portable'))
    p.add_argument('--all-widths',action='store_true');args=p.parse_args()
    if args.output.exists():p.error('Fresh output required')
    torch.set_num_threads(1);args.output.mkdir(parents=True)
    h=json.loads((args.bundle/'hardware.json').read_text())
    scenes=generate(args.output/'scenes',h)
    report=dict(status='running',stage='geometry_audit',cases=[],geometry=[],started_unix=time.time(),
                actor_sha256=hashlib.sha256((args.bundle/'actor.pt').read_bytes()).hexdigest(),
                hardware=h,guard_unchanged=True,training_started=False,deployment_approved=False,
                limitation='One centered straight-lane seed per selected width; no dynamic obstacle, no hardware calibration, not deployment acceptance.')
    source_names=['prototypes/narrow_passage.py','prototypes/narrow_runtime.py','prototypes/evaluate_narrow_passage.py']
    report['source_sha256']={n:hashlib.sha256(Path(n).read_bytes()).hexdigest() for n in source_names}
    for n in source_names:
        dest=args.output/'source_snapshot'/n;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(Path(n).read_bytes())
    def save():
        tmp=args.output/'RESULTS.tmp';tmp.write_text(json.dumps(report,indent=2));tmp.replace(args.output/'RESULTS.json')
        lines=['# 窄道同模型配对评估','',f"状态：{report['status']} / {report['stage']}",'','| 总余量 | 几何版本 | 安全到达 | 碰撞 | 用时 | 路程 |','|---|---|---|---|---|---|']
        for row in report['cases']:
            r=row['result'];lines.append(f"| {row['extra_width_m']*100:.1f} cm | {row['variant']} | {r['safe_success']} | {r['collisions']} | {r['seconds']:.1f}s | {r['travel_m']:.3f}m |")
        lines+=['','本轮未训练，不替换模型；只有直线窄道测试，不能代表转角、动态堵塞或实车效果。']
        (args.output/'REPORT.md').write_text('\n'.join(lines)+'\n')
    save()
    try:
        for scene in scenes:
            m=StaticMap(Path(scene['scene'])/'decision_map.yaml',radius=.3)
            points=np.c_[np.linspace(2.3,3.7,30),np.full(30,2.)]
            report['geometry'].append(dict(extra_width_m=scene['extra_width_m'],
                original_circle_allows_centerline=bool(m.lookup(points,m.free,outside=False).all()),
                oriented_sweep_allows_centerline=swept_segment_clear(m,[1,2],[5,2],h['robot_footprint_m'],h['near_field_memory']['static_footprint_margin_m'])))
        report['stage']='paired_mujoco_evaluation';save()
        for scene in scenes if args.all_widths else scenes[:1]:
            case=dict(id=f"narrow_{round(scene['extra_width_m']*1000)}_centered",category='narrow_development',
                scene=str(Path(scene['scene']).resolve()),seed=113150,start=scene['start'],goal=scene['goal'])
            for variant,cls in [('original',TimeConsistentFreshRuntime),('oriented',NarrowRuntime)]:
                evaluation.QualitySensorNavigationRuntime=cls
                trace=evaluation.evaluate(case,args.bundle)
                path=args.output/f"{case['id']}_{variant}.json";path.write_text(json.dumps(trace,allow_nan=False))
                report['cases'].append(dict(extra_width_m=scene['extra_width_m'],variant=variant,result=trace['result'],trace_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
                save();print(json.dumps(report['cases'][-1]),flush=True)
        report.update(status='complete',stage='finished',finished_unix=time.time(),
            source_hashes_unchanged=all(hashlib.sha256(Path(n).read_bytes()).hexdigest()==sha for n,sha in report['source_sha256'].items()))
    except BaseException as error:
        report.update(status='failed',error=repr(error));raise
    finally:save()

if __name__=='__main__':main()
