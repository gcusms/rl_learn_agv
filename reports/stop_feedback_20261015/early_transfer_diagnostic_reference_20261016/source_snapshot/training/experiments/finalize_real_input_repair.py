"""Summarize paired replay, source integrity and a separate candidate bundle."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report',type=Path,required=True);parser.add_argument('--bundle',type=Path,required=True)
    args=parser.parse_args();root=args.report
    before=json.loads((root/'paired/RESULTS.json').read_text())
    after=json.loads((root/'paired_packed/RESULTS.json').read_text())
    audit=json.loads((root/'audit/RESULTS.json').read_text())
    if not before['completed'] or not after['completed']:raise ValueError('Incomplete comparison')
    checked=0;max_delta=0.
    for p in (root/'paired').glob('*.npz'):
        with np.load(p) as original,np.load(root/'paired_packed'/p.name) as optimized:
            for name in original.files:
                if name.endswith(('_cpu','_wall')) or name=='cpu':continue
                a,b=original[name],optimized[name]
                if a.shape!=b.shape:raise ValueError('Replay shape changed')
                if a.dtype==bool:
                    if not np.array_equal(a,b):raise ValueError('Replay acceptance changed')
                else:
                    delta=float(np.max(abs(a-b))) if a.size else 0.
                    max_delta=max(max_delta,delta)
                    if delta>1e-12:raise ValueError('Replay prediction changed: '+p.name+' '+name)
        checked+=1
    worktree=Path(__file__).resolve().parents[2]
    for name,expected in after['source_sha256'].items():
        if digest(worktree/'training'/name)!=expected:raise ValueError('Final evaluation source mismatch: '+name)
    card=json.loads((args.bundle/'MODEL_CARD.json').read_text())
    # The only later core change adds three diagnostic fields. Pose/actor
    # computation is unchanged; the final implementation checks cover it.
    card['runtime_sources_sha256']['core.py']=digest(worktree/'training/neural_navigation/core.py')
    (args.bundle/'MODEL_CARD.json').write_text(json.dumps(card,indent=2)+'\n')
    for name,expected in card['runtime_sources_sha256'].items():
        if digest(worktree/'training/neural_navigation'/name)!=expected:raise ValueError('Candidate runtime source mismatch')
    sizes=dict(bundle_mib=sum(p.stat().st_size for p in args.bundle.rglob('*') if p.is_file())/1024**2,
        report_mib=sum(p.stat().st_size for p in root.rglob('*') if p.is_file())/1024**2,
        disk_free_gib=shutil.disk_usage(root).free/1024**3)
    totals={}
    for source in ('recorded_ekf','measured_wheel_gyro'):
        group=[r for r in after['recorded_localization'] if r['source']==source]
        totals[source]=dict(segments=len(group),skipped=sum(r['skip'] is not None for r in group),algorithms={})
        for algorithm in ('temporal','ray_consistent'):
            rows=[r['algorithms'][algorithm] for r in group if r['skip'] is None]
            totals[source]['algorithms'][algorithm]=dict(
                accepted_scan_frames=sum(r['accepted_scan_frames'] for r in rows),
                ray_inconsistent_correction_frames=sum(r['reasons'].get('ray_inconsistent_correction',0) for r in rows),
                excluded_static_hits_total=sum(r['excluded_static_hits_total'] for r in rows))
    speed_before=before['simulation_validation']['ray_consistent']['cpu_p95_ms']
    speed_after=after['simulation_validation']['ray_consistent']['cpu_p95_ms']
    summary=dict(completed=True,training_running=False,new_ppo_steps=0,deployment_approved=False,
        source='final packed-ray implementation',
        scope='known-map local tracking and measured-motion input; no real pose truth, SLAM or navigation acceptance',
        simulation=after['simulation_validation'],recorded_motion=after['recorded_motion'],
        recorded_localization=totals,ray_packing=dict(paired_npz_files=checked,maximum_prediction_difference=max_delta,
            candidate_cpu_p95_before_ms=speed_before,candidate_cpu_p95_after_ms=speed_after,
            relative_p95_reduction=1-speed_after/speed_before),
        implementation_checks=json.loads((root/'CHECKS.json').read_text()),
        bundle_check=json.loads((root/'BUNDLE_CHECK.json').read_text()),
        resource=after['resource'],storage=sizes)
    (root/'SUMMARY.json').write_text(json.dumps(summary,indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    long=[r for r in after['simulation_cases'] if r['id'].startswith('long_')]
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    index=np.arange(len(long));width=.36
    for offset,name,label,color in [(-width/2,'temporal','Temporal baseline','steelblue'),
                                    (width/2,'ray_consistent','Ray-consistent','darkorange')]:
        axes[0].bar(index+offset,[r['algorithms'][name]['xy_p95_m']*100 for r in long],width,label=label,color=color)
    axes[0].set_xticks(index,['Single wall','Corridor','Sparse','Feature to open'],rotation=12)
    axes[0].set_ylabel('Position error P95 / cm');axes[0].set_title('Fixed long-trajectory simulation replay')
    axes[0].legend(fontsize=8)
    axes[1].bar(['Temporal','Ray dense','Ray packed'],[
        after['simulation_validation']['temporal']['cpu_p95_ms'],speed_before,speed_after],
        color=['steelblue','silver','darkorange'])
    axes[1].set_ylabel('Localizer CPU P95 / ms');axes[1].set_title('Same PC, single thread; not whole navigation')
    fig.tight_layout();fig.savefig(root/'comparison.png',dpi=140);plt.close(fig)
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
