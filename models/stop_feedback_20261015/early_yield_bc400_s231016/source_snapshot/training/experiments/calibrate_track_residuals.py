"""Test previous sensor forecasts against subsequent observed track centroids.

Forecasts use past residuals only. Centroids are partial contours, not true
object centers. This measures consistency, not collision prediction accuracy.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from neural_navigation.track_uncertainty import TrackInnovationEnvelope


def audit(path):
    trace=json.loads(path.read_text());estimator=TrackInnovationEnvelope()
    previous={};samples=[]
    for row in trace['samples']:
        if 'sensor_diagnostic' not in row:continue
        tracks=row['sensor_diagnostic']['tracks'];now=row['timestamp']
        support=row['policy_observation'][281]>.5
        for t in tracks:
            old=previous.get(t['id']);stamp=float(t['time'])
            if old is None or stamp<=old['time'] or t['hits']<old['hits']:continue
            dt=stamp-old['time']
            if not (0<dt<=.8 and old['supported'] and support):continue
            predicted=old['center']+old['velocity']*dt
            error=np.asarray(t['center'])-predicted
            normal=old['normal'];matrix=old['matrix']
            directional=max(.025,2*np.sqrt(max(0.,normal@matrix@normal)))
            isotropic=max(.025,2*np.sqrt(max(0.,np.linalg.eigvalsh(matrix)[-1])))
            # Same declared positional floor on both forecasts. No fit to
            # this next measurement, no claim that .04 is a calibrated CI.
            bounds=[.04+directional*dt,.04+isotropic*dt]
            projected=abs(float(error@normal))
            samples.append(dict(timestamp=now,track_id=t['id'],dt_s=dt,
                past_residual_samples=old['samples'],centroid_error_m=float(np.linalg.norm(error)),
                normal_error_m=projected,directional_bound_m=float(bounds[0]),
                isotropic_bound_m=float(bounds[1]),directional_exceeded=bool(projected>bounds[0]),
                isotropic_exceeded=bool(projected>bounds[1])))
        # Update only AFTER measuring old predictions, avoiding future leakage.
        annotated=estimator.update(tracks,now)
        for t in annotated:
            if t['hits']<3 or np.linalg.norm(t['velocity'])<.04:continue
            old=previous.get(t['id'])
            if old is not None and t['time']==old['time']:continue
            residuals=np.asarray(estimator.states[t['id']]['residuals']).reshape(-1,2)
            matrix=residuals.T@residuals/len(residuals) if len(residuals) else np.zeros((2,2))
            delta=np.asarray(t['center'])-np.asarray(row['estimate'])[:2]
            norm=np.linalg.norm(delta)
            if norm<1e-12:continue
            previous[t['id']]=dict(time=t['time'],hits=t['hits'],center=np.asarray(t['center']),
                velocity=np.asarray(t['velocity']),normal=delta/norm,matrix=matrix,
                samples=len(residuals),supported=support)
        keys={t['id'] for t in annotated};previous={k:v for k,v in previous.items() if k in keys}
    mature=[s for s in samples if s['past_residual_samples']>=3]
    def summarize(rows):
        return dict(measurement_pairs=len(rows),directional_exceeded=sum(s['directional_exceeded'] for s in rows),
            isotropic_exceeded=sum(s['isotropic_exceeded'] for s in rows),
            centroid_error_p95_m=float(np.quantile([s['centroid_error_m'] for s in rows],.95)) if rows else None,
            normal_error_p95_m=float(np.quantile([s['normal_error_m'] for s in rows],.95)) if rows else None)
    return dict(trace=str(path.resolve()),trace_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        all_pairs=summarize(samples),mature_pairs=summarize(mature),samples=samples)


def main():
    p=argparse.ArgumentParser();p.add_argument('--trace',type=Path,action='append',required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():p.error('Fresh output required')
    results=[audit(p) for p in a.trace]
    a.output.write_text(json.dumps(dict(results=results,future_observations_used_in_forecast=False,
        truth_labels_used=False,runtime_changed=False,
        limitation='Small correlated centroid sequences; partial shape and localization affect residuals. '
            'Not object-center truth, a confidence interval, future collision coverage, or closed-loop success.'),indent=2))
    print(json.dumps([dict(trace=r['trace'],all_pairs=r['all_pairs'],mature_pairs=r['mature_pairs']) for r in results]))


if __name__=='__main__':main()
