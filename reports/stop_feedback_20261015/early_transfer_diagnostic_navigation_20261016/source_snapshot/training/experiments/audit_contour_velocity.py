"""Read-only contour quality audit on native sensor captures."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from neural_navigation.contour_velocity import ContourAwareVelocity


def audit(path):
    trace=json.loads(path.read_text());quality=ContourAwareVelocity();last={};events=[]
    counts=dict(frames=0,frames_with_held_velocity=0,frames_with_unknown_velocity=0,
        frames_unknown_despite_native_confirmation=0,degraded_measurements=0,recovered_measurements=0)
    for row in trace['samples']:
        if 'sensor_diagnostic' not in row:continue
        source=row['sensor_diagnostic']['tracks'];processed=quality.update(source,row['timestamp'])
        counts['frames']+=1
        counts['frames_with_held_velocity']+=int(any(t['contour_velocity_source']=='held' for t in processed))
        counts['frames_with_unknown_velocity']+=int(any(t['contour_velocity_source']=='unknown' for t in processed))
        counts['frames_unknown_despite_native_confirmation']+=int(any(t['contour_velocity_source']=='unknown' and t['hits']>=3 for t in processed))
        original_by_id={t['id']:t for t in source}
        for track in processed:
            original=original_by_id[track['id']]
            previous=last.get(track['id'])
            if previous and previous['time']==track['time']:continue
            loss=track['contour_loss_event']
            recovered=previous and previous['degraded'] and not track['contour_degraded']
            counts['degraded_measurements']+=int(loss);counts['recovered_measurements']+=int(bool(recovered))
            if loss or recovered:
                events.append(dict(timestamp=row['timestamp'],track_id=track['id'],
                    native_hits=track['hits'],point_count=len(track['points']),loss=loss,
                    recovered=bool(recovered),velocity_source=track['contour_velocity_source'],
                    native_velocity=original['velocity'],processed_velocity=track['velocity'].tolist(),
                    policy_prediction_supported=row['policy_observation'][281]>.5))
            last[track['id']]=dict(time=track['time'],degraded=track['contour_degraded'])
    return dict(trace=str(path.resolve()),trace_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        counts=counts,events=events,recorded_outcome=trace['result'])


def main():
    p=argparse.ArgumentParser();p.add_argument('--trace',type=Path,action='append',required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():p.error('Fresh output required')
    results=[audit(path) for path in a.trace]
    a.output.write_text(json.dumps(dict(results=results,runtime_changed=False,commands_issued=False,
        uncertainty_calibrated=False,limitation='Thresholds are diagnostic assumptions. '
        'Held velocity may be wrong during real acceleration; unknown flags must be explicit to future actor consumers. '
        'No new collision avoidance outcome was tested.'),indent=2))
    print(json.dumps([dict(trace=r['trace'],counts=r['counts']) for r in results]))


if __name__=='__main__':main()
