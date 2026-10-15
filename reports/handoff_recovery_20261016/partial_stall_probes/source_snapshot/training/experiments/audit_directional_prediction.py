"""Read-only isotropic versus directional scan-residual forecast audit."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from experiments.audit_predictor_lifecycle import ReplayState
from navrl.geometry import ROOT
from neural_navigation.laser_fusion import LaserMapFusion
from neural_navigation.track_uncertainty import TrackInnovationEnvelope
from neural_navigation.motion_risk import predicted_body_risk
from neural_navigation.directional_risk import directional_body_risk
from neural_navigation.predictive_risk import RISK_COMMANDS
from neural_navigation.masked_actions import ACTIONS,masks_from_observation


def audit(path,hardware,predictor_config):
    trace=json.loads(path.read_text());estimator=TrackInnovationEnvelope()
    state=None;fusion=None;last=-100.;samples=[]
    counts=dict(frames=0,supported_frames=0,isotropic_selected_warnings=0,
        directional_selected_warnings=0,isotropic_all_choices_warning=0,
        directional_all_choices_warning=0)
    for row in trace['samples']:
        if 'sensor_diagnostic' in row:tracks=row['sensor_diagnostic']['tracks']
        elif 'scan' in row:
            if state is None:
                scene=Path(trace['case']['scene']);scene=scene if scene.is_absolute() else ROOT/scene
                state=ReplayState(scene,hardware,predictor_config);fusion=LaserMapFusion(state)
            state.elapsed=row['timestamp'];state.estimate=np.asarray(row['estimate'])
            state.scan=np.asarray([np.nan if x is None else x for x in row['scan']])
            d=row['diagnostics'];state.localizer.std=np.asarray(d['localization_std'])
            state.localizer.scan_update_accepted=d['localization_scan_update_accepted']
            state.localizer.map_agreement=d['localization_map_agreement']
            if state.elapsed-last>=.4:fusion.update(state);last=state.elapsed
            if fusion.predictor.moving_tracks!=d['scan_moving_tracks'] or fusion.forecast_cells!=d['scan_forecast_cells']:
                raise ValueError('Fusion counters failed reproduction')
            tracks=fusion.predictor.tracks
        else:continue
        tracks=estimator.update(tracks,row['timestamp'])
        for t in tracks:
            residuals=np.asarray(estimator.states[t['id']]['residuals']).reshape(-1,2)
            t['velocity_error_second_moment']=(residuals.T@residuals/len(residuals) if len(residuals) else np.zeros((2,2)))
        args=(tracks,row['estimate'],row['timestamp'],RISK_COMMANDS,hardware['robot_footprint_m'])
        isotropic=predicted_body_risk(*args,period_s=.2,adaptive_track_uncertainty=True)
        directional=directional_body_risk(*args)
        obs=np.asarray(row['policy_observation'],np.float32);mask=masks_from_observation(obs)
        measured_velocity=obs[220:222]*np.array([.2,.5])
        response=directional_body_risk(*args,measured_velocity=measured_velocity)
        chosen=int(np.argmin(np.max(np.abs(ACTIONS-np.asarray(row['proposed_action'])),axis=1)))
        support=obs[281]>.5
        old=np.array([x['predicted_margin_contact_s'] is not None for x in isotropic])
        new=np.array([x['predicted_margin_contact_s'] is not None for x in directional])
        finite=np.array([x['predicted_margin_contact_s'] is not None for x in response])
        counts['frames']+=1;counts['supported_frames']+=int(support)
        for prefix,warn in [('isotropic',old),('directional',new),('directional_response',finite)]:
            counts.setdefault(prefix+'_selected_warnings',0);counts.setdefault(prefix+'_all_choices_warning',0)
            counts[prefix+'_selected_warnings']+=int(support and warn[chosen])
            counts[prefix+'_all_choices_warning']+=int(support and warn[mask].all())
        if support and (np.any(new!=old) or np.any(finite!=new) or 9.2<=row['timestamp']<=10.8):
            samples.append(dict(timestamp=row['timestamp'],chosen=chosen,
                masks=mask.tolist(),isotropic_ttc_s=[x['predicted_margin_contact_s'] for x in isotropic],
                directional_ttc_s=[x['predicted_margin_contact_s'] for x in directional],
                directional_response_ttc_s=[x['predicted_margin_contact_s'] for x in response],
                observed_velocity=measured_velocity.tolist()))
    return dict(trace=str(path.resolve()),trace_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        recorded_outcome=trace['result'],counts=counts,samples=samples)


def main():
    p=argparse.ArgumentParser();p.add_argument('--components',type=Path,required=True)
    p.add_argument('--trace',type=Path,action='append',required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():p.error('Fresh output required')
    torch.set_num_threads(1);hardware=json.loads((a.components/'hardware.json').read_text())
    config=torch.load(a.components/'router.pt',map_location='cpu',weights_only=True)['motion_prediction']
    results=[audit(path,hardware,config) for path in a.trace]
    a.output.write_text(json.dumps(dict(results=results,runtime_changed=False,commands_issued=False,
        simulator_truth_used=False,confidence_calibrated=False,
        limitation='Isotropic/directional share instantaneous point forecasts; additional response variant uses measured velocity and uncalibrated force response. '
        'Fewer warnings are not proof of correctness. Unknown shape, association and turn response remain unmodeled.'),indent=2))
    print(json.dumps([dict(trace=r['trace'],counts=r['counts']) for r in results]))


if __name__=='__main__':main()
