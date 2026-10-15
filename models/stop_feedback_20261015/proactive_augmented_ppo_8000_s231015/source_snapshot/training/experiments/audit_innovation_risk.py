"""Sensor-only comparison of fixed and empirical velocity envelopes."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from experiments.audit_predictor_lifecycle import ReplayState
from neural_navigation.laser_fusion import LaserMapFusion
from neural_navigation.motion_risk import predicted_body_risk
from neural_navigation.predictive_risk import RISK_COMMANDS
from neural_navigation.masked_actions import ACTIONS,masks_from_observation
from neural_navigation.track_uncertainty import TrackInnovationEnvelope
from navrl.geometry import ROOT


def audit(path,hardware,predictor_config):
    trace=json.loads(path.read_text());estimator=TrackInnovationEnvelope();state=None;fusion=None;last=-100.
    counts=dict(frames=0,supported_frames=0,fixed_selected_warning_frames=0,
        adaptive_selected_warning_frames=0,adaptive_all_accepted_choices_warning_frames=0)
    changes=[]
    for row in trace['samples']:
        if 'sensor_diagnostic' in row:
            tracks=row['sensor_diagnostic']['tracks']
        else:
            if state is None:
                scene=Path(trace['case']['scene'])
                if not scene.is_absolute():scene=ROOT/scene
                state=ReplayState(scene,hardware,predictor_config);fusion=LaserMapFusion(state)
            state.elapsed=row['timestamp'];state.estimate=np.asarray(row['estimate'])
            state.scan=np.asarray([np.nan if x is None else x for x in row['scan']])
            d=row['diagnostics'];state.localizer.std=np.asarray(d['localization_std'])
            state.localizer.scan_update_accepted=d['localization_scan_update_accepted']
            state.localizer.map_agreement=d['localization_map_agreement']
            if state.elapsed-last>=.4:
                fusion.update(state);last=state.elapsed
            if fusion.predictor.moving_tracks!=d['scan_moving_tracks'] or fusion.forecast_cells!=d['scan_forecast_cells']:
                raise ValueError('Fusion counters failed reproduction')
            tracks=fusion.predictor.tracks
        annotated=estimator.update(tracks,row['timestamp'])
        args=(annotated,row['estimate'],row['timestamp'],RISK_COMMANDS,hardware['robot_footprint_m'])
        fixed=predicted_body_risk(*args,period_s=.2)
        adaptive=predicted_body_risk(*args,period_s=.2,adaptive_track_uncertainty=True)
        obs=np.asarray(row['policy_observation'],np.float32);mask=masks_from_observation(obs)
        chosen=int(np.argmin(np.max(np.abs(ACTIONS-np.asarray(row['proposed_action'])),axis=1)))
        supported=obs[281]>.5
        old=np.array([r['predicted_margin_contact_s'] is not None for r in fixed])
        new=np.array([r['predicted_margin_contact_s'] is not None for r in adaptive])
        counts['frames']+=1;counts['supported_frames']+=int(supported)
        counts['fixed_selected_warning_frames']+=int(supported and old[chosen])
        counts['adaptive_selected_warning_frames']+=int(supported and new[chosen])
        counts['adaptive_all_accepted_choices_warning_frames']+=int(supported and new[mask].all())
        if supported and np.any(new&~old):
            changes.append(dict(timestamp=row['timestamp'],chosen=chosen,
                additional_warning_choices=np.flatnonzero(new&~old).tolist(),
                velocity_envelopes_m_s=[t['velocity_uncertainty_m_s'] for t in annotated]))
    return dict(trace=str(path),trace_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        outcome=trace['result'],counts=counts,changes=changes)


def main():
    p=argparse.ArgumentParser();p.add_argument('--components',type=Path,required=True)
    p.add_argument('--trace',type=Path,action='append',required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():p.error('Fresh output required')
    torch.set_num_threads(1);hardware=json.loads((a.components/'hardware.json').read_text())
    config=torch.load(a.components/'router.pt',map_location='cpu',weights_only=True)['motion_prediction']
    results=[audit(path,hardware,config) for path in a.trace]
    a.output.write_text(json.dumps(dict(results=results,commands_issued=False,
        simulator_truth_used_for_prediction=False,runtime_changed=False,
        calibrated_confidence=False,limitation='Warnings on successful episodes are not proven false alarms: later commands may avoid forecast collisions. All-choice warnings measure potential overconservatism, not physical infeasibility.'),indent=2))
    print(json.dumps([{k:r[k] for k in ('trace','counts')} for r in results]))


if __name__=='__main__':main()
