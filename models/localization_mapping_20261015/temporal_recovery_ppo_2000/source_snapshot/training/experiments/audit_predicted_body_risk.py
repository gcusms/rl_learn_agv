"""Replay sensor fusion and diagnose time-indexed body risk, read-only."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from experiments.audit_predictor_lifecycle import ReplayState
from experiments.predicted_body_risk import predicted_body_risk
from neural_navigation.laser_fusion import LaserMapFusion
from neural_navigation.safety_context import RECOVERY_PROBES
from navrl.geometry import ROOT


def audit(path, hardware, config):
    trace = json.loads(path.read_text())
    scene = Path(trace['case']['scene'])
    if not scene.is_absolute(): scene = ROOT/scene
    state = ReplayState(scene, hardware, config)
    fusion = LaserMapFusion(state); last_update = -100.
    commands = np.vstack([np.zeros(2), RECOVERY_PROBES, [.2, .5], [.2, -.5]])
    frames, mismatches = [], []
    for row in trace['samples']:
        state.elapsed = row['timestamp']; state.estimate = np.asarray(row['estimate'])
        state.scan = np.asarray([np.nan if x is None else x for x in row['scan']])
        diagnostic = row['diagnostics']
        state.localizer.std = np.asarray(diagnostic['localization_std'])
        state.localizer.scan_update_accepted = diagnostic['localization_scan_update_accepted']
        state.localizer.map_agreement = diagnostic['localization_map_agreement']
        if state.elapsed-last_update >= .4:
            fusion.update(state); last_update = state.elapsed
        matches = (fusion.predictor.moving_tracks == diagnostic['scan_moving_tracks']
            and fusion.forecast_cells == diagnostic['scan_forecast_cells'])
        if not matches: mismatches.append(state.elapsed)
        risk = predicted_body_risk(fusion.predictor.tracks, state.estimate, state.elapsed,
            commands, hardware['robot_footprint_m'], period_s=.2,
            horizon_s=float(config.get('horizon_s', 2.4)),
            uncertainty_m_s=float(config.get('uncertainty_m_s', .025)))
        command = row['command']
        frames.append(dict(timestamp=state.elapsed, replay_counter_match=matches,
            idle=abs(command[0]) < .01 and abs(command[1]) < .05,
            accepted_current_probes=row['safety_context']['action_feasibility'],
            predicted_stationary_margin_contact_s=risk[0]['predicted_margin_contact_s'],
            supported_tracks=risk[0]['supported_tracks'], command_risks=risk))
    warnings = [row for row in frames if row['replay_counter_match'] and row['idle']
        and any(row['accepted_current_probes']) and row['predicted_stationary_margin_contact_s'] is not None]
    return dict(trace_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        policy_sha256=trace['result'].get('policy_sha256'), result=trace['result'],
        policy_attribution=('trace_embedded' if 'policy_sha256' in trace['result'] else
            'Not embedded in runtime trace; verify its result and actor against the evaluation report'),
        exact_recorded_counter_match=not mismatches, mismatches=mismatches,
        predicted_idle_margin_risk_with_current_probe_frames=len(warnings),
        first_warning_timestamp=warnings[0]['timestamp'] if warnings else None, frames=frames)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--components', type=Path, required=True)
    p.add_argument('--trace', type=Path, action='append', required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists(): p.error('Fresh output required')
    torch.set_num_threads(1)
    dependencies = [Path(__file__), Path(__file__).with_name('predicted_body_risk.py'),
        Path(__file__).with_name('audit_predictor_lifecycle.py'),
        Path(__file__).parents[1]/'neural_navigation/motion_prediction.py',
        Path(__file__).parents[1]/'neural_navigation/laser_fusion.py']
    fingerprints = {str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest() for path in dependencies}
    hardware = json.loads((args.components/'hardware.json').read_text())
    config = torch.load(args.components/'router.pt', map_location='cpu', weights_only=True)['motion_prediction']
    results = [audit(path, hardware, config) for path in args.trace]
    if any(hashlib.sha256(path.read_bytes()).hexdigest() != fingerprints[str(path.resolve())] for path in dependencies):
        raise ValueError('Prediction audit dependency changed while running')
    args.output.write_text(json.dumps(dict(cases=results, policy_inputs_used='scans and estimated poses only',
        source_sha256=fingerprints,
        hardware_sha256=hashlib.sha256((args.components/'hardware.json').read_bytes()).hexdigest(),
        router_sha256=hashlib.sha256((args.components/'router.pt').read_bytes()).hexdigest(),
        commands_issued=False, runtime_changed=False,
        limitation='Counter reproduction supports fusion consistency, not complete track correctness. '
            'Constant-velocity point forecasts and instantaneous command following are approximate. '
            'No supported forecast means unknown risk, not clear space; predicted margin contacts are not safety certificates.'), indent=2, allow_nan=False))
    print(json.dumps([{key: value for key, value in result.items() if key != 'frames'} for result in results], indent=2))


if __name__ == '__main__': main()
