"""Separate actor waits and guard stops in the original real recording.

The robot's estimated pose is not physical ground truth. This report
describes logged choices and constraints, not unjustified stops or accuracy.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import zipfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--zip', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Fresh output file required')
    with zipfile.ZipFile(args.zip) as archive:
        names = [n for n in archive.namelist() if n.endswith('/decisions.jsonl')]
        if len(names) != 1:
            raise ValueError('Exactly one decision stream required')
        raw = archive.read(names[0])
        decisions = [json.loads(line)['data'] for line in raw.splitlines()]
    counts = Counter()
    stop_reasons = Counter()
    rows = []
    runs = []
    active = []

    def close_run():
        if active:
            runs.append(dict(mission_id=active[0]['mission_id'], start_s=active[0]['timestamp'],
                last_sample_s=active[-1]['timestamp'],
                observed_span_s=active[-1]['timestamp'] - active[0]['timestamp'],
                samples=len(active)))
            active.clear()

    for item in decisions:
        result = item['result']
        diagnostics = result['diagnostics']
        legal = sum(float(x) > .5 for x in result['action_feasibility'] + result['fast_probe_feasibility'])
        idle = max(abs(float(x)) for x in result['command']) < 1e-8
        actor_idle = max(abs(float(x)) for x in result['proposed_action']) < 1e-8
        estimated_error = math.dist(diagnostics['estimated_pose'][:2], item['goal'][:2])
        reason = result['stop_reason'] or 'none'
        row = dict(mission_id=item['mission_id'], timestamp=float(result['timestamp']),
            legal_motion_choices=legal, executed_idle=idle, actor_selected_stop=actor_idle,
            executed_stop_reason=reason, estimated_goal_distance_m=estimated_error,
            localization_scan_update_accepted=diagnostics['localization_scan_update_accepted'],
            route_complete=diagnostics['route_complete'])
        counts['frames'] += 1
        counts['only_stop_allowed'] += legal == 0
        counts['executed_idle'] += idle
        counts['idle_with_motion_options'] += idle and legal > 0
        counts['actor_stop_with_motion_options'] += actor_idle and legal > 0
        counts['nonzero_proposal_stopped_with_motion_options'] += idle and not actor_idle and legal > 0
        # This is a logged policy wait away from its estimated target, not a
        # claim that driving would have been useful or physically safe.
        selected = actor_idle and legal > 0 and reason == 'none' and estimated_error > .2
        counts['actor_wait_away_from_estimated_target'] += selected
        if active and (not selected or item['mission_id'] != active[-1]['mission_id']
                       or not 0 < row['timestamp'] - active[-1]['timestamp'] <= .6):
            close_run()
        if selected:
            active.append(row)
        stop_reasons[reason] += 1
        rows.append(row)
    close_run()
    result = dict(completed=True, counts=dict(counts), executed_reasons=dict(stop_reasons),
        recorded_policy_wait_runs=sorted(runs, key=lambda x: -x['observed_span_s']), rows=rows,
        zip_sha256=hashlib.sha256(args.zip.read_bytes()).hexdigest(),
        decisions_member=names[0], decisions_sha256=hashlib.sha256(raw).hexdigest(),
        scope='Original deployed-model logs only; no candidate intervention, physical pose truth or new navigation outcome.',
        limitations=['Legal movement is not necessarily useful or future-safe.',
                     'Estimated target distance is not physical target error.',
                     'A sampled run span omits an unobserved interval after the last sample.',
                     'Manual pose corrections and mission changes break temporal continuity.'],
        deployment_approved=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(completed=True, counts=dict(counts),
        longest_observed_wait_s=max((r['observed_span_s'] for r in runs), default=0.)), indent=2))


if __name__ == '__main__':
    main()
