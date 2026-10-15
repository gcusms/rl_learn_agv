"""Measure waiting and resumption from physical traces, without calling a policy.

A legal alternative is not necessarily useful or safe in the future. These
are descriptive metrics; they never relabel a collision as a successful wait.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def intervals(timestamps, end_seconds):
    times = np.asarray(timestamps, float)
    if times.ndim != 1 or not len(times) or not np.isfinite(times).all():
        raise ValueError('Finite nonempty frame times required')
    delta = np.diff(np.r_[times, float(end_seconds)])
    if not np.isfinite(delta).all() or np.any(delta < -1e-8):
        raise ValueError('Trace times must precede the physical episode end')
    return np.maximum(delta, 0.)


def segments(selected):
    selected = np.asarray(selected, bool)
    changes = np.diff(np.r_[False, selected, False].astype(int))
    return list(zip(np.flatnonzero(changes == 1), np.flatnonzero(changes == -1)))


def measure(trace, end_seconds, position_tolerance, remaining=None, yaw_tolerance=None):
    labels = np.asarray(trace['labels'], float)
    masks = np.asarray(trace['masks'], bool)
    choices = np.asarray(trace['choices'], int)
    commands = np.asarray(trace['commands'], float)
    n = len(labels)
    if labels.ndim != 2 or labels.shape[1] < 9 or masks.shape != (n, 12):
        raise ValueError('Physical labels and twelve-choice masks required')
    if choices.shape != (n,) or commands.shape != (n, 2):
        raise ValueError('Actions must match physical frames')
    if not np.isfinite(labels).all() or not np.isfinite(commands).all():
        raise ValueError('Finite physical labels and commands required')
    if np.any((choices < 0) | (choices >= 12)) or not masks[np.arange(n), choices].all():
        raise ValueError('Sampled action must be legal in its original mask')
    if remaining is not None:
        remaining = np.asarray(remaining, float)
        if remaining.shape != (n,) or not np.isfinite(remaining).all():
            raise ValueError('Geodesic labels must match physical frames')
    times = labels[:, 0]
    dt = intervals(times, end_seconds)
    available = masks[:, 1:].any(axis=1)
    voluntary = (choices == 0) & available
    idle = np.max(np.abs(commands), axis=1) < 1e-8
    enroute = labels[:, 7] >= position_tolerance
    heading_pending = np.zeros(n, bool) if yaw_tolerance is None else (
        ~enroute & (labels[:, 8] >= yaw_tolerance))
    pending = enroute | heading_pending
    moving = ~idle
    waits = []
    for start, end in segments(voluntary & pending):
        last = min(end, n - 1)
        duration = float(dt[start:end].sum())
        row = dict(start_s=float(times[start]), end_s=float(times[start] + duration),
                   duration_s=duration,
                   includes_heading_alignment=bool(heading_pending[start:end].any()),
                   physical_displacement_m=float(np.linalg.norm(labels[last, 1:3] - labels[start, 1:3])),
                   minimum_legal_motion_choices=int(masks[start:end, 1:].sum(axis=1).min()))
        if remaining is not None:
            row['geodesic_progress_m'] = float(remaining[start] - remaining[last])
        waits.append(row)
    resumptions = []
    for start, end in segments(~available):
        if end >= n or not pending[end]:
            continue
        # Start the recovery clock when a movement first becomes legal, not
        # when perception/geometry still forbids all movements.
        release = float(times[end])
        candidates = np.flatnonzero(moving[end:])
        motion = end + int(candidates[0]) if len(candidates) else None
        horizon = int(np.searchsorted(times, release + 5., side='right') - 1)
        row = dict(forced_start_s=float(times[start]), release_s=release,
                   preceding_only_stop_s=float(dt[start:end].sum()),
                   first_motion_delay_s=None if motion is None else float(times[motion] - release),
                   observation_window_s=float(times[horizon] - release),
                   physical_displacement_m=float(np.linalg.norm(labels[horizon, 1:3] - labels[end, 1:3])))
        if remaining is not None:
            row['geodesic_progress_m'] = float(remaining[end] - remaining[horizon])
        resumptions.append(row)
    late_delays = [r['first_motion_delay_s'] for r in resumptions if r['first_motion_delay_s'] is not None]
    return dict(frames=n, physical_seconds=float(end_seconds),
                voluntary_wait_s=float(dt[voluntary].sum()),
                enroute_voluntary_wait_s=float(dt[voluntary & enroute].sum()),
                heading_alignment_wait_s=float(dt[voluntary & heading_pending].sum()),
                unfinished_task_wait_s=float(dt[voluntary & pending].sum()),
                only_stop_s=float(dt[~available].sum()),
                longest_unfinished_task_wait_s=max((r['duration_s'] for r in waits), default=0.),
                long_unfinished_task_waits=[r for r in waits if r['duration_s'] >= 2. - 1e-8],
                unfinished_task_wait_segments=waits, resumptions=resumptions,
                releases_without_further_motion=sum(r['first_motion_delay_s'] is None for r in resumptions),
                longest_observed_release_delay_s=max(late_delays, default=0.),
                final_command_idle=bool(idle[-1]), final_only_stop=bool(not available[-1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evaluation', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--variant', action='append', help='Audit only named archived variants')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    report = json.loads(args.evaluation.read_text())
    if not report['completed']:
        raise ValueError('Completed evaluation required')
    folder = args.evaluation.parent
    cases = {c['id']: c for c in json.loads((folder / 'cases.json').read_text())}
    from navrl.geometry import ROOT
    from navrl.map import StaticMap
    from neural_navigation.training_distance import TrainingDistanceGraph
    from neural_navigation.goal_pose import contract
    graphs = {}
    rows = []
    for row in report['rows']:
        if args.variant and row['variant'] not in args.variant:
            continue
        case = cases[row['case_id']]
        filename = folder / (row['case_id'] + '_' + row['variant'] + '.npz')
        if hashlib.sha256(filename.read_bytes()).hexdigest() != row['trace_sha256']:
            raise ValueError('Trace differs from evaluated artifact: ' + str(filename))
        scene = case['scene']
        if scene not in graphs:
            graphs[scene] = TrainingDistanceGraph(StaticMap(ROOT / scene / 'decision_map.yaml', radius=.3))
        field = graphs[scene].field(case['goal'][:2])
        with np.load(filename) as trace:
            remaining = np.array([field.value(p) for p in trace['labels'][:, 1:3]])
            tolerance = contract()['position_tolerance_m'] if len(case['goal']) == 3 else .2
            yaw = contract()['yaw_tolerance_rad'] if len(case['goal']) == 3 else None
            metrics = measure(trace, row['seconds'], tolerance, remaining, yaw)
        rows.append(dict(case_id=row['case_id'], variant=row['variant'],
                         safe_success=row['safe_success'], collisions=row['collisions'],
                         false_arrival=row['false_arrival'], timeout=row['timeout'], **metrics))
    if not rows:
        raise ValueError('No matching evaluated rows')
    summary = {}
    for name in sorted({r['variant'] for r in rows}):
        group = [r for r in rows if r['variant'] == name]
        summary[name] = dict(tasks=len(group), safe_success=sum(r['safe_success'] for r in group),
            collisions=sum(r['collisions'] for r in group), timeouts=sum(r['timeout'] for r in group),
            false_arrivals=sum(r['false_arrival'] for r in group),
            enroute_voluntary_wait_s=sum(r['enroute_voluntary_wait_s'] for r in group),
            heading_alignment_wait_s=sum(r['heading_alignment_wait_s'] for r in group),
            unfinished_task_wait_s=sum(r['unfinished_task_wait_s'] for r in group),
            only_stop_s=sum(r['only_stop_s'] for r in group),
            longest_unfinished_task_wait_s=max(r['longest_unfinished_task_wait_s'] for r in group),
            long_unfinished_task_wait_segments=sum(len(r['long_unfinished_task_waits']) for r in group),
            releases_without_further_motion=sum(r['releases_without_further_motion'] for r in group))
    result = dict(completed=True, rows=rows, summary=summary, deployment_approved=False,
        evaluation_sha256=hashlib.sha256(args.evaluation.read_bytes()).hexdigest(),
        method='Actual frame durations; exclude pose-goal settling but count an unfinished heading alignment; report legal-choice and only-stop periods separately.',
        limitations=['A legal alternative does not guarantee useful movement or future safety.',
                     'Offline physical geodesic labels are metrics only, not policy inputs.',
                     'No wait count alone proves better navigation or real-world readiness.'])
    (args.output / 'RESULTS.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(completed=True, summary=summary), indent=2))


if __name__ == '__main__':
    main()
