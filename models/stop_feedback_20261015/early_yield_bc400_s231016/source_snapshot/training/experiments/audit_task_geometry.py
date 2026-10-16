"""Check frozen tasks against the unchanged physical footprint and map.

This is an offline feasibility check. The A* witness is never supplied to the
learned router or actor and does not establish dynamic or nonholonomic success.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import yaml

from navrl.geometry import ROOT
from navrl.map import StaticMap
from neural_navigation.experiment_budget import ExperimentBudget
from neural_navigation.static_footprint import static_footprint_clear


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['cases', 'hardware', 'output']:
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--cpu-cores', type=float, default=.05)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    budget = ExperimentBudget(args.cpu_cores, memory_mib=1024, directory=args.output)
    cases = json.loads(args.cases.read_text())
    if not cases or len({c['id'] for c in cases}) != len(cases):
        raise ValueError('Nonempty unique frozen task identities required')
    hardware = json.loads(args.hardware.read_text())
    footprint = hardware['robot_footprint_m']
    margin = hardware['near_field_memory']['static_footprint_margin_m']
    rows = []
    for case in cases:
        source = ROOT / case['scene'] / 'decision_map.yaml'
        # Same published radius, no start projection or goal relocation.
        static = StaticMap(source, radius=.30)
        map_image = source.parent / yaml.safe_load(source.read_text())['image']
        start = np.asarray(case['start'], float)
        goal = np.asarray(case['goal'], float)
        start_cell, goal_cell = static.cells(start[:2]), static.cells(goal[:2])
        inside = lambda cell: (0 <= cell[0] < static.height and
                                0 <= cell[1] < static.width)
        start_free = bool(inside(start_cell) and static.free[start_cell])
        goal_free = bool(inside(goal_cell) and static.free[goal_cell])
        start_body = bool(static_footprint_clear(static, start, footprint, margin))
        goal_heading = float(goal[2]) if len(goal) == 3 else float(start[2])
        goal_body = bool(static_footprint_clear(static, [*goal[:2], goal_heading], footprint, margin))
        path = None
        error = None
        if start_free and goal_free:
            try:
                path = static.plan(start[:2], goal[:2])
            except ValueError as exception:
                error = str(exception)
        else:
            error = 'Endpoint outside unchanged inflated free space; no snapping allowed'
        path_body = False
        if path is not None:
            delta = np.diff(path, axis=0)
            headings = np.arctan2(delta[:, 1], delta[:, 0])
            headings = np.r_[headings, goal_heading] if len(headings) else [goal_heading]
            path_body = all(static_footprint_clear(static, [*xy, yaw], footprint, margin)
                            for xy, yaw in zip(path, headings))
        row = dict(case_id=case['id'], scene=case['scene'],
            map_sha256=digest(source), map_image_sha256=digest(map_image), map_resolution_m=static.res,
            start_body_clear=start_body, goal_body_clear=goal_body,
            start_in_inflated_free=start_free, goal_in_inflated_free=goal_free,
            connected_in_static_map=path is not None,
            sampled_witness_body_clear=bool(path_body),
            witness_length_m=None if path is None else float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum()),
            straight_distance_m=float(np.linalg.norm(goal[:2] - start[:2])),
            dynamic_obstacles_present=bool(case.get('obstacle')), error=error)
        row['static_geometry_passed'] = bool(start_body and goal_body and path is not None and path_body)
        rows.append(row)
        budget.tick()
    result = dict(completed=True, cases_sha256=digest(args.cases),
        hardware_sha256=digest(args.hardware), footprint_m=footprint, margin_m=margin,
        route_radius_m=.30, start_or_goal_moved=False, rows=rows,
        all_static_geometry_passed=all(r['static_geometry_passed'] for r in rows),
        failed_case_ids=[r['case_id'] for r in rows if not r['static_geometry_passed']],
        failed_cases_filtered=False, runtime_or_training_changed=False,
        limitations=['Offline static witness only; dynamic feasibility and closed-loop navigation remain to be measured.',
                     'Sampled headings do not certify all continuous turns or the fastest/shortest achievable robot trajectory.',
                     'A witness plan is not inserted into policy observations, runtime routing or training.'],
        resource=budget.statistics(), deployment_approved=False)
    (args.output / 'RESULTS.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
