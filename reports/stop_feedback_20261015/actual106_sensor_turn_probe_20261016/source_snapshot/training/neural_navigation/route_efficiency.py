"""Bounded decoding of learned routes, using sensor/map feasibility only.

No graph search or substitute velocity controller: each candidate follows the
same learned Q field, with a different first direction. Full-path length is
primary; initial goal alignment is a small tie preference, never a safety rule.
"""
import numpy as np


def route_cost(path, position, goal, yaw):
    path = np.asarray(path, float)
    length = float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum())
    offsets = path-position
    indices = np.flatnonzero(np.linalg.norm(offsets, axis=1) > .15)
    if not len(indices):
        return length
    initial = offsets[indices[0]]
    heading = np.arctan2(initial[1], initial[0])
    to_goal = np.asarray(goal)-position
    cosine = float(initial@to_goal/max(np.linalg.norm(initial)*np.linalg.norm(to_goal), 1e-9))
    turn = abs(np.arctan2(np.sin(heading-yaw), np.cos(heading-yaw)))
    # At most 10 cm equivalent: a necessary retreat can still win on length.
    return length+.025*(1-cosine)+.05*turn/np.pi


def efficient_learned_route(core, q, occupancy, original, max_steps=160):
    position = np.asarray(core.estimate[:2])
    start = core.cell(position); target = core.cell(core.goal)
    directions = core.router.directions
    choices = [original]
    h, w = occupancy.shape
    for first in range(len(directions)):
        r, c = start; visited = {(r, c)}; path = [position.copy()]
        for step in range(max_steps):
            if not (0 <= r < h and 0 <= c < w):
                break
            if (r, c) == target:
                path.append(np.asarray(core.goal).copy()); break
            order = [first] if step == 0 else np.argsort(-q[:,r,c], kind='stable')
            selected = None
            for direction in order:
                dr, dc = directions[int(direction)]; nr, nc = r+dr, c+dc
                if (0 <= nr < h and 0 <= nc < w and occupancy[nr,nc]
                        and (nr,nc) not in visited and np.isfinite(q[int(direction),r,c])):
                    selected = nr, nc; break
            if selected is None:
                break
            r, c = selected; visited.add((r,c)); path.append(core.xy((r,c)))
        candidate = np.asarray(path)
        if np.linalg.norm(candidate[-1]-core.goal) < .12:
            candidate = core._shortcut_route(candidate)
            if core._route_usable(candidate):
                choices.append(candidate)
    valid = [p for p in choices if core._route_usable(p)]
    core.route_candidate_count = len(valid)
    if not valid:
        return original
    selected = min(valid, key=lambda p: route_cost(p, position, core.goal, core.estimate[2]))
    core.route_selection_cost = route_cost(selected, position, core.goal, core.estimate[2])
    return selected
