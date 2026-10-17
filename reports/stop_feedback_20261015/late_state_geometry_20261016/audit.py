"""Physical-geometry oracle for diagnosis only, never a policy input or label."""
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
scene = ROOT / 'training/assets/room_navigation_scripted'
description = json.loads((scene / 'physical_scene.json').read_text())
half = np.asarray(description['robot_collision_box_m'][:2]) / 2
circumradius = float(np.linalg.norm(half))
xml = ET.parse(scene / 'physical_scene.xml')
radius = [float(xml.find(".//geom[@name='" + name + "']").attrib['size'].split()[0])
    for name in ['crossing_obstacle_geom', 'corridor_obstacle_geom']]
walls = np.asarray(description['walls'])
trace = ROOT / 'reports/stop_feedback_20261015/context_broad_diagnostic_20261016/physical/hard3_w11_training_stall_t106.npz'
with np.load(trace, allow_pickle=False) as z:
    index = np.flatnonzero(z['model_controlled'])[0]
    timestamp = float(z['labels'][index, 0])
    pose = z['labels'][index, 1:4].copy()
    humans = z['human_positions'][index].copy()
    legal = np.flatnonzero(z['masks'][index]).tolist()


def body_circle_gaps(center, yaw):
    offsets = humans - center
    c, s = np.cos(yaw), np.sin(yaw)
    local = np.column_stack([offsets[:, 0] * c + offsets[:, 1] * s,
        -offsets[:, 0] * s + offsets[:, 1] * c])
    outside = np.maximum(np.abs(local) - half, 0.)
    return np.linalg.norm(outside, axis=1) - radius


def point_wall_distances(center):
    outside = np.maximum(np.maximum(walls[:, :2] - center, center - walls[:, 2:]), 0.)
    return np.linalg.norm(outside, axis=1)


rows = []
for retreat in [0., .02, .04, .08, .10]:
    center = pose[:2] - retreat * np.array([np.cos(pose[2]), np.sin(pose[2])])
    # For a fixed rectangle center, its furthest corner can align with each
    # circle: the exact minimum circle gap over all yaw angles is d-r-R.
    gaps = np.linalg.norm(humans - center, axis=1) - radius - circumradius
    wall_bound = float((point_wall_distances(center) - circumradius).min())
    minimum = float(min(gaps.min(), wall_bound))
    # Independently compare the formula with dense actual box/circle rotation.
    angles = np.linspace(-np.pi, np.pi, 36001)
    offset = humans[0] - center
    local = np.column_stack([offset[0] * np.cos(angles) + offset[1] * np.sin(angles),
        -offset[0] * np.sin(angles) + offset[1] * np.cos(angles)])
    sampled = float(np.linalg.norm(np.maximum(np.abs(local) - half, 0.), axis=1).min() - radius[0])
    assert abs(sampled - gaps[0]) < 1e-7
    current_yaw_gaps = []
    for distance in np.linspace(0., retreat, 101):
        waypoint = pose[:2] - distance * np.array([np.cos(pose[2]), np.sin(pose[2])])
        current_yaw_gaps.append(min(body_circle_gaps(waypoint, pose[2]).min(),
            (point_wall_distances(waypoint) - circumradius).min()))
    rows.append(dict(hypothetical_straight_retreat_m=retreat,
        center=center.tolist(), exact_circle_min_gap_over_all_yaws_m=float(gaps.min()),
        conservative_wall_min_gap_over_all_yaws_m=wall_bound,
        min_gap_minus_nominal4cm_m=minimum - .04,
        min_gap_minus4cm_plus5mm_sensitivity_m=minimum - .045,
        straight_retreat_static_geometry_min_gap_m=float(min(current_yaw_gaps)),
        dense_rotation_formula_difference_m=float(abs(sampled - gaps[0]))))
result = dict(completed=True, trace_sha256=hashlib.sha256(trace.read_bytes()).hexdigest(),
    time_s=timestamp, truth_pose=pose.tolist(), humans=humans.tolist(),
    robot_half_size_m=half.tolist(), robot_circumradius_m=circumradius,
    physical_circle_radii_m=radius, nominal_margin_m=.04, currently_legal_choices=legal,
    current_heading_physical_circle_gap_m=float(body_circle_gaps(pose[:2], pose[2]).min()),
    rows=rows, simulator_truth_is_diagnostic_only=True,
    new_physical_steps=0, new_training_or_positive_labels=False,
    no_action_or_guard_or_runtime_change=True,
    limitations=['This is static geometry, not a dynamics/guard/observability reachability proof.',
        'A5mm perturbation is a sensitivity scenario, not measured uncertainty calibration.',
        'Hypothetical retreat has no rear-visibility certificate and is not authorized by this audit.',
        'The current12action policy cannot command reverse; any added action needs a new contract and training.',
        'A currently legal creep can still lead to a constrained continuation.'])
(OUT/'RESULTS.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({'completed':True,'time_s':timestamp,'rows':rows}))
