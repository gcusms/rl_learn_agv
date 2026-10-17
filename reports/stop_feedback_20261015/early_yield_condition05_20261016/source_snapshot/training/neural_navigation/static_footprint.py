"""Conservative oriented-body intersection with occupied/unknown map cells.

Uses the physical rectangular footprint, an explicit margin, and every
intersected cell. This helper does not select commands or change a guard.
"""
import numpy as np


def static_footprint_clear(static_map, pose, footprint, margin=.04):
    pose = np.asarray(pose, float)
    footprint = np.asarray(footprint, float)
    if pose.shape != (3,) or footprint.shape != (2,) or not np.isfinite(np.r_[pose, footprint, margin]).all():
        raise ValueError('Finite planar pose and rectangular footprint required')
    if min(footprint) <= 0 or margin < 0:
        raise ValueError('Positive footprint and nonnegative margin required')
    half = footprint/2+margin
    c, s = np.cos(pose[2]), np.sin(pose[2])
    rotation = np.array([[c, -s], [s, c]])
    extent = np.abs(rotation)@half
    origin = static_map.origin[:2]
    size = np.array([static_map.width, static_map.height])*static_map.res
    lower, upper = pose[:2]-extent-origin, pose[:2]+extent-origin
    if np.any(lower < 0) or np.any(upper >= size):
        return False  # Outside the known map is never assumed clear.
    low = np.floor(lower/static_map.res).astype(int)
    high = np.floor(upper/static_map.res).astype(int)
    r0, r1 = static_map.height-1-high[1], static_map.height-low[1]
    c0, c1 = low[0], high[0]+1
    occupied = np.argwhere(static_map.blocked[r0:r1, c0:c1])
    if not len(occupied):
        return True
    rows = occupied[:, 0]+r0; cols = occupied[:, 1]+c0
    centers = origin+np.c_[(cols+.5)*static_map.res,
        (static_map.height-rows-.5)*static_map.res]
    delta = centers-pose[:2]
    cell_half = static_map.res/2
    # Separating-axis theorem: the two map axes and two body axes.
    world_overlap = (np.abs(delta) <= extent+cell_half+1e-12).all(axis=1)
    body_overlap = (np.abs(delta@rotation) <= half+cell_half*(abs(c)+abs(s))+1e-12).all(axis=1)
    return not bool(np.any(world_overlap & body_overlap))
