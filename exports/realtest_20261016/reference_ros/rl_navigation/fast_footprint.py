"""Equivalent rectangular map collision check with fewer small allocations."""
import math
import numpy as np


def static_footprint_clear(static_map, pose, footprint, margin=.04):
    pose = np.asarray(pose, float)
    footprint = np.asarray(footprint, float)
    if pose.shape != (3,) or footprint.shape != (2,) or not all(
            math.isfinite(float(v)) for v in (*pose, *footprint, margin)):
        raise ValueError('Finite planar pose and rectangular footprint required')
    if min(footprint) <= 0 or margin < 0:
        raise ValueError('Positive footprint and nonnegative margin required')
    hx, hy = footprint / 2 + margin
    c, s = np.cos(pose[2]), np.sin(pose[2])
    ex, ey = abs(c)*hx + abs(s)*hy, abs(s)*hx + abs(c)*hy
    x, y = pose[:2] - static_map.origin[:2]
    lx, ly, ux, uy = x-ex, y-ey, x+ex, y+ey
    res = static_map.res
    if lx < 0 or ly < 0 or ux >= static_map.width*res or uy >= static_map.height*res:
        return False
    c0, c1 = math.floor(lx/res), math.floor(ux/res)+1
    r0, r1 = static_map.height-1-math.floor(uy/res), static_map.height-math.floor(ly/res)
    patch = static_map.blocked[r0:r1, c0:c1]
    if not patch.any():
        return True
    rows, cols = np.nonzero(patch)
    dx = (cols+c0+.5)*res-x
    dy = (static_map.height-rows-r0-.5)*res-y
    cell_half = res/2
    tolerance = cell_half*(abs(c)+abs(s))+1e-12
    overlap = ((abs(dx) <= ex+cell_half+1e-12) &
               (abs(dy) <= ey+cell_half+1e-12) &
               (abs(dx*c+dy*s) <= hx+tolerance) &
               (abs(-dx*s+dy*c) <= hy+tolerance))
    return not bool(overlap.any())
