"""Sensor-track forecasts against a rectangular body; diagnostic only.

Constant-velocity point forecasts and instantaneous command following are
approximations. No simulator actor state is accepted and no action is issued.
"""
import numpy as np
from navrl.geometry import integrate


def predicted_body_risk(tracks, pose, timestamp, commands, footprint,
                        horizon_s=2.4, period_s=.1, margin_m=.04,
                        uncertainty_m_s=.025):
    pose = np.asarray(pose, float); commands = np.asarray(commands, float)
    half = np.asarray(footprint, float)/2
    if pose.shape != (3,) or commands.ndim != 2 or commands.shape[1] != 2 or half.shape != (2,):
        raise ValueError('Pose, two-dimensional commands and rectangular footprint required')
    if not np.isfinite(np.r_[pose, commands.ravel(), half, timestamp, horizon_s, period_s, margin_m, uncertainty_m_s]).all() or min(half) <= 0 or min(horizon_s, period_s) <= 0 or min(margin_m, uncertainty_m_s) < 0:
        raise ValueError('Finite positive geometry and prediction periods required')
    times = np.arange(0., horizon_s+1e-9, period_s)
    if times[-1] < horizon_s-1e-9: times = np.r_[times, horizon_s]
    supported = []
    for track in tracks:
        age = timestamp-float(track['time'])
        velocity = np.asarray(track['velocity'], float)
        points = np.asarray(track['points'], float)
        if velocity.shape != (2,) or points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(np.r_[velocity, points.ravel(), age]).all():
            raise ValueError('Finite observed map-frame track points required')
        if 0 <= age <= .8 and track['hits'] >= 3 and np.linalg.norm(velocity) >= .04 and len(points):
            supported.append((points, velocity, age))
    result = []
    for command in commands:
        first = None; minimum = None
        for future in times:
            body = integrate(pose, *command, float(future))
            c, s = np.cos(body[2]), np.sin(body[2])
            rotation = np.array([[c, -s], [s, c]])
            for points, velocity, age in supported:
                relative = (points+velocity*(age+future)-body[:2])@rotation
                gap = np.linalg.norm(relative-np.clip(relative, -half, half), axis=1)
                clearance = float(np.min(gap)-margin_m-uncertainty_m_s*(age+future))
                minimum = clearance if minimum is None else min(minimum, clearance)
                if clearance <= 1e-9 and first is None: first = float(future)
        result.append(dict(command=command.tolist(), supported_tracks=len(supported),
            predicted_margin_contact_s=first, minimum_predicted_clearance_m=minimum))
    return result
