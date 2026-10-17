"""Batch the existing sensor-track body forecast without choosing an action.

This computes the same finite-response, constant-track-velocity approximation
as motion_risk.predicted_body_risk. It accepts no simulator actor or future
state. Forecast validity and model admission must be handled by callers.
"""
import numpy as np

from navrl.geometry import integrate
from neural_navigation.motion_response import force_response_trajectory


def batch_predicted_body_risk(tracks, pose, timestamp, commands, footprint,
        horizon_s=5., period_s=.2, margin_m=.04, uncertainty_m_s=.025,
        measured_velocity=None, response_gain=20., response_acceleration=(.5, 1.),
        adaptive_track_uncertainty=True):
    pose = np.asarray(pose, float)
    commands = np.asarray(commands, float)
    half = np.asarray(footprint, float) / 2
    if pose.shape != (3,) or commands.ndim != 2 or commands.shape[1] != 2 or half.shape != (2,):
        raise ValueError('Pose, two-dimensional commands and rectangular footprint required')
    if (not np.isfinite(np.r_[pose, commands.ravel(), half, timestamp, horizon_s,
            period_s, margin_m, uncertainty_m_s]).all() or min(half) <= 0
            or min(horizon_s, period_s) <= 0 or min(margin_m, uncertainty_m_s) < 0):
        raise ValueError('Finite positive geometry and prediction periods required')
    times = np.arange(0., horizon_s + 1e-9, period_s)
    if times[-1] < horizon_s - 1e-9:
        times = np.r_[times, horizon_s]
    if measured_velocity is not None:
        times = np.linspace(0., horizon_s, int(np.ceil(horizon_s / period_s)) + 1)
    points, velocities, ages, uncertainties = [], [], [], []
    supported = 0
    for track in tracks:
        age = timestamp - float(track['time'])
        velocity = np.asarray(track['velocity'], float)
        observed = np.asarray(track['points'], float)
        if (velocity.shape != (2,) or observed.ndim != 2 or observed.shape[1] != 2
                or not np.isfinite(np.r_[velocity, observed.ravel(), age]).all()):
            raise ValueError('Finite observed odometry-frame tracks required')
        if not (0 <= age <= .8 and track['hits'] >= 3 and np.linalg.norm(velocity) >= .04 and len(observed)):
            continue
        uncertainty = uncertainty_m_s
        if adaptive_track_uncertainty:
            measured = float(track.get('velocity_uncertainty_m_s', uncertainty_m_s))
            if not np.isfinite(measured) or measured < 0:
                raise ValueError('Finite nonnegative track uncertainty required')
            uncertainty = max(uncertainty, measured)
        points.append(observed)
        velocities.append(np.broadcast_to(velocity, observed.shape))
        ages.append(np.full(len(observed), age))
        uncertainties.append(np.full(len(observed), uncertainty))
        supported += 1
    if not len(commands) or not supported:
        return [dict(command=command.tolist(), supported_tracks=supported,
            predicted_margin_contact_s=None, minimum_predicted_clearance_m=None) for command in commands]
    if measured_velocity is None:
        bodies = np.asarray([[integrate(pose, *command, float(future)) for future in times]
            for command in commands])
    else:
        local = np.stack([force_response_trajectory(measured_velocity, command, horizon_s,
            gain=response_gain, acceleration=response_acceleration, samples=len(times)) for command in commands])
        c, s = np.cos(pose[2]), np.sin(pose[2])
        bodies = np.empty_like(local)
        bodies[:, :, 0] = pose[0] + local[:, :, 0] * c - local[:, :, 1] * s
        bodies[:, :, 1] = pose[1] + local[:, :, 0] * s + local[:, :, 1] * c
        bodies[:, :, 2] = pose[2] + local[:, :, 2]
    observed = np.concatenate(points)
    velocity = np.concatenate(velocities)
    age = np.concatenate(ages)
    uncertainty = np.concatenate(uncertainties)
    elapsed = times[:, None] + age[None, :]
    future_points = observed[None, :, :] + velocity[None, :, :] * elapsed[:, :, None]
    relative = future_points[None, :, :, :] - bodies[:, :, None, :2]
    c, s = np.cos(bodies[:, :, 2])[:, :, None], np.sin(bodies[:, :, 2])[:, :, None]
    local = np.stack([relative[:, :, :, 0] * c + relative[:, :, :, 1] * s,
        -relative[:, :, :, 0] * s + relative[:, :, :, 1] * c], axis=-1)
    outside = local - np.clip(local, -half, half)
    gap = np.linalg.norm(outside, axis=-1)
    clearance = (gap - margin_m - uncertainty[None, None, :] * elapsed[None, :, :]).min(axis=2)
    result = []
    for index, command in enumerate(commands):
        hit = np.flatnonzero(clearance[index] <= 1e-9)
        result.append(dict(command=command.tolist(), supported_tracks=supported,
            predicted_margin_contact_s=float(times[hit[0]]) if len(hit) else None,
            minimum_predicted_clearance_m=float(clearance[index].min())))
    return result
