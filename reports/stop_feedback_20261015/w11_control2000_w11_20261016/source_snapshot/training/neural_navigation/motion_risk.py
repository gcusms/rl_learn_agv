"""Sensor-track forecasts against a rectangular body; diagnostic only.

Constant-velocity point forecasts and instantaneous command following are
approximations. No simulator actor state is accepted and no action is issued.
"""
import numpy as np
from navrl.geometry import integrate


def predicted_body_risk(tracks, pose, timestamp, commands, footprint,
                        horizon_s=2.4, period_s=.1, margin_m=.04,
                        uncertainty_m_s=.025,measured_velocity=None,
                        response_gain=20.,response_acceleration=(.5,1.),
                        adaptive_track_uncertainty=False):
    pose = np.asarray(pose, float); commands = np.asarray(commands, float)
    half = np.asarray(footprint, float)/2
    if pose.shape != (3,) or commands.ndim != 2 or commands.shape[1] != 2 or half.shape != (2,):
        raise ValueError('Pose, two-dimensional commands and rectangular footprint required')
    if not np.isfinite(np.r_[pose, commands.ravel(), half, timestamp, horizon_s, period_s, margin_m, uncertainty_m_s]).all() or min(half) <= 0 or min(horizon_s, period_s) <= 0 or min(margin_m, uncertainty_m_s) < 0:
        raise ValueError('Finite positive geometry and prediction periods required')
    times = np.arange(0., horizon_s+1e-9, period_s)
    if times[-1] < horizon_s-1e-9: times = np.r_[times, horizon_s]
    if measured_velocity is not None:
        from neural_navigation.motion_response import force_response_trajectory
        # Finite-response forecasts use a uniform grid with maximum spacing
        # period_s. Inputs are measured body velocity, never actor truth.
        times=np.linspace(0.,horizon_s,int(np.ceil(horizon_s/period_s))+1)
        trajectories=[force_response_trajectory(measured_velocity,command,horizon_s,
            gain=response_gain,acceleration=response_acceleration,samples=len(times))
            for command in commands]
    supported = []
    for track in tracks:
        age = timestamp-float(track['time'])
        velocity = np.asarray(track['velocity'], float)
        points = np.asarray(track['points'], float)
        if velocity.shape != (2,) or points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(np.r_[velocity, points.ravel(), age]).all():
            raise ValueError('Finite observed map-frame track points required')
        if 0 <= age <= .8 and track['hits'] >= 3 and np.linalg.norm(velocity) >= .04 and len(points):
            uncertainty=uncertainty_m_s
            if adaptive_track_uncertainty:
                measured=float(track.get('velocity_uncertainty_m_s',uncertainty_m_s))
                if not np.isfinite(measured) or measured<0:
                    raise ValueError('Finite nonnegative track uncertainty required')
                uncertainty=max(uncertainty,measured)
            supported.append((points, velocity, age,uncertainty))
    result = []
    for index,command in enumerate(commands):
        first = None; minimum = None
        for sample_index,future in enumerate(times):
            if measured_velocity is None:
                body = integrate(pose, *command, float(future))
            else:
                local=trajectories[index][sample_index];c,s=np.cos(pose[2]),np.sin(pose[2])
                body=np.r_[pose[:2]+local[:2]@np.array([[c,s],[-s,c]]),pose[2]+local[2]]
            c, s = np.cos(body[2]), np.sin(body[2])
            rotation = np.array([[c, -s], [s, c]])
            for points, velocity, age,uncertainty in supported:
                relative = (points+velocity*(age+future)-body[:2])@rotation
                gap = np.linalg.norm(relative-np.clip(relative, -half, half), axis=1)
                clearance = float(np.min(gap)-margin_m-uncertainty*(age+future))
                minimum = clearance if minimum is None else min(minimum, clearance)
                if clearance <= 1e-9 and first is None: first = float(future)
        result.append(dict(command=command.tolist(), supported_tracks=len(supported),
            predicted_margin_contact_s=first, minimum_predicted_clearance_m=minimum))
    return result
