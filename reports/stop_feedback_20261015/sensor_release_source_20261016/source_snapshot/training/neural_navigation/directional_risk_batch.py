"""Same finite-response forecast, batching commands/time and bounding scratch RAM."""
import numpy as np
from navrl.geometry import integrate


def batched_directional_body_risk(tracks, pose, timestamp, commands, footprint,
        horizon_s=2.4, period_s=.2, margin_m=.04, uncertainty_floor_m_s=.025,
        residual_scale=2., measured_velocity=None, response_gain=20.,
        response_acceleration=(.5, 1.)):
    tracks = tuple(tracks)
    pose = np.asarray(pose, float)
    commands = np.asarray(commands, float)
    half = np.asarray(footprint, float) / 2
    if pose.shape != (3,) or commands.ndim != 2 or commands.shape[1] != 2 or half.shape != (2,):
        raise ValueError('Pose, commands and footprint dimensions required')
    if (not np.isfinite(np.r_[pose, commands.ravel(), half, timestamp, horizon_s,
            period_s, margin_m, uncertainty_floor_m_s, residual_scale]).all()
            or min(half) <= 0 or min(horizon_s, period_s, residual_scale) <= 0
            or min(margin_m, uncertainty_floor_m_s) < 0):
        raise ValueError('Finite positive geometry and valid uncertainty required')
    supported = []
    for track in tracks:
        points = np.asarray(track['points'], float)
        velocity = np.asarray(track['velocity'], float)
        age = timestamp - float(track['time'])
        second = np.asarray(track.get('velocity_error_second_moment', np.zeros((2, 2))), float)
        if (velocity.shape != (2,) or points.ndim != 2 or points.shape[1] != 2
                or second.shape != (2, 2)
                or not np.isfinite(np.r_[points.ravel(), velocity, second.ravel(), age]).all()):
            raise ValueError('Finite track points and residual second moment required')
        if not np.allclose(second, second.T, atol=1e-12, rtol=0) or np.linalg.eigvalsh(second)[0] < -1e-12:
            raise ValueError('Symmetric positive semidefinite second moment required')
        if 0 <= age <= .8 and track['hits'] >= 3 and np.linalg.norm(velocity) >= .04 and len(points):
            supported.append((points, velocity, age, second))
    times = np.arange(0., horizon_s + 1e-9, period_s)
    if times[-1] < horizon_s - 1e-9:
        times = np.r_[times, horizon_s]
    if measured_velocity is not None:
        times = np.linspace(0., horizon_s, int(np.ceil(horizon_s / period_s)) + 1)
    if not len(commands):
        return []
    first_trajectory = None
    if measured_velocity is not None:
        from neural_navigation.motion_response import force_response_trajectory
        # Preserve the response validation even if no eligible tracks exist.
        first_trajectory = force_response_trajectory(measured_velocity, commands[0], horizon_s,
            gain=response_gain, acceleration=response_acceleration, samples=len(times))
    if not supported:
        return [dict(command=c.tolist(), supported_tracks=0,
            predicted_margin_contact_s=None, minimum_predicted_clearance_m=None) for c in commands]
    if measured_velocity is None:
        bodies = np.asarray([[integrate(pose, *c, float(t)) for t in times] for c in commands])
    else:
        local = np.stack([first_trajectory] + [force_response_trajectory(measured_velocity, c,
            horizon_s, gain=response_gain, acceleration=response_acceleration, samples=len(times))
            for c in commands[1:]])
        c, s = np.cos(pose[2]), np.sin(pose[2])
        bodies = np.empty_like(local)
        bodies[:, :, :2] = pose[:2] + local[:, :, :2] @ np.array([[c, s], [-s, c]])
        bodies[:, :, 2] = pose[2] + local[:, :, 2]
    c, s = np.cos(bodies[:, :, 2]), np.sin(bodies[:, :, 2])
    rotation = np.empty((*c.shape, 2, 2))
    rotation[:, :, 0, 0] = c
    rotation[:, :, 0, 1] = -s
    rotation[:, :, 1, 0] = s
    rotation[:, :, 1, 1] = c
    clearances = np.full((len(commands), len(times)), np.inf)
    coordinate_scale = max(1., float(np.max(np.abs(bodies[:, :, :2]))))
    for points, velocity, age, second in supported:
        elapsed = times + age
        coordinate_scale = max(coordinate_scale,
            float(np.max(np.abs(points))), float(np.max(np.abs(velocity))) * (age + horizon_s))
        # A long retained contour must not create an unbounded C*T*N tensor.
        for offset in range(0, len(points), 256):
            future = points[offset:offset + 256][None, :, :] + velocity * elapsed[:, None, None]
            relative = (future[None, :, :, :] - bodies[:, :, None, :2]) @ rotation
            delta = relative - np.clip(relative, -half, half)
            gap = np.linalg.norm(delta, axis=-1)
            normals = np.divide(delta, gap[:, :, :, None], out=np.zeros_like(delta),
                where=gap[:, :, :, None] > 1e-12) @ rotation.swapaxes(-1, -2)
            projected = np.maximum(0., np.einsum('ctni,ij,ctnj->ctn', normals, second, normals))
            spread = np.maximum(uncertainty_floor_m_s, residual_scale * np.sqrt(projected))
            clearance = (gap - margin_m - spread * elapsed[None, :, None]).min(axis=2)
            clearances = np.minimum(clearances, clearance)
    # BLAS/vector arithmetic can round differently near the contact threshold.
    # Keep the original scalar decision at such boundaries (also at nonfinite
    # intermediate values), without weakening any contact/margin condition.
    tolerance = np.finfo(float).eps * 1024 * coordinate_scale
    if not np.isfinite(clearances).all() or np.any(np.abs(clearances - 1e-9) <= tolerance):
        from neural_navigation.directional_risk import directional_body_risk_reference
        return directional_body_risk_reference(tracks, pose, timestamp, commands, footprint,
            horizon_s, period_s, margin_m, uncertainty_floor_m_s, residual_scale,
            measured_velocity, response_gain, response_acceleration)
    result = []
    for index, command in enumerate(commands):
        hit = np.flatnonzero(clearances[index] <= 1e-9)
        result.append(dict(command=command.tolist(), supported_tracks=len(supported),
            predicted_margin_contact_s=float(times[hit[0]]) if len(hit) else None,
            minimum_predicted_clearance_m=float(clearances[index].min())))
    return result
