"""Append measured scan quality; never equate a missing return with free space."""
import numpy as np

from neural_navigation.long_risk_observation import contract as risk_contract


RUNTIME_DIM = 625
TRAIN_DIM = 622
SECTORS = 72
SCHEMA = 'risk_and_measured_release_context_v1'


def contract():
    return dict(schema=SCHEMA, dimensions=RUNTIME_DIM, training_dimensions=TRAIN_DIM,
        unchanged_runtime_prefix_dimensions=481, parent_contract=risk_contract(),
        append_dimensions=144, sectors=SECTORS, pooling='numpy.array_split_in_scan_order',
        fields=['72_finite_hit_fractions', '72_no_return_fractions'],
        finite_hit='finite and range_min <= range < range_max',
        no_return='positive infinity or finite range == range_max',
        unknown='NaN, negative infinity, below range_min or finite above range_max',
        unknown_fraction='1 - finite_hit_fraction - no_return_fraction',
        no_return_is_not_a_measured_surface=True, unknown_is_not_clear=True,
        source='same current raw scan consumed by localization, routing and guard',
        simulator_truth_used=False, future_sensor_frames_used=False,
        action_masks_body_margin_blind_memory_and_deadlines_unchanged=True,
        commands_issued=False)


def scan_quality(scan, lidar):
    scan = np.asarray(scan, float)
    if scan.shape != (lidar.beams,) or lidar.beams < SECTORS:
        raise ValueError('Scan must match the declared LiDAR geometry')
    hit = np.isfinite(scan) & (scan >= lidar.range_min) & (scan < lidar.range_max)
    no_return = np.isposinf(scan) | (np.isfinite(scan) & (scan == lidar.range_max))
    return np.r_[[part.mean() for part in np.array_split(hit, SECTORS)],
        [part.mean() for part in np.array_split(no_return, SECTORS)]].astype(np.float32)


def augment(observation, scan, lidar):
    observation = np.asarray(observation, np.float32)
    if observation.shape != (481,) or not np.isfinite(observation).all():
        raise ValueError('Finite unchanged 481-dimensional risk prefix required')
    return np.r_[observation, scan_quality(scan, lidar)].astype(np.float32)


def to_training(observations):
    observations = np.asarray(observations, np.float32)
    if observations.ndim != 2 or observations.shape[1] != RUNTIME_DIM or not np.isfinite(observations).all():
        raise ValueError('Finite release observation matrix required')
    return np.concatenate([observations[:, :300], observations[:, 303:]], axis=1)
