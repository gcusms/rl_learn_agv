"""Isolated v2 feature prototype: preserve the guard's existing TTC evidence.

No runtime import or action selection. Existing model bundles cannot consume
the changed semantics until their training, runtime and manifests are updated.
The guard remains the authority for admissibility; a risk value is not a mask.
"""
import numpy as np


SCHEMA = 'long_action_risk_history_v2_guard_lower_bound'


def contract():
    return dict(schema=SCHEMA, runtime_dimensions=481, native_release_dimensions=625,
        current_frame_dimensions=14, actions=12, stop_index=0,
        horizon_s=5., score_denominator_s=5.2, short_guard_horizon_s=1.2,
        current_frame='earliest_of_original_long_forecast_and_same_frame_guard_contact',
        history_ages_s=[.4, .2, 0.], maximum_historical_sample_age_s=.05,
        supported_prediction_valid='some sensor-supported forecast is available; missing risk is not clearance',
        original_current_guard_masks_and_stop_permission_unchanged=True,
        time_aligned_sensor_track_evidence_only=True, simulator_truth_used=False,
        additional_learned_inference=False, requires_retraining_and_versioned_export=True,
        not_integrated_into_existing_models=True)


def fuse_current_frame(frame, contacts, supported_tracks, timestamp, guard_timestamp,
                       input_valid=True, scan_valid=True):
    """Return a copy; short TTCs are normalized using the long score's scale."""
    frame = np.asarray(frame, np.float32)
    contacts = np.asarray(contacts, float)
    if frame.shape != (14,) or contacts.shape != (12,):
        raise ValueError('One14D history frame and12 guard TTCs required')
    if not np.isfinite(frame).all() or np.any((frame < 0) | (frame > 1)):
        raise ValueError('Normalized finite history features required')
    if not np.isfinite([timestamp, guard_timestamp]).all() or abs(timestamp-guard_timestamp) > 1e-8:
        raise ValueError('Guard forecasts must refer to the current sensor timestamp')
    if not isinstance(supported_tracks, (int, np.integer)) or supported_tracks < 0:
        raise ValueError('Nonnegative integer supported-track count required')
    finite = np.isfinite(contacts)
    if np.any(np.isneginf(contacts)) or np.any(contacts[finite] < 0) or np.any(contacts[finite] > 1.2+1e-8):
        raise ValueError('Guard TTC is missing or lies within its1.2second horizon')
    if not input_valid or not scan_valid:
        if frame.any():
            raise ValueError('An invalid input must not contain a valid long prediction')
        return frame.copy()
    if finite.any() and not supported_tracks:
        raise ValueError('Contact evidence requires at least one supported track')
    result = frame.copy()
    if finite.any():
        lower_bound = np.zeros(12, np.float32)
        lower_bound[finite] = np.clip(1.-contacts[finite]/5.2, 0., 1.)
        result[:12] = np.maximum(result[:12], lower_bound)
        result[12] = 1.
        result[13] = max(result[13], min(supported_tracks/4., 1.))
    return result
