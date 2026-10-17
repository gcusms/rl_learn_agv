"""Opt-in confirmation of disruptive changes caused only by forecast cells."""


def contract():
    return dict(schema='confirm_forecast_detour_v1',confirmation_s=.8,
        excess_length_m=.8,excess_length_ratio=.25,behind_cosine=-.2,
        candidate_geometry_tolerance_m=.4,require_observed_route_clear=True,
        raw_scan_and_guard_unchanged=True)


def soft_forecast_contract():
    return dict(schema='forecast_soft_route_v1',forecast_cost=.02,
        measured_occupancy_is_hard=True,persistent_occupancy_is_hard=True,
        forecast_is_not_simultaneous_static_occupancy=True,
        raw_scan_and_guard_unchanged=True,local_prediction_unchanged=True)
