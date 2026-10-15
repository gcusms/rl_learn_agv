"""Opt-in confirmation of disruptive changes caused only by forecast cells."""


def contract():
    return dict(schema='confirm_forecast_detour_v1',confirmation_s=.8,
        excess_length_m=.8,excess_length_ratio=.25,behind_cosine=-.2,
        candidate_geometry_tolerance_m=.4,require_observed_route_clear=True,
        raw_scan_and_guard_unchanged=True)
