"""Prototype actor features for supported scan-track body risk.

Not wired into existing actors. No command selection, guard relaxation or
simulator truth access. Risk scores describe approximate forecasts only.
"""
import numpy as np
from neural_navigation.safety_context import RECOVERY_PROBES
from experiments.predicted_body_risk import predicted_body_risk


RISK_COMMANDS = np.vstack([np.zeros(2), RECOVERY_PROBES, [.2, .5], [.2, -.5]])
RISK_DIMENSIONS = len(RISK_COMMANDS)+1


class PredictiveRiskContext:
    def __init__(self, footprint, config=None):
        config = dict(config or {})
        self.footprint = np.asarray(footprint, float).copy()
        self.horizon = float(config.get('horizon_s', 2.4))
        self.period = .2
        self.uncertainty = float(config.get('uncertainty_m_s', .025))
        # Validate fixed geometry/options even if the first frame has no tracks.
        predicted_body_risk([], np.zeros(3), 0., RISK_COMMANDS, self.footprint,
            self.horizon, self.period, uncertainty_m_s=self.uncertainty)

    def contract(self):
        return dict(schema='predicted_body_risk_features_v1', dimensions=RISK_DIMENSIONS,
            commands_m_s_rad_s=RISK_COMMANDS.tolist(), horizon_s=self.horizon,
            period_s=self.period, margin_m=.04, uncertainty_m_s=self.uncertainty,
            score='zero_if_no_predicted_contact_else_one_minus_time_over_horizon_plus_period',
            final_feature='at_least_one_supported_moving_track',
            absence_of_support_means='unknown', instantaneous_command_following=True,
            simulator_truth_used=False, runtime_wired=False)

    def features(self, core):
        # Reject unsupported localization before reading/predicting tracks.
        localizer = core.localizer
        std = np.asarray(localizer.std, float)
        supported_pose = (bool(localizer.scan_update_accepted)
            and np.isfinite(localizer.map_agreement) and localizer.map_agreement >= .8
            and std.shape == (3,) and np.isfinite(std).all()
            and max(std[:2]) <= .2 and std[2] <= .25)
        if not supported_pose:
            return np.zeros(RISK_DIMENSIONS, dtype=np.float32)
        fusion = getattr(core, 'laser_fusion', None)
        predictor = getattr(fusion, 'predictor', None)
        tracks = predictor.tracks if predictor is not None else []
        risk = predicted_body_risk(tracks, core.estimate, core.elapsed, RISK_COMMANDS,
            self.footprint, self.horizon, self.period, uncertainty_m_s=self.uncertainty)
        scores = [0. if row['predicted_margin_contact_s'] is None else
            1.-row['predicted_margin_contact_s']/(self.horizon+self.period) for row in risk]
        return np.asarray(scores+[float(risk[0]['supported_tracks'] > 0)], dtype=np.float32)
