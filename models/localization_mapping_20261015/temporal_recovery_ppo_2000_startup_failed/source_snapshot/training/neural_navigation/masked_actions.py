"""Versioned finite velocity choices and sensor-only masks; no RL library."""
import numpy as np
from neural_navigation.safety_context import PREDICTIVE_DIM,RECOVERY_PROBES,FAST_PROBES

COMMANDS=np.vstack([np.zeros(2),RECOVERY_PROBES,FAST_PROBES])
ACTIONS=(COMMANDS/np.array([.2,.5])).astype(np.float32)


def action_contract():
    return dict(schema='masked_velocity_choices_v1',choices=COMMANDS.tolist(),
        normalized_actions=ACTIONS.tolist(),stop_index=0,
        mask_source='current_sensor_guard_probe_acceptance',wait_always_allowed=True,
        future_safety_certified=False,simulator_truth_used=False)


def masks_from_observation(observation):
    observation=np.asarray(observation)
    if observation.shape[-1:]!=(PREDICTIVE_DIM,) or not np.isfinite(observation).all():
        raise ValueError('Finite native V4 observation required')
    return np.concatenate([np.ones(observation.shape[:-1]+(1,),bool),
        observation[...,235:244]>.5,observation[...,267:269]>.5],axis=-1)
