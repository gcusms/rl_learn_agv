"""Versioned finite velocity choices and sensor-only masks; no RL library."""
import numpy as np
from neural_navigation.safety_context import PREDICTIVE_DIM,RECOVERY_PROBES,FAST_PROBES

COMMANDS=np.vstack([np.zeros(2),RECOVERY_PROBES,FAST_PROBES])
ACTIONS=(COMMANDS/np.array([.2,.5])).astype(np.float32)


def action_contract(predictive=False):
    value=dict(schema='masked_velocity_choices_v1',choices=COMMANDS.tolist(),
        normalized_actions=ACTIONS.tolist(),stop_index=0,
        mask_source='current_sensor_guard_probe_acceptance',wait_always_allowed=True,
        future_safety_certified=False,simulator_truth_used=False)
    if predictive:
        from neural_navigation.predictive_constraint import contract,cumulative_contract,registered_contract
        value.update(schema='masked_velocity_choices_predictive_v2',
            mask_source='current_guard_and_odometry_frame_predictive_probe_acceptance',
            predictive_constraint=(registered_contract(True) if predictive=='registered_residual' else registered_contract() if predictive=='registered' else
                cumulative_contract() if predictive=='cumulative' else contract()))
    return value


def validate_action_contract(value):
    if value not in (action_contract(),action_contract(True),action_contract('cumulative'),action_contract('registered'),action_contract('registered_residual')):
        raise ValueError('Unsupported masked action contract')
    return value!=action_contract()


def validate_hardware_action(hardware):
    value=hardware['policy_action'];predictive=validate_action_contract(value)
    actual=hardware.get('near_field_memory',{}).get('predictive_action_constraint')
    if predictive!=bool(actual) or predictive and actual!=value['predictive_constraint']:
        raise ValueError('Hardware/action predictive mask contract mismatch')
    return predictive


def masks_from_observation(observation):
    observation=np.asarray(observation)
    if observation.shape[-1:]!=(PREDICTIVE_DIM,) or not np.isfinite(observation).all():
        raise ValueError('Finite native V4 observation required')
    return np.concatenate([np.ones(observation.shape[:-1]+(1,),bool),
        observation[...,235:244]>.5,observation[...,267:269]>.5],axis=-1)
