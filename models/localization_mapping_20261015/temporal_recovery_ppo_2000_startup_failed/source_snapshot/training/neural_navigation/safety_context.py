"""Sensor-only policy context for the deployed deterministic safety transform.

Counterfactual commands are measurements of feasibility, never commands sent
to the robot. Each check clones the guard so observation has no side effects.
"""
import copy
import numpy as np


SCHEMA = 'safety_context_v2'
LEGACY_SCHEMA = 'safety_context_v1'
RECOVERY_SCHEMA = 'safety_context_v3'
PREDICTIVE_SCHEMA = 'safety_context_v4'
BOUNDED_PREDICTIVE_SCHEMA = 'safety_context_v4_bounded_pose'
PREDICTIVE_SCHEMAS = (PREDICTIVE_SCHEMA,BOUNDED_PREDICTIVE_SCHEMA)
BASE_DIM = 235
PROBES = np.array([[.2, 0.], [.1, 0.], [.05, 0.], [0., .5], [0., -.5],
                   [.1, .25], [.1, -.25], [.2, .25], [.2, -.25]], dtype=float)
# V3 keeps dimensions/order of the other context fields, but explicitly
# versions different probe semantics. V1/V2 bundles keep their exact probes.
RECOVERY_PROBES = np.array([[.2, 0.], [.1, 0.], [.05, 0.], [0., .5], [0., -.5],
                           [.05, .5], [.05, -.5], [.01, .5], [.01, -.5]], dtype=float)
REASONS = ('none', 'nonfinite_input', 'stale_or_future_scan', 'scan_unavailable',
           'odometry_jump', 'localization_uncertain', 'learned_route_incomplete',
           'reverse_without_rear_visibility', 'reverse_duration_limit',
           'static_braking_footprint', 'laser_braking_footprint',
           'escape_not_increasing_clearance', 'unsafe_turn_blocked',
           'unsafe_translation_blocked', 'estimated_goal_reached', 'other')
CONTEXT_DIM = len(PROBES) + 1 + 2 + 2 + 1 + len(REASONS) + 1
OBSERVATION_DIM = BASE_DIM + CONTEXT_DIM
PREDICTIVE_DIM = OBSERVATION_DIM + 15
FAST_PROBES = np.array([[.2, .5], [.2, -.5]], dtype=float)
REASON_START = len(PROBES) + 6


def contract(schema=SCHEMA):
    if schema not in (SCHEMA,LEGACY_SCHEMA,RECOVERY_SCHEMA)+PREDICTIVE_SCHEMAS:raise ValueError('Unsupported context schema')
    probes = RECOVERY_PROBES if schema in (RECOVERY_SCHEMA,)+PREDICTIVE_SCHEMAS else PROBES
    value=dict(schema=schema, dimensions=OBSERVATION_DIM, base_dimensions=BASE_DIM,
                probes_m_s_rad_s=probes.tolist(), reasons=list(REASONS),
                wait_normalization_s=5.)
    if schema!=LEGACY_SCHEMA:
        value['idle_wait']=dict(translation_below_m_s=.01,rotation_below_rad_s=.05,
                                include_voluntary_idle=True,exclude_reached_goal=True)
    if schema in PREDICTIVE_SCHEMAS:
        from neural_navigation.predictive_risk import PredictiveRiskContext,BoundedPoseRiskContext
        risk_type=BoundedPoseRiskContext if schema==BOUNDED_PREDICTIVE_SCHEMA else PredictiveRiskContext
        value['dimensions']=PREDICTIVE_DIM
        value['predictive_features']=dict(prefix_dimensions=OBSERVATION_DIM,
            fast_probe_acceptance_commands=FAST_PROBES.tolist(),
            risk=risk_type([1.,1.]).contract())
    return value


def validate_contract(value):
    if not isinstance(value,dict) or value != contract(value.get('schema')):
        raise ValueError('Unsupported or mismatched safety-context observation contract')


class SafetyContext:
    def __init__(self,schema=SCHEMA):
        self.schema=schema
        contract(schema)
        self.probes = (RECOVERY_PROBES if schema in (RECOVERY_SCHEMA,)+PREDICTIVE_SCHEMAS else PROBES).copy()
        self.predictive_context = None
        self.last_proposal = np.zeros(2)
        self.last_executed = np.zeros(2)
        self.last_override = False
        self.last_reason = 'none'
        self.blocked_since = None

    def observation(self, core, guard, timestamp, received_timestamp=None,
                    scan_valid=True, rear_clearance_known=False):
        now = timestamp if received_timestamp is None else received_timestamp
        if not np.isfinite([timestamp, now]).all() or abs(timestamp-core.elapsed) > 1e-8:
            raise ValueError('Context must refer to the current finite sensor frame')
        # All probes see the same sensor frame. Transform/clear the copied
        # memory once, rather than repeating that work for each command.
        # The live guard and its reverse timer remain untouched.
        snapshot = copy.deepcopy(guard)
        if snapshot.memory is not None:
            points, uncertainty = snapshot.memory.update(core)
            class PreparedMemory:
                def update(self, unused_core):
                    return points, uncertainty
                def reset(self):
                    pass  # A rejected probe cannot mutate the live memory.
            snapshot.memory = PreparedMemory()
        feasible = []
        for command in self.probes:
            executed, _ = copy.copy(snapshot).check(core, command, timestamp, now,
                                                    scan_valid, rear_clearance_known)
            feasible.append(float(np.allclose(executed, command, atol=1e-9, rtol=0)))
        held = 0. if self.blocked_since is None else np.clip((timestamp-self.blocked_since)/5., 0, 1)
        reasons = np.zeros(len(REASONS))
        reasons[REASONS.index(self.last_reason)] = 1.
        result=np.r_[feasible, float(rear_clearance_known), self.last_proposal,
                     self.last_executed, float(self.last_override), reasons, held].astype(np.float32)
        if self.schema in PREDICTIVE_SCHEMAS:
            from neural_navigation.predictive_risk import PredictiveRiskContext,BoundedPoseRiskContext
            fast=[]
            for command in FAST_PROBES:
                executed,_=copy.copy(snapshot).check(core,command,timestamp,now,scan_valid,rear_clearance_known)
                fast.append(float(np.allclose(executed,command,atol=1e-9,rtol=0)))
            if self.predictive_context is None:
                risk_type=BoundedPoseRiskContext if self.schema==BOUNDED_PREDICTIVE_SCHEMA else PredictiveRiskContext
                self.predictive_context=risk_type(guard.half*2)
            risk=self.predictive_context.features(core)
            result=np.r_[result,fast,risk].astype(np.float32)
        return result

    def augment(self, base, core, guard, timestamp, **kwargs):
        base = np.asarray(base, dtype=np.float32)
        if base.shape != (BASE_DIM,) or not np.isfinite(base).all():
            raise ValueError('Finite base navigation observation required')
        return np.r_[base, self.observation(core, guard, timestamp, **kwargs)].astype(np.float32)

    def record(self, result):
        proposal = np.asarray(result['proposed_action'], dtype=float)
        command = np.asarray(result['command'], dtype=float)
        if proposal.shape != (2,) or command.shape != (2,) or not np.isfinite(np.r_[proposal, command, result['timestamp']]).all():
            raise ValueError('Finite action-transform result required')
        self.last_proposal = np.clip(proposal, -1, 1)
        self.last_executed = np.clip([command[0]/(.2 if command[0] >= 0 else .08), command[1]/.5], -1, 1)
        self.last_override = bool(result['safety_override'])
        reason = result.get('stop_reason') or 'none'
        self.last_reason = reason if reason in REASONS else 'other'
        idle=(self.last_override and np.linalg.norm(command)<1e-8) if self.schema==LEGACY_SCHEMA else (
            abs(command[0])<.01 and abs(command[1])<.05 and self.last_reason!='estimated_goal_reached')
        if idle:
            if self.blocked_since is None:
                self.blocked_since = float(result['timestamp'])
        else:
            self.blocked_since = None
