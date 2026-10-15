import copy
from types import SimpleNamespace
import unittest
import numpy as np
from neural_navigation.safety_context import SafetyContext, PREDICTIVE_SCHEMA
from neural_navigation.masked_actions import COMMANDS
from neural_navigation.decision_feedback import (
    DecisionFeedback, DIM, TRAIN_DIM, RUNTIME_DIM, GROUPS, FEEDBACK_REASONS,
    GROUP_DIM, REASON_SLICE, RISK_SLICE, STAT_SLICE)


class Forecast:
    def __init__(self):
        self.timestamp=None; self.contacts=np.full(12, np.inf)
        self.contacts[[0, 2]]=[.4, 1.]
        self.config=dict(horizon_s=1.2, period_s=.1)
        self.supported_tracks=1; self.tracks=[dict(time=0.)]
    def prepare(self, core): self.timestamp=core.elapsed


class Guard:
    def __init__(self, reason=None):
        self.reason=reason; self.memory=None; self.half=np.ones(2)*.2
        self.predictive_constraint=Forecast(); self.max_scan_age=.4
        self.calls=0
    def check(self, core, command, *args):
        self.calls+=1
        blocked=self.reason is not None and not np.allclose(command, 0.)
        return (np.zeros(2) if blocked else command.copy()), (self.reason if blocked else None)


class DummyRisk:
    def features(self, core): return np.zeros(13, np.float32)


def fixture(reason=None):
    core=SimpleNamespace(elapsed=0., odom=np.zeros(3), scan=np.ones(220),
        lidar=SimpleNamespace(range_min=.1, range_max=12.))
    guard=Guard(reason)
    context=SafetyContext(PREDICTIVE_SCHEMA); context.capture_decision_feedback=True
    context.predictive_context=DummyRisk()
    context.observation(core,guard,0.)
    return core,guard,context


class DecisionFeedbackTests(unittest.TestCase):
    def test_dimensions_and_unchanged_legacy_prefix(self):
        core,guard,context=fixture()
        plain=SafetyContext(PREDICTIVE_SCHEMA); plain.predictive_context=DummyRisk()
        np.testing.assert_array_equal(context.observation(core,guard,0.),plain.observation(core,guard,0.))
        self.assertIsNone(plain.guard_probe_frame)
        feedback=DecisionFeedback(); prefix=np.linspace(-1,1,300,dtype=np.float32)
        obs=feedback.augment(prefix,core,guard,context)
        self.assertEqual(TRAIN_DIM,436); self.assertEqual(RUNTIME_DIM,439)
        np.testing.assert_array_equal(obs[:300],prefix)
        self.assertEqual(guard.calls,0)  # Probes mutate copies, not the live guard.

    def test_all_current_command_explanations_match_exact_masks(self):
        core,guard,context=fixture('predicted_dynamic_contact')
        feedback=DecisionFeedback(); x=feedback.features(core,guard,context)
        groups=x[:GROUP_DIM].reshape(12,len(GROUPS))
        self.assertTrue((groups[1:,GROUPS.index('dynamic_prediction')]==1).all())
        self.assertEqual(groups[0,GROUPS.index('accepted')],1)
        self.assertEqual(feedback.frame_mask.sum(),1)

    def test_stop_risk_uses_the_same_current_guard_forecast(self):
        core,guard,context=fixture(); feedback=DecisionFeedback()
        x=feedback.features(core,guard,context)
        self.assertAlmostEqual(x[RISK_SLICE][0],1-.4/1.3,places=6)
        self.assertAlmostEqual(x[RISK_SLICE][2],1-1./1.3,places=6)
        self.assertEqual(x[RISK_SLICE][1],0.)
        self.assertEqual(x[STAT_SLICE][4],1)
        guard.predictive_constraint.contacts[0]=.1
        self.assertGreater(feedback.features(core,guard,context)[RISK_SLICE][0],x[RISK_SLICE][0])

    def test_exact_prior_executed_reason_not_other_or_current_leak(self):
        core,guard,context=fixture(); feedback=DecisionFeedback()
        first=feedback.features(core,guard,context)
        self.assertEqual(first[REASON_SLICE][FEEDBACK_REASONS.index('none')],1)
        feedback.record(dict(timestamp=0.,command=[0.,0.],stop_reason='predicted_dynamic_contact'))
        np.testing.assert_array_equal(first[REASON_SLICE],feedback.last_features[REASON_SLICE])
        core.elapsed=.2; context.observation(core,guard,.2)
        second=feedback.features(core,guard,context)
        self.assertEqual(second[REASON_SLICE][FEEDBACK_REASONS.index('predicted_dynamic_contact')],1)
        self.assertEqual(second[REASON_SLICE][FEEDBACK_REASONS.index('other')],0)

    def test_policy_wait_and_forced_only_stop_are_distinct(self):
        for reason,expected in [(None,'policy_wait'),('laser_braking_footprint','only_stop_available')]:
            core,guard,context=fixture(reason); f=DecisionFeedback(); f.features(core,guard,context)
            f.record(dict(timestamp=0.,command=[0.,0.],stop_reason=None))
            self.assertEqual(f.previous_reason,expected)
            core.elapsed=.4; context.observation(core,guard,.4)
            x=f.features(core,guard,context)
            self.assertAlmostEqual(x[STAT_SLICE][2],.4/5,places=6)
            self.assertEqual(x[STAT_SLICE][3]>0,expected=='policy_wait')

    def test_unknown_future_reason_is_explicitly_other(self):
        core,guard,context=fixture(); f=DecisionFeedback(); f.features(core,guard,context)
        f.record(dict(timestamp=0.,command=[0.,0.],stop_reason='future_reason'))
        self.assertEqual(f.previous_reason,'other')

    def test_blank_stale_and_invalid_motion_are_not_clear_prediction(self):
        for kind in ('blank','stale','odometry_jump','invalid'):
            core,guard,context=fixture()
            if kind=='blank':core.scan[:]=np.nan
            if kind=='odometry_jump':core.odometry_sanity_failure=True
            if kind=='stale':context.guard_probe_frame['received_timestamp']=.5
            if kind=='invalid':context.guard_probe_frame['scan_valid']=False
            self.assertEqual(DecisionFeedback().features(core,guard,context)[STAT_SLICE][4],0.)

    def test_reset_and_goal_stop_clear_wait_history(self):
        core,guard,context=fixture(); f=DecisionFeedback(); f.features(core,guard,context)
        f.record(dict(timestamp=0.,command=[0.,0.],stop_reason='estimated_goal_reached'))
        self.assertIsNone(f.idle_since); self.assertIsNone(f.idle_with_choices_since)
        f.reset(); self.assertEqual(f.previous_reason,'none'); self.assertIsNone(f.frame_mask)

    def test_stale_or_wrong_action_order_is_rejected(self):
        core,guard,context=fixture(); f=DecisionFeedback()
        core.elapsed=.2
        with self.assertRaises(ValueError):f.features(core,guard,context)
        core.elapsed=0.; context.guard_probe_frame['commands']=COMMANDS[::-1]
        with self.assertRaises(ValueError):f.features(core,guard,context)

    def test_duplicate_execution_cannot_extend_history(self):
        core,guard,context=fixture(); f=DecisionFeedback(); f.features(core,guard,context)
        result=dict(timestamp=0.,command=[0.,0.],stop_reason=None); f.record(result)
        with self.assertRaises(ValueError):f.record(result)


if __name__=='__main__':unittest.main()
