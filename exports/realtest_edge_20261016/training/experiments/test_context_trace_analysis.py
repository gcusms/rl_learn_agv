import unittest
import json
from experiments.analyze_context_traces import analyze


class TraceAnalysisTests(unittest.TestCase):
    def sample(self, time, command, flags, reason='learned_route_incomplete'):
        return dict(timestamp=time, command=command, model_command=[.1,.003],
                    stop_reason=reason, safety_context=dict(action_feasibility=flags,
                    policy_previous_stop_reason='learned_route_incomplete'))

    def test_small_rotation_is_not_mistaken_for_effective_escape(self):
        only_turn=[0,0,0,1,1,0,0,0,0]
        rows=[self.sample(i*.2,[0.,.003],only_turn) for i in range(5)]
        trace=dict(case={'id':'synthetic'},samples=rows,result={})
        result=analyze(trace)
        self.assertEqual(result['classifications']['idle_with_only_turn_probe']['steps'],5)
        self.assertAlmostEqual(result['classifications']['idle_with_only_turn_probe']['longest_seconds'],1.)
        self.assertEqual(result['classifications']['stopped_with_forward_probe']['steps'],0)
        self.assertEqual(result['partial_projection_steps'],5)
        json.dumps(result,allow_nan=False)

    def test_missing_context_and_reached_goal_are_not_counted_as_failure(self):
        row=self.sample(0.,[0.,0.],[1]*9,'estimated_goal_reached')
        result=analyze(dict(case={'id':'goal'},samples=[row],result={}))
        self.assertEqual(result['classifications']['idle_with_forward_probe']['steps'],0)
        row['safety_context']={}
        result=analyze(dict(case={'id':'legacy'},samples=[row],result={}))
        self.assertIsNone(result['classifications'])

    def test_v4_fast_only_alternative_is_not_reported_as_no_action(self):
        row = self.sample(0., [0., 0.], [0]*9)
        row['policy_observation'] = [0.]*282
        row['policy_observation'][267] = 1.
        result = analyze(dict(case={'id':'v4'}, samples=[row], result={}))
        self.assertEqual(result['classifications']['no_verified_probe']['steps'], 0)
        self.assertEqual(result['classifications']['idle_with_forward_probe']['steps'], 1)
        self.assertEqual(result['context_counters']['fast_only_forward_probe_steps'], 1)

    def test_repeated_rejected_reverse_is_visible_even_while_rotating(self):
        rows=[self.sample(i*.2,[0.,-.5],[1]*9,'reverse_without_rear_visibility')
              for i in range(8)]
        for row in rows: row['model_command']=[-.02,-.5]
        result=analyze(dict(case={'id':'reverse'},samples=rows,result={}))
        metric=result['classifications']['rejected_reverse_with_forward_probe']
        self.assertEqual(metric['steps'],8)
        self.assertAlmostEqual(metric['longest_seconds'],1.6)
        self.assertEqual(result['classifications']['idle_with_forward_probe']['steps'],0)
        for row in rows: row['safety_context']['action_feasibility']=[0]*9
        result=analyze(dict(case={'id':'no_forward'},samples=rows,result={}))
        self.assertEqual(result['classifications']['rejected_reverse_with_forward_probe']['steps'],0)


if __name__=='__main__':unittest.main()
