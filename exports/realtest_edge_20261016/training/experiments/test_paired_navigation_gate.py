import copy
import unittest
from experiments.paired_navigation_gate import compare


def row(key,success=True,path=4.,seconds=30.):
    return dict(case_id=key,safe_success=success,collisions=int(not success),false_arrival=False,travel_m=path,seconds=seconds)


class PairedGateTests(unittest.TestCase):
    def test_average_cannot_hide_a_new_failure(self):
        old=[row('a'),row('b',False)];new=[row('a',False),row('b')]
        result=compare(old,new)
        self.assertFalse(result['safety_nonregression']);self.assertEqual(result['lost_successes'],['a'])

    def test_same_success_count_does_not_hide_inefficient_navigation(self):
        old=[row('a'),row('b')];new=copy.deepcopy(old);new[0]['travel_m']*=1.3
        result=compare(old,new)
        self.assertFalse(result['efficiency_nonregression']);self.assertFalse(result['larger_training_gate_passed'])

    def test_a_new_success_is_not_compared_with_a_short_collision_trajectory(self):
        old=[row('a'),row('b',False,path=.2,seconds=1.)];new=[row('a'),row('b',path=10.,seconds=90.)]
        result=compare(old,new)
        self.assertEqual(result['common_successes'],1);self.assertTrue(result['larger_training_gate_passed'])
        self.assertEqual(result['gained_successes'],['b'])

    def test_unpaired_or_duplicate_or_invalid_metrics_are_rejected(self):
        for old,new in [([row('a')],[row('b')]),([row('a'),row('a')],[row('a')]),
            ([row('a')],[row('a',path=float('nan'))])]:
            with self.assertRaises(ValueError):compare(old,new)


if __name__=='__main__':unittest.main()
