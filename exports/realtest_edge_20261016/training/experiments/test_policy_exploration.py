import unittest
from experiments.analyze_policy_exploration import clipped_interval_mass,analyze_inputs


class ExplorationTests(unittest.TestCase):
    def test_clipped_boundary_includes_gaussian_tail(self):
        # A strongly saturated positive mean almost always clips to +1.
        self.assertGreater(clipped_interval_mass(2.,.3,.9,1.1),.999)
        self.assertLess(clipped_interval_mass(2.,.3,-1.1,-.9),1e-20)
        self.assertAlmostEqual(clipped_interval_mass(0.,1.,-1.,1.),1.)

    def test_missing_exact_inputs_are_unknown_not_reconstructed(self):
        result=analyze_inputs(dict(case={'id':'old'},samples=[dict(timestamp=1.)]),None,{})
        self.assertFalse(result['available'])


if __name__=='__main__':unittest.main()
