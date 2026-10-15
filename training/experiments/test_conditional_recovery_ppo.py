import unittest
import torch
from experiments.conditional_recovery_ppo import actor_terms


class ConditionalGradientTests(unittest.TestCase):
    def test_forced_stops_do_not_dilute_recovery_gradient(self):
        observations=torch.zeros(3,300);observations[:,235]=1.;observations[:,266]=.2
        probability=torch.tensor([-.2,-.5,-.8],requires_grad=True)
        advantage=torch.tensor([1.,-1.,.2]);old=probability.detach()-.02
        pg,*_=actor_terms(observations,probability,old,advantage,-probability,.2)
        expected=torch.autograd.grad(pg,probability)[0]
        # Add100 states with no legal moving command. Their value targets may
        # be useful; their zero-gradient actor rows must not change this update.
        all_observations=torch.cat([observations,torch.zeros(100,300)])
        all_probability=torch.cat([probability.detach(),torch.zeros(100)]).requires_grad_()
        actual,*_=actor_terms(all_observations,all_probability,torch.cat([old,torch.zeros(100)]),
            torch.cat([advantage,torch.ones(100)*100]),-all_probability,.2)
        gradient=torch.autograd.grad(actual,all_probability)[0]
        torch.testing.assert_close(gradient[:3],expected,rtol=0,atol=0)
        self.assertEqual(float(gradient[3:].abs().sum()),0.)

    def test_only_stop_batch_is_finite_and_has_no_actor_update(self):
        logp=torch.zeros(5,requires_grad=True)
        terms=actor_terms(torch.zeros(5,300),logp,torch.zeros(5),torch.ones(5),-logp,.2)
        self.assertEqual(terms[-1],0)
        for value in terms[:-1]:self.assertTrue(torch.isfinite(value))
        terms[0].backward();self.assertEqual(float(logp.grad.abs().sum()),0.)


if __name__=='__main__':unittest.main()
