import unittest
import torch
from experiments.recovery_temperature_policy import RecoveryDistribution


class RecoveryTemperatureTests(unittest.TestCase):
    def test_two_choice_exploration_matches_log_probability_and_mask(self):
        logits=torch.full((1,12),-20.);logits[0,0]=0.;logits[0,4]=-6.33;logits.requires_grad_()
        distribution=RecoveryDistribution(12).proba_distribution(logits)
        mask=torch.zeros(1,12,dtype=torch.bool);mask[:,[0,4]]=True
        distribution.apply_masking(mask)
        probability=distribution.distribution.probs[0,4]
        self.assertGreater(float(probability.detach()),.1)
        self.assertLess(float(probability.detach()),.12)
        self.assertEqual(int(distribution.mode()),0)
        self.assertEqual(float(distribution.distribution.probs[~mask].sum()),0.)
        torch.testing.assert_close(distribution.log_prob(torch.tensor([4])),probability.log()[None])
        distribution.log_prob(torch.tensor([4])).sum().backward()
        self.assertGreater(float(logits.grad[0,4]),0.)
        torch.manual_seed(81)
        choices=distribution.distribution.sample((1000,))
        self.assertGreater(int((choices==4).sum()),50)
        self.assertTrue(bool(((choices==0)|(choices==4)).all()))

    def test_open_states_and_repeated_masks_preserve_distribution(self):
        logits=torch.arange(12,dtype=torch.float32)[None]
        distribution=RecoveryDistribution(12).proba_distribution(logits)
        full=torch.ones(1,12,dtype=torch.bool);distribution.apply_masking(full)
        torch.testing.assert_close(distribution.distribution.probs,logits.softmax(-1))
        small=full.clone();small[:,2:]=False;distribution.apply_masking(small)
        first=distribution.distribution.probs.clone();distribution.apply_masking(small)
        torch.testing.assert_close(distribution.distribution.probs,first)
        distribution.apply_masking(None)
        torch.testing.assert_close(distribution.distribution.probs,logits.softmax(-1))
        with self.assertRaises(ValueError):distribution.apply_masking(torch.zeros_like(full))


if __name__=='__main__':unittest.main()
