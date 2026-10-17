import unittest
import torch
from experiments.greedy_stable_quality_ppo import rank_violation


class GreedyProtection(unittest.TestCase):
    def test_near_tie_flip_with_small_kl_is_rejected(self):
        p=torch.tensor([[.5116,.4514]+[.0037]*10]);p=p/p.sum(dim=1,keepdim=True)
        q=torch.tensor([[.4519,.5094]+[.00387]*10]);q=q/q.sum(dim=1,keepdim=True)
        kl=(p*(p/q).log()).sum()
        self.assertLess(float(kl),.02)
        self.assertGreater(float(rank_violation(p,q)),0.)
        self.assertEqual(float(rank_violation(p,p)),0.)

    def test_winner_has_no_hardcoded_command_and_violation_has_gradient(self):
        logits=torch.zeros(1,12,requires_grad=True)
        p=torch.full((1,12),.01);p[0,9]=.89
        violation=rank_violation(p,logits.softmax(dim=1)).sum()
        violation.backward()
        self.assertLess(float(logits.grad[0,9]),0.)
        self.assertGreater(float(logits.grad[0,0]),0.)


if __name__=='__main__':unittest.main()
