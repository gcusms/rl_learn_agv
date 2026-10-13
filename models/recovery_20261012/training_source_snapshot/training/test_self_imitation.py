import unittest
import torch
from experiments.self_imitation import losses


class Policy:
    def __init__(self):
        self.logits=torch.tensor([[0.,0.],[0.,0.]],requires_grad=True)
        self.values=torch.tensor([1.,3.],requires_grad=True)
    def evaluate_actions(self,obs,actions,action_masks):
        dist=torch.distributions.Categorical(logits=self.logits)
        return self.values,dist.log_prob(actions),dist.entropy()


class SelfImitationTests(unittest.TestCase):
    def test_actor_uses_detached_positive_advantage(self):
        p=Policy();obs=torch.zeros((2,300));actions=torch.tensor([0,1]);masks=torch.ones((2,12),dtype=torch.bool)
        actor,critic,stats=losses(p,obs,actions,masks,torch.tensor([3.,2.]))
        actor.backward();self.assertIsNone(p.values.grad)
        self.assertLess(p.logits.grad[0,0].item(),0);self.assertEqual(p.logits.grad[1].abs().sum().item(),0)
        self.assertEqual(stats['positive_advantage_states'],1)
        critic.backward();torch.testing.assert_close(p.values.grad,torch.tensor([-1.,0.]))

    def test_no_advantage_produces_no_learning(self):
        p=Policy();a,b,s=losses(p,torch.zeros((2,300)),torch.tensor([0,1]),torch.ones((2,12),dtype=torch.bool),torch.tensor([0.,1.]))
        self.assertEqual(a.item(),0);self.assertEqual(b.item(),0);self.assertEqual(s['positive_advantage_states'],0)

    def test_masked_demonstration_is_rejected(self):
        masks=torch.ones((2,12),dtype=torch.bool);masks[0,0]=False
        with self.assertRaises(ValueError):
            losses(Policy(),torch.zeros((2,300)),torch.tensor([0,1]),masks,torch.tensor([3.,2.]))


if __name__=='__main__':unittest.main()
