"""Behavioral tests for accepted, shortened and fully rejected PPO updates."""
import copy
import unittest

import numpy as np
import torch

from neural_navigation.masked_actions import ACTIONS
from retained_stop_ppo import RetainedStopOnPolicy


class TinyNativeActor(torch.nn.Module):
    """A trainable actor with the production input/action and mask interfaces."""
    def __init__(self):
        super().__init__()
        self.head = torch.nn.Linear(408, 12)
        torch.nn.init.zeros_(self.head.weight)
        torch.nn.init.zeros_(self.head.bias)
        with torch.no_grad():
            self.head.bias[0] = .2
        self.register_buffer('actions', torch.from_numpy(ACTIONS.copy()))

    def features(self, observations):
        features = torch.zeros((len(observations), 408))
        features[:, 0] = observations[:, 0]
        return features, self.actions[0].repeat(len(observations), 1)

    def masks(self, observations):
        masks = torch.zeros((len(observations), 12), dtype=torch.bool)
        masks[:, 0] = True
        masks[:, 1] = observations[:, 1] < .5
        return masks

    def support(self, observations):
        return torch.ones(len(observations), dtype=torch.bool)

    def forward(self, observations):
        features, _ = self.features(observations)
        choice = self.head(features).masked_fill(~self.masks(observations), -1e8).argmax(1)
        return self.actions[choice]


def observation(value=1., only_stop=False):
    result = np.zeros((1,625), np.float32)
    result[0,0] = value
    result[0,1] = float(only_stop)
    return result


def update(learner, x):
    with torch.no_grad():
        terms = learner.evaluate(x)
        action = torch.tensor([0 if x[0,1] > .5 else 1])
        old_probability = terms['distribution'].log_prob(action).numpy()
    return learner.update(x, terms['masks'].numpy(), action.numpy(), old_probability,
        np.array([1.],np.float32), np.array([2.],np.float32), sampled_temperature=1.,
        entropy_coefficient=0., anchor_strength=0.)


def assert_state_equal(test, a, b):
    if isinstance(a, torch.Tensor):
        test.assertTrue(torch.equal(a,b))
    elif isinstance(a, dict):
        test.assertEqual(a.keys(),b.keys())
        for key in a:
            assert_state_equal(test,a[key],b[key])
    elif isinstance(a,(list,tuple)):
        test.assertEqual(len(a),len(b))
        for left,right in zip(a,b):
            assert_state_equal(test,left,right)
    else:
        test.assertEqual(a,b)


class RetainedStopTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(17)

    def learner(self, **kwargs):
        return RetainedStopOnPolicy(TinyNativeActor(), observation(),
            learning_rate=1., temperature=1., **kwargs)

    def test_shortened_step_retains_stop_with_motion_still_legal(self):
        learner = self.learner()
        before = copy.deepcopy(learner.actor.head.state_dict())
        statistics = update(learner, observation())
        self.assertTrue(statistics['actor_updated'])
        self.assertGreater(statistics['proposed_stop_mismatches'],0)
        self.assertGreater(statistics['accepted_actor_step_fraction'],0.)
        self.assertLess(statistics['accepted_actor_step_fraction'],1.)
        self.assertEqual(learner.stop_mismatches(),0)
        self.assertTrue(learner.actor.masks(torch.from_numpy(observation()))[0,1])
        self.assertFalse(torch.equal(before['weight'],learner.actor.head.weight))

    def test_compatible_motion_update_is_fully_accepted(self):
        learner = self.learner()
        statistics = update(learner, observation(-1.))
        self.assertTrue(statistics['actor_updated'])
        self.assertEqual(statistics['accepted_actor_step_fraction'],1.)
        self.assertEqual(learner.stop_mismatches(),0)
        self.assertTrue(torch.equal(learner.actor(torch.from_numpy(observation(-1.)))[0],learner.actor.actions[1]))

    def test_rejected_step_restores_actor_and_adam_but_retains_critic_update(self):
        learner = self.learner(maximum_backtracks=0)
        # Populate Adam moments first, so rollback covers existing state.
        update(learner, observation(-1.))
        before = copy.deepcopy(learner.actor.head.state_dict())
        optimizer_before = copy.deepcopy(learner.actor_optimizer.state_dict())
        critic_before = copy.deepcopy(learner.critic.state_dict())
        statistics = update(learner, observation())
        self.assertFalse(statistics['actor_updated'])
        self.assertTrue(statistics['actor_update_attempted'])
        self.assertTrue(statistics['actor_optimizer_restored'])
        assert_state_equal(self,before,learner.actor.head.state_dict())
        assert_state_equal(self,optimizer_before,learner.actor_optimizer.state_dict())
        self.assertTrue(any(not torch.equal(critic_before[k],v) for k,v in learner.critic.state_dict().items()))

    def test_no_motion_keeps_actor_and_optimizer_unchanged(self):
        learner = self.learner()
        before = copy.deepcopy(learner.actor.head.state_dict())
        optimizer_before = copy.deepcopy(learner.actor_optimizer.state_dict())
        statistics = update(learner, observation(only_stop=True))
        self.assertFalse(statistics['actor_updated'])
        self.assertFalse(statistics['actor_update_attempted'])
        self.assertEqual(statistics['eligible_samples'],0)
        assert_state_equal(self,before,learner.actor.head.state_dict())
        assert_state_equal(self,optimizer_before,learner.actor_optimizer.state_dict())

    def test_initial_stop_violation_is_rejected(self):
        actor = TinyNativeActor()
        with torch.no_grad():
            actor.head.bias[1] = 1.
        with self.assertRaisesRegex(ValueError,'Initial actor'):
            RetainedStopOnPolicy(actor,observation())


if __name__ == '__main__':
    unittest.main()
