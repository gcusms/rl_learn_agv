"""Check outcome credit, episode boundaries and immutable behavior likelihoods."""
import unittest

import numpy as np
import torch

from batch_credit_ppo import BatchCreditOnPolicy,prepare_completed_batch
from test_retained_stop_ppo import TinyNativeActor,observation


class BatchCreditTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(17)
        self.learner=BatchCreditOnPolicy(TinyNativeActor(),observation(),
            learning_rate=.01,temperature=1.)

    def episode(self,terminal_reward,only_stop=False):
        x=np.repeat(observation(-1.,only_stop=only_stop),2,axis=0)
        actions=np.full(2,0 if only_stop else 1,dtype=np.int64)
        with torch.no_grad():
            terms=self.learner.evaluate(x,actions=actions)
        return dict(observations=x,masks=terms['masks'].numpy(),actions=actions,
            old_log_probability=terms['log_probability'].numpy(),old_values=terms['values'].numpy(),
            rewards=np.array([0.,terminal_reward]),terminated=np.array([False,True]),
            episode_ends=np.array([False,True]),temperature=1.)

    def test_success_positive_failure_negative_after_shared_normalization(self):
        batch,record=prepare_completed_batch(self.learner,[self.episode(10.),self.episode(-5.)])
        self.assertTrue((batch['advantages'][:2]>0.).all())
        self.assertTrue((batch['advantages'][2:]<0.).all())
        self.assertAlmostEqual(float(batch['advantages'].mean()),0.,places=6)
        self.assertEqual(record['episode_offsets'],[0,2,4])
        np.testing.assert_allclose(batch['returns'],[9.9,10.,-4.95,-5.],rtol=0,atol=1e-6)

    def test_minibatch_keeps_precomputed_sign_and_scale(self):
        episode=self.episode(10.)
        with torch.no_grad():
            old_probability=float(self.learner.evaluate(episode['observations'])['distribution'].probs[0,1])
        result=self.learner.update(episode['observations'],episode['masks'],episode['actions'],
            episode['old_log_probability'],np.array([.5,1.5],np.float32),np.zeros(2,np.float32),
            sampled_temperature=1.,rollout_advantages_normalized=True,
            entropy_coefficient=0.,anchor_strength=0.)
        self.assertAlmostEqual(result['actor_loss'],-1.,places=6)
        self.assertTrue(result['actor_updated'])
        with torch.no_grad():
            new_probability=float(self.learner.evaluate(episode['observations'])['distribution'].probs[0,1])
        self.assertGreater(new_probability,old_probability)
        self.assertEqual(self.learner.stop_mismatches(),0)

    def test_rejects_mixed_behavior_policies_before_any_update(self):
        first,second=self.episode(10.),self.episode(-5.)
        second['old_log_probability']=second['old_log_probability']+.1
        with self.assertRaisesRegex(ValueError,'unchanged behavior policy'):
            prepare_completed_batch(self.learner,[first,second])
        self.assertEqual(len(self.learner.actor_optimizer.state),0)

    def test_rejects_critic_changed_during_collection(self):
        first,second=self.episode(10.),self.episode(-5.)
        second['old_values']=second['old_values']+.1
        with self.assertRaisesRegex(ValueError,'Critic changed'):
            prepare_completed_batch(self.learner,[first,second])

    def test_rejects_nonterminal_cut_instead_of_bridging_reset(self):
        first,second=self.episode(10.),self.episode(-5.)
        second['terminated'][-1]=False
        with self.assertRaisesRegex(ValueError,'complete terminal task'):
            prepare_completed_batch(self.learner,[first,second])

    def test_only_stop_samples_do_not_set_actor_normalization(self):
        batch,record=prepare_completed_batch(self.learner,[self.episode(-5.,True),self.episode(-5.,True)])
        self.assertEqual(record['eligible_samples'],0)
        np.testing.assert_array_equal(batch['advantages'],np.zeros(4))
        result=self.learner.update(batch['observations'],batch['masks'],batch['actions'],
            batch['old_log_probability'],batch['advantages'],batch['returns'],
            sampled_temperature=1.,rollout_advantages_normalized=True)
        self.assertFalse(result['actor_updated'])
        self.assertTrue(result['actor_optimizer_untouched_on_no_choice'])

    def test_update_requires_explicit_batch_normalization(self):
        episode=self.episode(10.)
        with self.assertRaisesRegex(ValueError,'Normalize once'):
            self.learner.update(episode['observations'],episode['masks'],episode['actions'],
                episode['old_log_probability'],np.ones(2),np.ones(2),sampled_temperature=1.)


if __name__=='__main__':
    unittest.main()
