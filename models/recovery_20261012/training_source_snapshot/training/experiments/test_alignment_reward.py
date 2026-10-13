import unittest
import gymnasium as gym
import numpy as np
from experiments.alignment_reward import AlignmentReward,forward_alignment_cost


def state():
    obs=np.zeros(300,np.float32);obs[216]=-.1;obs[229]=1;obs[232]=-.5;obs[294]=1
    return obs


class SampleEnv(gym.Env):
    observation_space=gym.spaces.Box(-1,1,(300,),np.float32)
    action_space=gym.spaces.Discrete(12)
    reward_scale=1/60
    def reset(self,**kwargs):return state(),{}
    def step(self,action):
        obs=state();obs[232]=1.
        return obs,1/60,False,False,dict(commanded_velocity=np.array([.05,.5]),native_reward=1.)


class AlignmentRewardTests(unittest.TestCase):
    def test_cost_uses_executed_forward_motion_and_trusted_heading(self):
        obs=state();self.assertAlmostEqual(forward_alignment_cost(obs,[.05,.5],3.),.375)
        for command in ([0,0],[0,.5],[-.02,.5]):self.assertEqual(forward_alignment_cost(obs,command,3.),0)
        for index,value in ((232,.9),(294,0),(229,.1)):
            altered=obs.copy();altered[index]=value
            self.assertEqual(forward_alignment_cost(altered,[.05,.5],3.),0)
        self.assertEqual(forward_alignment_cost(obs,[.05,.5],0.),0)

    def test_pre_action_state_reward_units_and_disabled_control(self):
        env=AlignmentReward(SampleEnv(),3.);env.reset()
        obs,reward,_,_,info=env.step(6)
        self.assertEqual(obs[232],1.);self.assertAlmostEqual(reward,.625/60)
        self.assertEqual(info['native_base_reward'],1.);self.assertAlmostEqual(info['native_reward'],.625)
        np.testing.assert_array_equal(info['commanded_velocity'],[.05,.5])
        control=AlignmentReward(SampleEnv(),0.);control.reset()
        self.assertEqual(control.step(6)[1],1/60)


if __name__=='__main__':unittest.main()
