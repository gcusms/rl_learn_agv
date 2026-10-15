import tempfile
import unittest
from pathlib import Path
import gymnasium as gym
import numpy as np
import torch
from sb3_contrib import MaskablePPO
from neural_navigation.masked_actions import masks_from_observation
from experiments.focused_recovery_policy import recovery_support
from experiments.anticipatory_recovery_ppo import anticipatory_support,warm_start_anticipatory,AnticipatoryRecoveryPPO


def observations():
    x=np.zeros((4,300),np.float32)
    x[:,235:244]=1.;x[:,267:269]=1.
    x[:,250]=1.;x[:,234]=.8;x[:,232]=.5;x[:,294]=1.
    x[1,281]=1.                    # Moving track, before waiting or a rejection.
    x[2,235]=0.                    # Fewer choices, but a legal alternative exists.
    x[3,235:244]=0.;x[3,267:269]=0. # Only stop: never an actor-learning sample.
    return x


class ToySensors(gym.Env):
    reward_scale=1.
    observation_space=gym.spaces.Box(-1,1,(300,),np.float32)
    action_space=gym.spaces.Discrete(12)
    def reset(self,seed=None,options=None):
        super().reset(seed=seed);self.index=0
        return observations()[self.index],{}
    def action_masks(self):return masks_from_observation(observations()[self.index,:282])
    def step(self,action):
        if not self.action_masks()[action]:raise AssertionError('Invalid sampled action')
        reward=float(action==1);self.index=(self.index+1)%4
        return observations()[self.index],reward,False,False,{}


class AnticipatoryTests(unittest.TestCase):
    def test_exact_replay_does_not_rewind_general_task_randomness(self):
        from unittest.mock import patch
        from experiments.recovery_sampling import ScheduledRecoveryCurriculum
        from experiments.masked_navigation import FocusedMaskedCurriculum
        def replay(env,case):
            gym.Env.reset(env,seed=999)
            return np.array([999]),{}
        def general(env,seed=None,options=None):return np.array([env.np_random.random()]),{}
        def sequence():
            env=ScheduledRecoveryCurriculum.__new__(ScheduledRecoveryCurriculum)
            env.recorded_cases=[{}];env.recovery_reset_index=0
            env.recorded_episodes_per_cycle=1;env.recorded_case_index=0
            env.reset(seed=8102)
            first=env.reset()[0];env.reset();second=env.reset()[0]
            return np.r_[first,second]
        with patch('experiments.recovery_sampling.reset_training_episode',side_effect=replay),patch.object(FocusedMaskedCurriculum,'reset',general):
            a=sequence();b=sequence()
        self.assertNotEqual(a[0],a[1]);np.testing.assert_array_equal(a,b)

    def test_three_in_four_replay_covers_all_recorded_cases_and_keeps_general_tasks(self):
        from unittest.mock import patch
        from experiments.recovery_sampling import ScheduledRecoveryCurriculum
        from experiments.masked_navigation import FocusedMaskedCurriculum
        env=ScheduledRecoveryCurriculum.__new__(ScheduledRecoveryCurriculum)
        env.recorded_cases=[0,1,2];env.recovery_reset_index=0
        env.recorded_episodes_per_cycle=3;env.recorded_case_index=0
        replay=lambda env,case:(np.array([case]),{})
        general=lambda env,**kw:(np.array([-1]),{})
        with patch('experiments.recovery_sampling.reset_training_episode',side_effect=replay),patch.object(FocusedMaskedCurriculum,'reset',general):
            values=[int(env.reset(seed=42 if i==0 else None)[0][0]) for i in range(8)]
        self.assertEqual(values,[0,1,2,-1,0,1,2,-1])

    def test_early_moving_and_narrowed_choices_are_trainable_without_training_forced_stops(self):
        x=torch.from_numpy(observations())
        self.assertFalse(recovery_support(x).any())
        np.testing.assert_array_equal(anticipatory_support(x),[False,True,True,False])

    def test_sampling_update_freeze_and_checkpoint_restore_use_the_same_distribution(self):
        torch.set_num_threads(1);env=ToySensors()
        with tempfile.TemporaryDirectory() as directory:
            directory=Path(directory)
            parent=MaskablePPO('MlpPolicy',env,n_steps=8,batch_size=4,n_epochs=1,
                policy_kwargs=dict(net_arch=dict(pi=[8],vf=[8])),device='cpu',seed=8210)
            parent.training_reward_scale=1.;parent.save(directory/'parent.zip')
            model=warm_start_anticipatory(directory/'parent.zip',env,8211)
            x=torch.from_numpy(observations());masks=masks_from_observation(x.numpy()[:,:282])
            expected=parent.predict(x.numpy(),action_masks=masks,deterministic=True)[0]
            np.testing.assert_array_equal(model.predict(x.numpy(),action_masks=masks,deterministic=True)[0],expected)
            actions,_,old=model.policy(x,deterministic=False,action_masks=masks)
            _,again,_=model.policy.evaluate_actions(x,actions,action_masks=masks)
            torch.testing.assert_close(old,again,rtol=0,atol=0)
            frozen={name:value.detach().clone() for name,value in model.policy.named_parameters() if not value.requires_grad}
            residual=model.policy.action_net.residual.weight.detach().clone()
            with torch.no_grad():before_clear=model.policy.get_distribution(x[:1],action_masks=masks[:1]).distribution.logits.clone()
            model.learn(16)
            for name,value in model.policy.named_parameters():
                if name in frozen:torch.testing.assert_close(value,frozen[name],rtol=0,atol=0)
            self.assertFalse(torch.equal(residual,model.policy.action_net.residual.weight))
            with torch.no_grad():after_clear=model.policy.get_distribution(x[:1],action_masks=masks[:1]).distribution.logits
            torch.testing.assert_close(before_clear,after_clear,rtol=0,atol=0)
            model.save(directory/'trained.zip')
            restored=AnticipatoryRecoveryPPO.load(directory/'trained.zip',env=env,device='cpu')
            with torch.no_grad():
                original=model.policy.get_distribution(x,action_masks=masks).distribution.logits
                loaded=restored.policy.get_distribution(x,action_masks=masks).distribution.logits
            torch.testing.assert_close(original,loaded,rtol=0,atol=0)
            self.assertTrue(restored.policy.optimizer.state)


if __name__=='__main__':unittest.main()
