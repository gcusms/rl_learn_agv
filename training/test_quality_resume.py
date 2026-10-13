import copy
import tempfile
from pathlib import Path
import unittest
from unittest.mock import Mock
import numpy as np
import torch
from sb3_contrib import MaskablePPO
from training.test_quality_prediction_context import Toy
from experiments.quality_masked_navigation import QualityMaskedWrapper,resume_quality


class QualityResumeTests(unittest.TestCase):
    def test_resume_preserves_weights_adam_and_reward_units_then_learns(self):
        torch.set_num_threads(1)
        env=Toy(300);env.reward_scale=1/60.
        model=MaskablePPO('MlpPolicy',env,device='cpu',seed=24,n_steps=16,batch_size=16,n_epochs=1)
        model.learn(16);model.training_reward_scale=env.reward_scale
        weights=copy.deepcopy(model.policy.state_dict())
        optimizer=copy.deepcopy(model.policy.optimizer.state_dict())
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'policy.zip';model.save(path)
            resumed=resume_quality(path,env)
            for key,value in weights.items():torch.testing.assert_close(value,resumed.policy.state_dict()[key],rtol=0,atol=0)
            actual=resumed.policy.optimizer.state_dict()
            self.assertEqual(optimizer['param_groups'],actual['param_groups'])
            for index,state in optimizer['state'].items():
                for key,value in state.items():torch.testing.assert_close(value,actual['state'][index][key],rtol=0,atol=0)
            self.assertEqual(resumed.training_reward_scale,1/60.)
            self.assertIsNotNone(resumed.reference_policy)
            resumed.learn(16,reset_num_timesteps=True)
            env.reward_scale=1.
            with self.assertRaisesRegex(ValueError,'reward scale disagree'):resume_quality(path,env)

    def test_scale_changes_only_reward_and_keeps_native_diagnostic(self):
        base=Toy(282);base.hardware={'robot_footprint_m':[.394,.329]};base.core=None
        base.step=Mock(return_value=(np.zeros(282,np.float32),60.,True,False,{'command':(.1,0)}))
        base.action_masks=Mock(return_value=np.ones(12,bool))
        wrapped=QualityMaskedWrapper(base,reward_scale=1/60.)
        wrapped.prediction.augment=Mock(return_value=np.zeros(300,np.float32))
        obs,reward,done,truncated,info=wrapped.step(1)
        self.assertEqual(reward,1.);self.assertEqual(info['native_reward'],60.)
        self.assertEqual(info['command'],(.1,0));self.assertTrue(done);self.assertFalse(truncated)
        np.testing.assert_array_equal(wrapped.action_masks(),np.ones(12,bool))
        base.step.assert_called_once_with(1)
        self.assertEqual(obs.shape,(300,))


if __name__=='__main__':unittest.main()
