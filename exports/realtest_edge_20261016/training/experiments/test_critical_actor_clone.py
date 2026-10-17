import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from sb3_contrib import MaskablePPO
from experiments.anticipatory_recovery_ppo import AnticipatoryRecoveryPPO,warm_start_anticipatory
from experiments.critical_actor_clone import warm_start_clone
from experiments.test_anticipatory_recovery_ppo import ToySensors,observations
from neural_navigation.masked_actions import masks_from_observation
from experiments.masked_navigation import PortableMaskedActor


class CriticalCloneTests(unittest.TestCase):
    def test_pretrained_copy_learns_without_changing_clear_base_or_mask_likelihood(self):
        torch.set_num_threads(1);env=ToySensors()
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            parent=MaskablePPO('MlpPolicy',env,n_steps=8,batch_size=4,n_epochs=1,
                policy_kwargs=dict(net_arch=dict(pi=[8],vf=[8])),device='cpu',seed=8510)
            parent.training_reward_scale=1.;parent.save(root/'parent.zip')
            model=warm_start_clone(root/'parent.zip',env,8511)
            reference=warm_start_anticipatory(root/'parent.zip',env,8511)
            x=torch.from_numpy(observations());masks=masks_from_observation(x.numpy()[:,:282])
            np.testing.assert_array_equal(parent.predict(x.numpy(),action_masks=masks,deterministic=True)[0],
                model.predict(x.numpy(),action_masks=masks,deterministic=True)[0])
            policy=model.policy
            with torch.no_grad():
                torch.testing.assert_close(policy.get_distribution(x,action_masks=masks).distribution.logits,
                    reference.policy.get_distribution(x,action_masks=masks).distribution.logits,rtol=0,atol=0)
            for a,b in zip(policy.mlp_extractor.policy_net.base.parameters(),policy.mlp_extractor.policy_net.adapted.parameters()):
                torch.testing.assert_close(a,b,rtol=0,atol=0);self.assertIsNot(a,b)
            actions,_,old=policy(x,action_masks=masks)
            _,again,_=policy.evaluate_actions(x,actions,action_masks=masks)
            torch.testing.assert_close(old,again,rtol=0,atol=0)
            frozen={k:v.detach().clone() for k,v in policy.named_parameters() if not v.requires_grad}
            adapted={k:v.detach().clone() for k,v in policy.named_parameters() if '.adapted.' in k}
            with torch.no_grad():clear_before=policy.get_distribution(x[:1],action_masks=masks[:1]).distribution.logits.clone()
            model.learn(32)
            self.assertTrue(any(not torch.equal(value,dict(policy.named_parameters())[name]) for name,value in adapted.items()))
            for prefix in ('mlp_extractor.policy_net.adapted.','action_net.adapted.'):
                self.assertTrue(any(not torch.equal(value,dict(policy.named_parameters())[name])
                    for name,value in adapted.items() if name.startswith(prefix)))
            for name,value in frozen.items():torch.testing.assert_close(value,dict(policy.named_parameters())[name],rtol=0,atol=0)
            with torch.no_grad():clear_after=policy.get_distribution(x[:1],action_masks=masks[:1]).distribution.logits
            torch.testing.assert_close(clear_before,clear_after,rtol=0,atol=0)
            portable=torch.jit.trace(PortableMaskedActor(policy).eval(),x)
            expected=PortableMaskedActor(policy).eval()(x)
            torch.testing.assert_close(portable(x),expected,rtol=0,atol=0)
            model.save(root/'trained.zip')
            restored=AnticipatoryRecoveryPPO.load(root/'trained.zip',env=env,device='cpu')
            with torch.no_grad():
                torch.testing.assert_close(policy.get_distribution(x,action_masks=masks).distribution.logits,
                    restored.policy.get_distribution(x,action_masks=masks).distribution.logits,rtol=0,atol=0)
            self.assertTrue(restored.policy.optimizer.state)


if __name__=='__main__':unittest.main()
