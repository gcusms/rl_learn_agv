import unittest
import numpy as np
from prototypes.narrow_long_curriculum import sample_task,LongNarrowCurriculum
from prototypes.narrow_ppo import NarrowPolicy,PortableFineActor
from prototypes.test_narrow_ppo import Interface
from prototypes.narrow_observation import masks,ACTIONS
from sb3_contrib import MaskablePPO
import torch,tempfile
from pathlib import Path

class LongCurriculumTests(unittest.TestCase):
    def test_curriculum_preserves_general_and_full_tasks(self):
        for steps in [0,20000,50000]:
            rng=np.random.default_rng(73123);tasks=[sample_task(rng,steps)['stage'] for _ in range(1000)]
            self.assertGreater(tasks.count('open_room'),150);self.assertGreater(tasks.count('full_passage'),250)
            if steps==0:self.assertGreater(tasks.count('short_exit'),350)
            else:self.assertEqual(tasks.count('short_exit'),0)
    def test_potential_is_scan_local_direction_only(self):
        x=np.zeros(318);self.assertEqual(LongNarrowCurriculum.alignment_potential(x),0)
        x[216]=1;self.assertEqual(LongNarrowCurriculum.alignment_potential(x),0)
        x[216]=0;x[217]=1;self.assertAlmostEqual(LongNarrowCurriculum.alignment_potential(x),-.5)
    def test_prior_penalty_reload_and_portable_parity(self):
        torch.set_num_threads(1)
        model=MaskablePPO(NarrowPolicy,Interface(),n_steps=10,batch_size=5,device='cpu',policy_kwargs=dict(fine_prior_penalty=.2))
        x=np.zeros(318,np.float32);x[308]=1;x[300:308]=1
        index,_=model.predict(x,action_masks=masks(x),deterministic=True)
        portable=torch.jit.trace(PortableFineActor(model.policy).eval(),torch.from_numpy(x)[None])
        np.testing.assert_array_equal(portable(torch.from_numpy(x)[None])[0].detach().numpy(),ACTIONS[int(index)])
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'model.zip';model.save(path)
            other=MaskablePPO.load(path,device='cpu')
            self.assertEqual(other.policy.mlp_extractor.policy_net.fine_prior_penalty,.2)
            choice,_=other.predict(x,action_masks=masks(x),deterministic=True);self.assertEqual(index,choice)

if __name__=='__main__':unittest.main()
