from pathlib import Path
import random
import tempfile
import unittest

import numpy as np
import torch

from experiments.long_risk_actor import digest
from experiments.refit_phase_tail_pair import optimizer_digest, restore_fit
from experiments.train_long_risk_actor import state_digest


class SmallActor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.head = torch.nn.Linear(2, 2)


class PairedResumeTests(unittest.TestCase):
    def test_first_treatment_cannot_change_second_start_or_source(self):
        source = SmallActor()
        optimizer = torch.optim.Adam(source.head.parameters(), lr=.001)
        source.head(torch.ones(1, 2)).sum().backward(); optimizer.step()
        expected_actor, expected_optimizer = state_digest(source), optimizer_digest(optimizer)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'checkpoint.pt'
            torch.save(dict(actor=source.state_dict(), optimizer=optimizer.state_dict(),
                torch_rng=torch.get_rng_state(), numpy_rng=np.random.get_state(),
                python_rng=random.getstate()), path)
            original_sha = digest(path)
            first, second = SmallActor(), SmallActor()
            _, first_optimizer = restore_fit(path, first)
            first.head(torch.ones(1, 2)).square().sum().backward(); first_optimizer.step()
            self.assertNotEqual(optimizer_digest(first_optimizer), expected_optimizer)
            _, second_optimizer = restore_fit(path, second)
            self.assertEqual(state_digest(second), expected_actor)
            self.assertEqual(optimizer_digest(second_optimizer), expected_optimizer)
            self.assertEqual(digest(path), original_sha)


if __name__ == '__main__':
    unittest.main()
