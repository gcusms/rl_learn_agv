"""Retain all 2030 prior labels, then append only a verified safe suffix."""
from collections import Counter
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch

from experiments.recovery_demonstrations import load_demonstrations
from experiments.stop_recovery_policy import StopRecoveryPPO
from neural_navigation.experiment_budget import ExperimentBudget


ROOT = Path(__file__).resolve().parents[3]
PROTOCOL = Path(__file__).resolve().parent


def main():
    os.chdir(ROOT)
    torch.set_num_threads(1)
    budget = ExperimentBudget(.15, directory=PROTOCOL)
    plan = json.loads((PROTOCOL/'PLAN.json').read_text())
    if hashlib.sha256(Path(plan['source_safe_trace']).read_bytes()).hexdigest() != plan['source_safe_trace_sha256']:
        raise ValueError('The complete physically successful source changed')
    parent = Path('models/stop_feedback_20261015/proactive_augmented_bc_800_s231015')
    model = StopRecoveryPPO.load(parent/'policy_bc.zip', device='cpu')
    specs = json.loads((PROTOCOL/'demonstrations.json').read_text())
    observations, masks, choices, sources = load_demonstrations(specs, model.policy)
    with np.load(parent/'demonstrations.npz', allow_pickle=False) as previous:
        count = len(previous['choices'])
        if count != 2030:
            raise ValueError('The full prior dataset is required')
        for name, value in [('observations', observations), ('masks', masks), ('choices', choices)]:
            np.testing.assert_array_equal(previous[name], value[:count])
    # Identical observation/mask pairs must not receive contradictory labels.
    seen = {}
    for x, mask, label in zip(observations, masks, choices):
        key = hashlib.sha256(x.tobytes()+mask.tobytes()).digest()
        if key in seen and seen[key] != int(label):
            raise ValueError('Contradictory demonstration labels')
        seen[key] = int(label)
    np.savez_compressed(PROTOCOL/'demonstrations.npz', observations=observations,
                        masks=masks, choices=choices)
    budget.tick()
    result = dict(completed=True, total_frames=len(choices), retained_frames=count,
        old_arrays_byte_identical=True, added_frames=len(choices)-count,
        actions=dict(Counter(map(int, choices))),
        added_actions=dict(Counter(map(int, choices[count:]))),
        movement_frames=int(np.count_nonzero(choices)),
        wait_frames=int(np.count_nonzero(choices == 0)),
        sources=sources, contradictory_identical_inputs=0,
        failed_initialization_labels_excluded=True,
        new_episode_is_training_not_holdout=True,
        source_failure_retained=True, runtime_teacher_added=False,
        resource=budget.statistics(), goal_achieved=False, deployment_approved=False)
    (PROTOCOL/'DATA.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k != 'sources'}), flush=True)


if __name__ == '__main__':
    main()
