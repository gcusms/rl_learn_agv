"""Check whether every forced moving label can be represented by this residual."""
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch

from experiments.recovery_demonstrations import convert_observations
from experiments.stop_recovery_policy import StopRecoveryPPO
from neural_navigation.experiment_budget import ExperimentBudget


ROOT = Path(__file__).resolve().parents[3]
PROTOCOL = Path(__file__).resolve().parent


def main():
    os.chdir(ROOT)
    torch.set_num_threads(1)
    budget = ExperimentBudget(.05, directory=PROTOCOL)
    source = Path('reports/stop_feedback_20261015/w11_early_left_probes_20261016')
    report = json.loads((source/'RESULTS.json').read_text())
    row = next(r for r in report['rows'] if r['specification']['id'] == 'w11_left_turn_then_left_pass')
    path = source/'w11_left_turn_then_left_pass.npz'
    if (not report['completed'] or not row['safe_success']
            or hashlib.sha256(path.read_bytes()).hexdigest() != row['trace_sha256']):
        raise ValueError('Complete safe physical source required')
    model = StopRecoveryPPO.load(
        'models/stop_feedback_20261015/proactive_augmented_bc_800_s231015/policy_bc.zip', device='cpu')
    with np.load(path, allow_pickle=False) as data:
        times = data['labels'][:, 0]
        masks = data['masks'].astype(bool)
        observations = convert_observations(data['observations'], masks, data['commands'], times)
        choices = data['choices'].copy()
    with torch.no_grad():
        x = torch.from_numpy(observations)
        features = model.policy.mlp_extractor.policy_net(x)
        logits, support = model.policy.action_net.parts(features)
        frozen = logits.masked_fill(~torch.from_numpy(masks), -1e8).argmax(-1).numpy()
        supported = support.numpy()
    suffix = times >= 90.-1e-8
    forced = np.zeros(len(times), dtype=bool)
    for entry in row['intervention']:
        if entry['replaced_reference']:
            index = int(abs(times-entry['sensor_timestamp']).argmin())
            if abs(times[index]-entry['sensor_timestamp']) > 1e-7:
                raise ValueError('Intervention does not correspond to an actual sensor frame')
            forced[index] = True
    unsupported_change = suffix & forced & ~supported & (choices != frozen)
    if unsupported_change.any():
        raise ValueError('Demonstration needs changes outside the trainable residual support')
    budget.tick()
    result = dict(completed=True, source_trace_sha256=row['trace_sha256'],
        suffix_frames=int(suffix.sum()), supported_suffix_frames=int((suffix & supported).sum()),
        actually_applied_interventions=int((suffix & forced).sum()),
        interventions_outside_supported_states=int((suffix & forced & ~supported).sum()),
        required_action_changes_outside_support=int(unsupported_change.sum()),
        frozen_tail_disagreements=int((suffix & ~forced & ~supported & (choices != frozen)).sum()),
        failed_prefix_excluded=True, actor_inputs_include_truth=False,
        interpretation='Offline representability of the checked demonstration only; learned closed-loop success still required.',
        resource=budget.statistics(), goal_achieved=False, deployment_approved=False)
    (PROTOCOL/'SOURCE_SUPPORT.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
