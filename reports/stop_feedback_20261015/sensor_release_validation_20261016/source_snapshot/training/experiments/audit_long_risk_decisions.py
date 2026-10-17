"""Off-policy sensor replay: risk-head decisions at already recorded warnings.

These counterfactual decisions are never executed or admitted as demonstrations.
Only independent physical evaluation can establish safe learned navigation.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from experiments.long_risk_actor import digest
from neural_navigation.decision_feedback_runtime import StopFeedbackRuntime
from neural_navigation.experiment_budget import ExperimentBudget
from neural_navigation.long_risk_observation import LongRiskHistory
from neural_navigation.masked_actions import ACTIONS


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'reports/stop_feedback_20261015'


def main(output):
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    budget = ExperimentBudget(.15, memory_mib=2048, directory=output)
    frozen = BASE / 'early_yield_holdout_replay_20261016'
    reference = json.loads((frozen / 'RESULTS.json').read_text())
    source = json.loads((frozen / 'FROZEN_FINISHED_CASES.json').read_text())
    cases_path = BASE / 'early_yield_remaining_protocol_20261016/heldout_cases.json'
    cases = json.loads(cases_path.read_text())
    source_bundle = ROOT / 'models/stop_feedback_20261015/early_yield_remaining_bc600_s231016/bc_portable'
    new_bundle = ROOT / 'models/stop_feedback_20261015/long_risk_bc1000_s231016/portable'
    actor = torch.jit.load(str(new_bundle / 'actor.pt'), map_location='cpu').eval()
    manifest = json.loads((source_bundle / 'manifest.json').read_text())
    paths = [Path(__file__), frozen / 'RESULTS.json', frozen / 'FROZEN_FINISHED_CASES.json',
        cases_path, new_bundle / 'actor.pt', source_bundle / 'manifest.json']
    paths += [source_bundle / name for name in manifest['files']]
    paths += [ROOT / row['trace_path'] for row in source['rows']]
    for folder in ('neural_navigation', 'navrl'):
        paths += list((ROOT / 'training' / folder).rglob('*.py'))
    inputs = {str(path.relative_to(ROOT)): digest(path) for path in set(paths)}
    (output / 'PLAN.json').write_text(json.dumps(dict(inputs=inputs, new_physical_steps=0,
        new_optimizer_updates=0, no_counterfactual_actions_executed=True,
        failed_traces_used_only_for_diagnosis=True), indent=2) + '\n')
    records = []
    for row in source['rows']:
        case = next(case for case in cases if case['id'] == row['case_id'])
        with np.load(ROOT / row['trace_path'], allow_pickle=False) as archive:
            data = {key: archive[key].copy() for key in archive.files}
        expected = [record for record in reference['records'] if record['case_id'] == case['id']]
        runtime = StopFeedbackRuntime(source_bundle, ROOT / 'training' / case['scene'] / 'decision_map.yaml',
            np.asarray(case['start']) + np.asarray(case.get('initial_prior_offset', [0, 0, 0])), case['goal'], 'shortcut')
        history = LongRiskHistory()

        class Replay:
            def __init__(self, original):
                self.original, self.cursor = original, 0

            def predict(self, observation):
                i = self.cursor
                np.testing.assert_allclose(observation, data['observations'][i], rtol=0, atol=2e-6)
                original = self.original.predict(observation)
                np.testing.assert_array_equal(original, ACTIONS[data['choices'][i]])
                augmented = history.augment(observation, runtime.core, runtime.guard)
                x = torch.from_numpy(augmented[None])
                with torch.no_grad():
                    command = actor(x).numpy()[0]
                    active = bool(actor.support(x).item())
                    mask = torch.from_numpy(data['masks'][i][None])
                    pstop = float(actor.logits(x).masked_fill(~mask, -1e8).softmax(1)[0, 0]) if active else None
                    ablated = x.clone()
                    ablated[:, 439:] = 0.
                    without_risk = actor(ablated).numpy()[0]
                choice = int(np.abs(ACTIONS - command).max(axis=1).argmin())
                old_choice = int(data['choices'][i])
                forecasts = expected[i]['predictions']['5.0']
                warning = (old_choice != 0 and expected[i]['prediction_input_valid']
                    and forecasts[str(old_choice)]['supported_tracks'] > 0
                    and forecasts[str(old_choice)]['predicted_margin_contact_s'] is not None
                    and forecasts['0']['predicted_margin_contact_s'] is None)
                records.append(dict(case_id=case['id'], time_s=float(data['labels'][i, 0]),
                    recorded_choice=old_choice, new_counterfactual_choice=choice,
                    early_yield_warning=bool(warning), risk_head_active=active,
                    conditional_head_stop_probability=pstop,
                    old_actor_stop_probability=expected[i]['actor_stop_probability'],
                    choice_without_risk_history=int(np.abs(ACTIONS - without_risk).max(axis=1).argmin()),
                    counterfactual_is_not_physical_evidence=True))
                self.cursor += 1
                return original

        replay = Replay(runtime.policy.policy)
        runtime.policy.policy = replay
        for i in range(len(data['choices'])):
            result = runtime.step(data['scans'][i], data['odometry'][i], data['measured_velocity'][i],
                float(data['labels'][i, 0]), capture_observation=True)
            np.testing.assert_allclose(result['command'], data['commands'][i], rtol=0, atol=2e-8)
            np.testing.assert_allclose(runtime.core.estimate, data['labels'][i, 4:7], rtol=0, atol=2e-8)
            budget.tick()
        assert replay.cursor == len(expected)
    for name, expected_hash in inputs.items():
        assert digest(ROOT / name) == expected_hash, name
    warnings = [record for record in records if record['early_yield_warning']]
    budget.tick()
    (output / 'RESULTS.json').write_text(json.dumps(dict(completed=True, reproduced_frames=len(records),
        warnings=len(warnings), warnings_choosing_wait=sum(record['new_counterfactual_choice'] == 0 for record in warnings),
        first_warnings=[next(record for record in warnings if record['case_id'] == row['case_id']) for row in source['rows']],
        records=records, resource=budget.statistics(), new_physical_steps=0,
        new_optimizer_updates=0, new_ppo_steps=0, goal_achieved=False, deployment_approved=False,
        not_evidence_of_counterfactual_physical_success=True), indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    main(args.output)
