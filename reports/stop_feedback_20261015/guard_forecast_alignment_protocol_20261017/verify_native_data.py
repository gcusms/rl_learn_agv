"""Verify runtime history assembly against causally prepared training rows.

The existing v1 current forecast is provided from the saved source. This tests
the new fusion/history step, not localization, motion forecasting accuracy or
closed-loop navigation. All initialization frames warm history before labels.
"""
from datetime import datetime
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import numpy as np

from neural_navigation.aligned_risk_observation import AlignedRiskHistory
from neural_navigation.experiment_budget import ExperimentBudget


ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    output = HERE/'NATIVE_ALIGNMENT_PARITY.json'
    assert not output.exists()
    data_path = HERE/'ALIGNED_DEMONSTRATIONS.json'
    data = json.loads(data_path.read_text())
    assert data['data_preflight_passed']
    archive = ROOT/data['archive']
    assert digest(archive) == data['archive_sha256']
    with np.load(archive, allow_pickle=False) as saved:
        wanted = {key: saved[key].copy() for key in ['observations','episode_id','frame_index']}
    paths = [Path(__file__), data_path, archive,
        ROOT/'training/neural_navigation/aligned_risk_observation.py',
        ROOT/'training/neural_navigation/aligned_release_runtime.py',
        ROOT/'training/neural_navigation/long_risk_observation.py',
        ROOT/'training/neural_navigation/experiment_budget.py']
    budget = ExperimentBudget(.1, memory_mib=550, directory=HERE)
    compared = warmed = 0
    for episode, row in enumerate(data['source_rows']):
        path = ROOT/row['source']
        assert digest(path) == row['sha256']
        paths.append(path)
        selected = np.flatnonzero(wanted['episode_id'] == episode)
        lookup = {int(wanted['frame_index'][i]): int(i) for i in selected}
        history = AlignedRiskHistory()
        with np.load(path, allow_pickle=False) as saved:
            old = saved['observations']
            diagnostics = saved['constraint_diagnostics']

            class SavedCurrentForecast:
                def __init__(self):
                    self.index = 0
                def augment(self, observation, core, guard):
                    np.testing.assert_array_equal(observation, old[self.index, :439])
                    return old[self.index, :481].copy()

            original = SavedCurrentForecast()
            history.original = original
            for frame, timestamp in enumerate(saved['labels'][:, 0]):
                original.index = frame
                predictor = SimpleNamespace(contacts=diagnostics[frame, 3:],
                    supported_tracks=int(diagnostics[frame, 0]), timestamp=float(timestamp))
                actual = history.augment(old[frame, :439], SimpleNamespace(elapsed=float(timestamp)),
                                         SimpleNamespace(predictive_constraint=predictor))
                if frame in lookup:
                    np.testing.assert_array_equal(actual, wanted['observations'][lookup[frame], :481])
                    compared += 1
                else:
                    warmed += 1
                if frame % 100 == 0:
                    budget.tick()
    assert compared == data['admitted_training_rows']
    budget.tick()
    result = dict(completed=True, at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
        compared_training_rows=compared, initialization_warmup_frames=warmed,
        entire481_prefix_byte_exact=True, no_neural_network_or_physics_evaluated=True,
        existing_current_long_prediction_supplied_from_record=True,
        no_navigation_or_localization_accuracy_claim=True, new_ppo_steps=0,
        goal_achieved=False, deployment_approved=False,
        inputs={str(p.relative_to(ROOT)): digest(p) for p in set(paths)}, resource=budget.statistics())
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({key: result[key] for key in ['compared_training_rows',
        'initialization_warmup_frames','entire481_prefix_byte_exact','resource']}))


if __name__ == '__main__':
    main()
