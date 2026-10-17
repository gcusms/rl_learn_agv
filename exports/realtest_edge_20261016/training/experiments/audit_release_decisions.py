"""After physics, ablate measured context without executing counterfactuals."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'reports/stop_feedback_20261015'
DEPENDENCY = 'rl-release-broad-20261016.service'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def state():
    text = subprocess.check_output(['systemctl', '--user', 'show', DEPENDENCY,
        '-p', 'ActiveState', '-p', 'Result', '-p', 'MainPID', '-p', 'InvocationID'], text=True)
    return dict(line.split('=', 1) for line in text.splitlines())


def main(output):
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    first = state()
    script_sha = digest(Path(__file__))
    write(output / 'WAIT_PLAN.json', dict(predecessor=DEPENDENCY, predecessor_state=first,
        driver_sha256=script_sha, no_compute_or_physics_while_waiting=True,
        new_physical_steps=0, new_optimizer_updates=0,
        counterfactual_actions_never_executed_or_used_as_positive_labels=True))
    while True:
        current = state()
        if current['InvocationID'] not in ('', first['InvocationID']):
            raise RuntimeError('Pinned physical predecessor was restarted')
        if current['ActiveState'] in ('inactive', 'failed'):
            break
        write(output / 'STATUS.json', dict(status='waiting_for_physical_jobs', predecessor=current))
        time.sleep(5)
    assert digest(Path(__file__)) == script_sha
    import numpy as np
    import torch
    from experiments.release_actor import ReleaseActor
    from neural_navigation.experiment_budget import ExperimentBudget
    from neural_navigation.masked_actions import ACTIONS
    from neural_navigation.release_observation import augment
    from neural_navigation.sensors import LidarSpec
    torch.set_num_threads(1)
    budget = ExperimentBudget(.15, memory_mib=2048, directory=output)
    model = ROOT / 'models/stop_feedback_20261015/release_context_bc2000_s231016'
    old_report_path = BASE / 'long_risk_validation_20261016/RESULTS.json'
    new_report_path = BASE / 'release_validation_20261016/RESULTS.json'
    old = json.loads(old_report_path.read_text())
    new = json.loads(new_report_path.read_text())
    old_row = next(row for row in old['rows'] if row['case_id'] == 'early_yield_sensor_holdout_05_0')
    new_rows = [row for row in new['rows'] if row['case_id'] == old_row['case_id']]
    if not new_rows:
        write(output / 'RESULTS.json', dict(completed=True, skipped=True,
            reason='No completed physical repair trace', new_physical_steps=0, goal_achieved=False))
        write(output / 'STATUS.json', dict(status='complete', skipped=True))
        return
    new_row = new_rows[0]
    paths = [Path(__file__), ROOT / 'training/experiments/release_actor.py',
        ROOT / 'training/neural_navigation/release_observation.py', old_report_path, new_report_path,
        model / 'checkpoint.pt', model / 'parent_actor.pt', model / 'portable/hardware.json',
        ROOT / old_row['trace_path'], ROOT / new_row['trace_path']]
    inputs = {str(path.relative_to(ROOT)): digest(path) for path in paths}
    write(output / 'PLAN.json', dict(inputs=inputs, counterfactual_is_diagnostic_only=True,
        no_training_or_execution=True, new_physical_steps=0))
    actor = ReleaseActor(torch.jit.load(str(model / 'parent_actor.pt'), map_location='cpu')).eval()
    actor.load_state_dict(torch.load(model / 'checkpoint.pt', map_location='cpu', weights_only=False)['actor'])
    lidar = LidarSpec(json.loads((model / 'portable/hardware.json').read_text())['lidar'])
    reports = []
    for label, row in [('previous_timeout', old_row), ('new_closed_loop', new_row)]:
        trace_path = ROOT / row['trace_path']
        assert digest(trace_path) == row['trace_sha256']
        with np.load(trace_path, allow_pickle=False) as archive:
            data = {key: archive[key].copy() for key in archive.files}
        observations = data['observations']
        if observations.shape[1] == 481:
            observations = np.stack([augment(obs, scan, lidar) for obs, scan in zip(observations, data['scans'])])
        x = torch.from_numpy(observations)
        with torch.no_grad():
            features, baseline = actor.features(x)
            mask = actor.masks(x)
            active = actor.support(x)
            actual = actor(x).numpy()
        if label == 'new_closed_loop':
            np.testing.assert_array_equal(actual, ACTIONS[data['choices']])
        records = {}
        for name, removed in [('full_context', None), ('without_new_context', slice(88, 312)),
                ('without_ranges', slice(88, 160)), ('without_scan_quality', slice(160, 304)),
                ('without_waypoint_and_goal', slice(304, 308)),
                ('without_measured_motion_and_execution', slice(308, 312))]:
            changed = features.clone()
            if removed is not None:
                changed[:, removed] = 0.
            with torch.no_grad():
                logits = (actor.head(changed) + 2. * changed[:, 76:88]).masked_fill(~mask, -1e8)
                choice = logits.argmax(1)
                command = torch.where(active[:, None], actor.actions[choice], baseline).numpy()
                probability = logits.softmax(1)[:, 0].numpy()
            choices = np.abs(ACTIONS[None] - command[:, None]).max(2).argmin(1)
            wait = (choices == 0) & data['masks'][:, 1:].any(1)
            records[name] = dict(voluntary_wait_frames=int(wait.sum()),
                motion_choice_frames=int((choices != 0).sum()),
                choices_changed_vs_full_context=int((command != actual).any(1).sum()),
                sampled_stop_probabilities=[dict(sensor_time_s=float(data['labels'][i, 0]),
                    choice=int(choices[i]), pstop=float(probability[i])) for i in range(0, len(x), 75)])
            budget.tick()
        reports.append(dict(trace_kind=label, frames=len(x), physical_result={key: row[key]
            for key in ['safe_success', 'collisions', 'timeout', 'seconds', 'travel_m', 'voluntary_wait_frames']},
            counterfactual_ablations=records))
    for name, expected in inputs.items():
        assert digest(ROOT / name) == expected, name
    budget.tick()
    write(output / 'RESULTS.json', dict(completed=True, reports=reports,
        conclusions_require_physical_results=True,
        limitations=['Off-policy ablations do not prove safe counterfactual execution.',
            'Training remains development-data fitting; this is not unseen-map evidence.',
            'Same trained weights with ablated inputs test information use, not a matched retraining control.'],
        new_physical_steps=0, new_optimizer_updates=0, new_ppo_steps=0,
        resource=budget.statistics(), goal_achieved=False, deployment_approved=False))
    write(output / 'STATUS.json', dict(status='complete', diagnostic_only=True))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    main(parser.parse_args().output)
