"""One complete offline sensor-release demonstration, without a deployment claim."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from experiments.evaluate_docking_navigation import evaluate
from experiments.experiment_provenance import record_provenance
from experiments.long_risk_actor import digest
from experiments.sensor_release_teacher import SensorReleaseTeacher, contract
from neural_navigation.experiment_budget import ExperimentBudget
from neural_navigation.masked_actions import ACTIONS
from neural_navigation.release_runtime import ReleaseRuntime
from neural_navigation.sensors import LidarSpec


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'reports/stop_feedback_20261015'


def write(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def main(args):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    budget = ExperimentBudget(.35, memory_mib=3072, directory=output)
    model = ROOT / 'models/stop_feedback_20261015/release_phase300_s231016'
    bundle = model / 'portable'
    source_cases = BASE / 'long_risk_protocol_20261016/new_sensor_cases.json'
    source_result = BASE / 'release_phase_validation_20261016/RESULTS.json'
    prior = json.loads(source_result.read_text())
    assert prior['completed']
    previous = next(r for r in prior['rows'] if r['case_id'] == 'long_risk_sensor_05_1'
        and r['variant'] == 'recovery_phase')
    assert previous['safe_success'] and previous['voluntary_wait_frames'] == 323
    case = next(c for c in json.loads(source_cases.read_text()) if c['id'] == previous['case_id'])
    write(output / 'case.json', case)
    write(output / 'teacher_contract.json', contract())
    hardware = json.loads((bundle / 'hardware.json').read_text())
    lidar = LidarSpec(hardware['lidar'])
    forward = np.array([abs(angles.mean()+lidar.mount[3]) <= np.deg2rad(35.)
        for angles in np.array_split(lidar.angles, 72)])
    manifest = json.loads((bundle / 'manifest.json').read_text())
    files = [Path(__file__), source_cases, source_result, model / 'RESULTS.json',
        model / 'checkpoint.pt', bundle / 'manifest.json', output / 'case.json', output / 'teacher_contract.json']
    files += [bundle / name for name in manifest['files']]
    for folder in ['neural_navigation', 'navrl', 'experiments']:
        files += list((ROOT / 'training' / folder).rglob('*.py'))
    files += [p for p in (ROOT / 'training' / case['scene']).iterdir() if p.is_file()]
    hashes = {str(p.relative_to(ROOT)): digest(p) for p in set(files)}
    write(output / 'PLAN.json', dict(inputs=hashes, maximum_new_physical_episodes=1,
        known_development_failure_used_for_training_source=True, no_independent_test_claim=True,
        reference_model=str(model.relative_to(ROOT)), reference=previous,
        teacher_contract=contract(), teacher_forward_sector_indices=np.flatnonzero(forward).tolist(),
        teacher_is_offline_and_sensor_only=True, truth_only_for_physical_scoring=True,
        complete_safe_episode_required_for_any_positive_source=True,
        no_action_mask_body_margin_guard_memory_or_deadline_changes=True,
        new_optimizer_updates=0, new_ppo_steps=0, deployment_approved=False))
    record_provenance(output, bundle, [Path(__file__)])
    created = []

    class Policy:
        def __init__(self, actor, core):
            self.actor, self.core = actor, core
            self.teacher = SensorReleaseTeacher(forward)
            self.entries = []

        def predict(self, observation):
            reference = self.actor.predict(observation)
            differences = np.max(abs(ACTIONS-reference), axis=1)
            nominal = int(differences.argmin())
            assert differences[nominal] < 2e-6
            with torch.no_grad():
                active = bool(self.actor.actor.support(torch.from_numpy(observation[None])).item())
            entry = self.teacher.select(observation, nominal, active)
            # Sensor time is logged after selection; it is not a teacher input.
            entry['sensor_time_s'] = float(self.core.elapsed)
            self.entries.append(entry)
            return ACTIONS[entry['choice']].copy()

    class Runtime(ReleaseRuntime):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            teacher = Policy(self.policy.policy, self.core)
            self.policy.policy = teacher
            created.append(teacher)

    write(output / 'STATUS.json', dict(status='single_offline_teacher_episode', case_id=case['id']))
    with patch('neural_navigation.goal_pose_runtime.resolve_runtime', return_value=Runtime):
        row, trace = evaluate(case, bundle, budget)
    assert len(created) == 1 and len(created[0].entries) == len(trace['choices'])
    entries = created[0].entries
    controlled = np.array([e['teacher_controlled'] for e in entries], bool)
    trace['teacher_controlled'] = controlled
    trace['reference_choices'] = np.array([e['nominal'] for e in entries], np.int64)
    trace['teacher_reasons'] = np.array([e['reason'] for e in entries])
    np.testing.assert_array_equal(np.array([e['choice'] for e in entries]), trace['choices'])
    assert all(not e['teacher_controlled'] or (e['nominal'] == 0 and e['choice'] != 0
        and e['trainable']) for e in entries)
    path = output / 'sensor_release_demonstration.npz'
    np.savez_compressed(path, **trace)
    row.update(variant='offline_sensor_release_teacher', trace_path=str(path.relative_to(ROOT)),
        trace_sha256=digest(path), teacher_controlled_frames=int(controlled.sum()),
        model_controlled_frames=int((~controlled).sum()), intervention=entries,
        not_a_learned_model_success=True, not_automatically_admitted_as_training=True)
    for name, expected in hashes.items():
        assert digest(ROOT / name) == expected, name
    budget.tick()
    full_safe = bool(row['safe_success'] and controlled.any() and not row['false_arrival']
        and not row['physical_guards'] and not row['collisions'] and not row['timeout'])
    write(output / 'RESULTS.json', dict(completed=True, at=datetime.now(timezone.utc).isoformat(),
        rows=[row], full_safe_expert_trajectory_exists=full_safe,
        resource=budget.statistics(), new_ppo_steps=0, new_optimizer_updates=0,
        goal_achieved=False, deployment_approved=False))
    write(output / 'STATUS.json', dict(status='complete', safe_success=row['safe_success'],
        full_safe_expert_trajectory_exists=full_safe))
    (output / 'driver_snapshot.py').write_text(Path(__file__).read_text())
    print(json.dumps({k:v for k,v in row.items() if k != 'intervention'}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    try:
        main(args)
    except Exception as error:
        if args.output.is_dir():
            write(args.output / 'FAILURE.json', dict(error=repr(error), automatically_restarted=False))
        raise
