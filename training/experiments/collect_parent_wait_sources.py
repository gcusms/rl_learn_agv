"""Collect bounded reference-parent continuations after observed legal waiting.

Failed traces initialize the sensor and physical state only. Only complete
safe reference continuations can be used later as positive training sources.
The reference reads the unchanged 439 sensor prefix, not truth or time.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
from unittest.mock import patch

import numpy as np
import torch

from experiments.experiment_provenance import record_provenance
from experiments.long_risk_actor import digest
from experiments.release_recovery import evaluate_recovery
from neural_navigation.experiment_budget import ExperimentBudget
from neural_navigation.masked_actions import ACTIONS
from neural_navigation.release_runtime import ReleaseRuntime


ROOT = Path(__file__).resolve().parents[2]


def write(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def main(args):
    model, output, specifications_path = args.model.resolve(), args.output.resolve(), args.specifications.resolve()
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    budget = ExperimentBudget(.35, memory_mib=3072, directory=output)
    state = subprocess.check_output(['systemctl', '--user', 'show', args.predecessor_unit,
        '-p', 'MainPID', '-p', 'ActiveState'], text=True)
    assert 'MainPID=0\n' in state and 'ActiveState=inactive\n' in state
    specifications = json.loads(specifications_path.read_text())
    assert len(specifications) == 2
    protocol_path = specifications_path.parent / 'PLAN.json'
    protocol = json.loads(protocol_path.read_text())
    assert protocol['maximum_new_physical_episodes'] == 2
    for name, expected in protocol['inputs'].items():
        assert digest(ROOT / name) == expected, name
    bundle = model / 'portable'
    manifest = json.loads((bundle / 'manifest.json').read_text())
    files = [Path(__file__), specifications_path, protocol_path,
        model / 'checkpoint.pt', model / 'parent_actor.pt', bundle / 'manifest.json']
    files += [ROOT / name for name in protocol['inputs']]
    files += [bundle / name for name in manifest['files']]
    files += [ROOT / s['trace'] for s in specifications]
    for folder in ['neural_navigation', 'navrl', 'experiments']:
        files += list((ROOT / 'training' / folder).rglob('*.py'))
    for spec in specifications:
        files += [p for p in (ROOT / 'training' / spec['case']['scene']).iterdir() if p.is_file()]
    hashes = {str(p.relative_to(ROOT)):digest(p) for p in set(files)}
    write(output / 'PLAN.json', dict(inputs=hashes, maximum_new_physical_episodes=2,
        reference='exact frozen 439-dimensional parent of current release actor',
        no_truth_time_or_future_input_to_reference=True, prefix_initialization_only=True,
        failed_prefixes_not_positive_training_labels=True, no_runtime_or_guard_changes=True,
        first_complete_source_failure_stops_expansion=True, new_ppo_steps=0,
        new_optimizer_updates=0, deployment_approved=False))
    record_provenance(output, bundle, [Path(__file__)])
    created = []

    class ParentPolicy:
        def __init__(self, policy, core):
            self.policy, self.core = policy, core
            self.entries = []

        def predict(self, observation):
            tensor = torch.from_numpy(np.asarray(observation, np.float32)[None])
            assert tensor.shape == (1, 625)
            with torch.no_grad():
                command = self.policy.actor.parent(tensor[:, :439]).numpy()[0]
                nominal = self.policy.actor(tensor).numpy()[0]
                active = bool(self.policy.actor.support(tensor).item())
            distance = np.max(np.abs(ACTIONS-command), axis=1)
            choice = int(distance.argmin())
            assert distance[choice] < 2e-6
            nominal_choice = int(np.max(np.abs(ACTIONS-nominal), axis=1).argmin())
            self.entries.append(dict(choice=choice, nominal_choice=nominal_choice,
                trainable=active, sensor_time_s=float(self.core.elapsed)))
            return command.copy()

    class ParentRuntime(ReleaseRuntime):
        def __init__(self, *runtime_args, **kwargs):
            super().__init__(*runtime_args, **kwargs)
            reference = ParentPolicy(self.policy.policy, self.core)
            self.policy.policy = reference
            created.append(reference)

    report = dict(completed=False, rows=[], full_safe_expert_trajectory_exists=False,
        reference_success_is_not_candidate_success=True, no_failed_positive_sources=True,
        new_ppo_steps=0, new_optimizer_updates=0, goal_achieved=False, deployment_approved=False)

    def save():
        budget.tick()
        report.update(at=datetime.now(timezone.utc).isoformat(),resource=budget.statistics())
        write(output / 'RESULTS.json', report)

    def finish(reason):
        for name, expected in hashes.items():
            assert digest(ROOT / name) == expected, name
        report.update(completed=True,reason=reason,
            full_safe_expert_trajectory_exists=bool(report['rows'])
                and all(row['complete_safe_source_admitted'] for row in report['rows']))
        save()
        write(output / 'STATUS.json',dict(status='complete',reason=reason,completed_episodes=len(report['rows'])))
        (output / 'driver_snapshot.py').write_text(Path(__file__).read_text())

    save()
    for spec in specifications:
        write(output / 'STATUS.json',dict(status='reference_parent_recovery_source',case_id=spec['id'],
            finished_episodes=len(report['rows']),release_s=spec['release_s']))
        print(json.dumps(dict(status='starting',case_id=spec['id'],release_s=spec['release_s'])),flush=True)
        before = len(created)
        with patch('experiments.release_recovery.resolve_runtime', return_value=ParentRuntime):
            row, trace = evaluate_recovery(spec, bundle, budget)
        assert len(created) == before + 1
        entries = created[-1].entries
        controlled = trace['model_controlled'].astype(bool)
        assert len(entries) == int(controlled.sum())
        np.testing.assert_array_equal(trace['choices'][controlled], [e['choice'] for e in entries])
        trace['reference_parent_controlled'] = controlled.copy()
        trace['candidate_choices_on_reference_context'] = trace['choices'].copy()
        trace['candidate_choices_on_reference_context'][controlled] = [e['nominal_choice'] for e in entries]
        path = output / (spec['id']+'.npz')
        np.savez_compressed(path,**trace)
        admitted = bool(row['safe_success'] and row['protocol_valid'] and not row['false_arrival']
            and not row['collisions'] and not row['physical_guards'] and not row['timeout'])
        row.update(variant='offline_reference_parent',trace_path=str(path.relative_to(ROOT)),
            trace_sha256=digest(path),reference_parent_controlled_frames=int(controlled.sum()),
            parent_differs_from_candidate_frames=sum(e['choice']!=e['nominal_choice'] for e in entries),
            trainable_reference_frames=sum(e['trainable'] for e in entries),
            reference_success_is_not_candidate_success=True, complete_safe_source_admitted=admitted,
            failed_initialization_prefix_excluded_from_future_positive_labels=True)
        report['rows'].append(row)
        save()
        print(json.dumps(row),flush=True)
        if not admitted:
            finish('Reference did not complete a safe recovery; no failed source admission or expansion')
            return
    finish('Both complete-safe reference recoveries collected; source replay and model fitting still required')


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['model','specifications','output']:
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--predecessor-unit',default='rl-retained-sensor-release-broad-20261016.service')
    args=parser.parse_args()
    try:
        main(args)
    except Exception as error:
        if args.output.is_dir():
            write(args.output/'FAILURE.json',dict(error=repr(error),automatically_restarted=False))
        raise
