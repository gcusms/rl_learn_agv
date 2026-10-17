"""Reconstruct frozen complete reference missions with causal native625 inputs.

This is recorded-sensor replay, not new navigation success. It supplies normal
navigation retention data across the entire fixed development suite, rather
than adding one recently failed reference at a time.
"""
import argparse
from datetime import datetime, timezone
import gc
import json
from pathlib import Path
import subprocess

import numpy as np
import torch

from experiments.long_risk_actor import digest
from experiments.refit_phase_tail_pair import write
from neural_navigation.experiment_budget import ExperimentBudget
from neural_navigation.masked_actions import ACTIONS, masks_from_observation
from neural_navigation.release_runtime import ReleaseRuntime


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'reports/stop_feedback_20261015'


class FrozenReference:
    def __init__(self, actor):
        self.actor = actor

    @torch.no_grad()
    def predict(self, observation):
        return self.actor(torch.from_numpy(np.asarray(observation[:439], np.float32)[None])).numpy()[0].copy()


def main(args):
    output, model = args.output.resolve(), args.model.resolve()
    output.mkdir(parents=True, exist_ok=False)
    traces = output / 'traces'
    traces.mkdir()
    torch.set_num_threads(1)
    budget = ExperimentBudget(args.cpu_cores, memory_mib=1600, directory=output)
    tasks_path = BASE / 'context_feedback_longfit_protocol_20261016/tasks.json'
    tasks = json.loads(tasks_path.read_text())
    cases = tasks['normal18']
    if args.case_id:
        cases = [c for c in cases if c['id'] in args.case_id]
        assert len(cases) == len(set(args.case_id))
    references = {r['case_id']: r for r in tasks['reference']['normal18']}
    reusable = {}
    if args.reuse_verified_rows:
        if not args.reuse_source_unit or not args.reuse_source_invocation:
            raise ValueError('Reusing partial rows requires the exact terminal source unit')
        result = subprocess.run(['systemctl', '--user', 'show', args.reuse_source_unit,
            '--property=MainPID,ActiveState,InvocationID'], text=True, capture_output=True, check=True)
        state = dict(line.split('=', 1) for line in result.stdout.splitlines() if '=' in line)
        assert state['MainPID'] == '0' and state['ActiveState'] in ('inactive', 'failed')
        assert state['InvocationID'] in ('', args.reuse_source_invocation)
        reusable = {r['case_id']: r for r in json.loads(args.reuse_verified_rows.read_text())['rows']}
    bundle = model / 'portable'
    reference_actor_path = args.reference_actor.resolve()
    manifest = json.loads((bundle / 'manifest.json').read_text())
    files = [Path(__file__), tasks_path, reference_actor_path, bundle / 'manifest.json']
    files += [bundle / name for name in manifest['files']]
    files += [ROOT / references[c['id']]['trace_path'] for c in cases]
    if args.reuse_verified_rows:
        files.append(args.reuse_verified_rows.resolve())
        files += [ROOT / row['trace_path'] for row in reusable.values()]
    for folder in ['navrl', 'neural_navigation']:
        files += list((ROOT / 'training' / folder).rglob('*.py'))
    files += [ROOT / 'training/neural_navigation/static_tracking_kernel.cpp']
    for case in cases:
        files += [p for p in (ROOT / 'training' / case['scene']).iterdir() if p.is_file()]
    hashes = {str(p.relative_to(ROOT)): digest(p) for p in set(files)}
    write(output / 'PLAN.json', dict(inputs=hashes, case_ids=[c['id'] for c in cases],
        reference_actor_path=str(reference_actor_path.relative_to(ROOT)),
        raw_scan_odometry_velocity_timestamps_only=True,
        complete439_inputs_masks_choices_and_execution_must_reproduce=True,
        no_simulator_or_future_truth_inputs=True, no_physics_or_optimizer=True,
        reused_development_references_not_new_or_independent_success=True,
        new_ppo_steps=0, deployment_approved=False))
    (output / 'driver_snapshot.py').write_text(Path(__file__).read_text())
    parent = torch.jit.load(str(reference_actor_path), map_location='cpu').eval()
    report = dict(completed=False, rows=[], source_frames=0, new_ppo_steps=0,
        goal_achieved=False, deployment_approved=False)
    for case in cases:
        row = references[case['id']]
        assert row['safe_success'] and row['protocol_valid']
        assert not any(row[k] for k in ['collisions', 'false_arrival', 'physical_guards', 'timeout'])
        source = ROOT / row['trace_path']
        assert digest(source) == row['trace_sha256']
        with np.load(source, allow_pickle=False) as archive:
            arrays = {key: archive[key].copy() for key in archive.files}
        assert arrays['observations'].shape == (row['frames'], 439)
        with torch.no_grad():
            np.testing.assert_array_equal(parent(torch.from_numpy(arrays['observations'])).numpy(),
                ACTIONS[arrays['choices']])
        if case['id'] in reusable:
            reused = reusable[case['id']]
            target = ROOT / reused['trace_path']
            assert reused['reference_actor_sha256'] == digest(reference_actor_path)
            assert reused['source_sha256'] == digest(source) and digest(target) == reused['trace_sha256']
            assert reused['complete_masks_actions_and_commands_reproduced']
            assert reused['maximum439_input_difference'] <= 2e-6
            with np.load(target, allow_pickle=False) as cached:
                np.testing.assert_allclose(cached['observations'][:, :439], arrays['observations'], rtol=0, atol=2e-6)
                for key in ['choices', 'masks', 'commands', 'scans', 'odometry', 'measured_velocity', 'labels']:
                    np.testing.assert_array_equal(cached[key], arrays[key])
            report['rows'].append(dict(reused, reused_verified_case_from_terminal_partial_run=True))
            report['source_frames'] += row['frames']
            budget.tick()
            report['resource'] = budget.statistics()
            write(output / 'RESULTS.json', report)
            continue
        prior = np.asarray(case['start']) + np.asarray(case.get('initial_prior_offset', [0., 0., 0.]))
        runtime = ReleaseRuntime(bundle, ROOT / 'training' / case['scene'] / 'decision_map.yaml',
            prior, case['goal'], 'shortcut')
        runtime.policy.policy = FrozenReference(parent)
        write(output / 'STATUS.json', dict(status='causal_recorded_reference_replay',
            case_id=case['id'], matched_frames=0, complete_cases=len(report['rows'])))
        observations, maximum_difference = [], 0.
        for frame in range(row['frames']):
            result = runtime.step(arrays['scans'][frame], arrays['odometry'][frame],
                arrays['measured_velocity'][frame], float(arrays['labels'][frame, 0]),
                capture_observation=True)
            observation = np.asarray(result['policy_observation'], np.float32)
            np.testing.assert_allclose(observation[:439], arrays['observations'][frame], rtol=0, atol=2e-6)
            np.testing.assert_array_equal(masks_from_observation(observation[:282]), arrays['masks'][frame])
            np.testing.assert_array_equal(result['proposed_action'], ACTIONS[arrays['choices'][frame]])
            np.testing.assert_allclose(result['command'], arrays['commands'][frame], rtol=0, atol=2e-8)
            maximum_difference = max(maximum_difference, float(abs(observation[:439] - arrays['observations'][frame]).max()))
            observations.append(observation)
            if frame % 25 == 0:
                write(output / 'STATUS.json', dict(status='causal_recorded_reference_replay',
                    case_id=case['id'], matched_frames=frame+1, complete_cases=len(report['rows'])))
            budget.tick()
        assert result['stop_reason'] == 'estimated_goal_reached'
        arrays['original439_observations'] = arrays['observations']
        arrays['observations'] = np.asarray(observations)
        target = traces / (case['id'] + '.npz')
        np.savez_compressed(target, **arrays)
        report['rows'].append(dict(case_id=case['id'], source_frames=len(observations),
            maximum439_input_difference=maximum_difference,
            complete_masks_actions_and_commands_reproduced=True,
            source_trace=str(source.relative_to(ROOT)), source_sha256=digest(source),
            trace_path=str(target.relative_to(ROOT)), trace_sha256=digest(target),
            reference_actor_path=str(reference_actor_path.relative_to(ROOT)),
            reference_actor_sha256=digest(reference_actor_path),
            original_safe_reference=row, reused_reference_not_new_success=True))
        report['source_frames'] += len(observations)
        report['resource'] = budget.statistics()
        write(output / 'RESULTS.json', report)
        print(json.dumps(report['rows'][-1]), flush=True)
        del runtime, arrays
        gc.collect()
    for name, expected in hashes.items():
        assert digest(ROOT / name) == expected, name
    budget.tick()
    report.update(completed=True, at=datetime.now(timezone.utc).isoformat(),
        input_hashes_verified=len(hashes), resource=budget.statistics())
    write(output / 'RESULTS.json', report)
    write(output / 'STATUS.json', dict(status='complete', complete_cases=len(report['rows'])))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--reference-actor', type=Path,
        default=ROOT / 'models/stop_feedback_20261015/recovery_followup_bc400_s231016/bc_portable/actor.pt')
    parser.add_argument('--case-id', action='append', default=[])
    parser.add_argument('--cpu-cores', type=float, default=.15)
    parser.add_argument('--reuse-verified-rows', type=Path)
    parser.add_argument('--reuse-source-unit')
    parser.add_argument('--reuse-source-invocation')
    args = parser.parse_args()
    try:
        main(args)
    except Exception as error:
        if args.output.is_dir():
            write(args.output / 'FAILURE.json', dict(error=repr(error), automatically_restarted=False))
        raise
