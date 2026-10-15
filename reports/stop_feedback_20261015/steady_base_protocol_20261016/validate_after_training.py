"""Finish this frozen trial without restarting or replacing its dependency."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import zipfile


ROOT = Path(__file__).resolve().parents[3]
PROTOCOL = Path(__file__).resolve().parent
BASE = Path('reports/stop_feedback_20261015')
MODEL = Path('models/stop_feedback_20261015/steady_base_ppo_2000_s231015')
OLD_MODEL = Path('models/stop_feedback_20261015/prefix_ppo_2000_s231015')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def wait_for(unit, invocation):
    while True:
        state = dict(line.split('=', 1) for line in subprocess.check_output(
            ['systemctl', '--user', 'show', unit, '--property=InvocationID,ActiveState,Result'],
            text=True).splitlines())
        if state['InvocationID'] and state['InvocationID'] != invocation:
            raise RuntimeError('Dependency invocation changed: '+unit)
        if state['ActiveState'] in ('inactive', 'failed'):
            if state['ActiveState'] != 'inactive' or state['Result'] != 'success':
                raise RuntimeError('Dependency failed: '+unit)
            return
        time.sleep(5)


def run(module, *arguments):
    command = [sys.executable, '-m', module, *map(str, arguments)]
    print(json.dumps(dict(starting=command)), flush=True)
    subprocess.run(command, check=True)


def check_training():
    status = json.loads((MODEL/'status.json').read_text())
    if (status['status'] != 'complete' or status['additional_ppo_steps'] != 2000
            or status['recovery_contract']['schema'] != 'stop_feedback_recovery_steady_base_v1'):
        raise ValueError('Complete versioned 2k experiment required')


def summarize():
    import numpy as np
    import torch
    from experiments.paired_navigation_gate import compare
    from neural_navigation.masked_actions import ACTIONS
    torch.set_num_threads(1)
    reference = BASE/'prefix_recovery_threeway_20261016'
    candidate = BASE/'steady_base_navigation_2000_20261016'
    old = json.loads((reference/'RESULTS.json').read_text())
    new = json.loads((candidate/'RESULTS.json').read_text())
    if not old['completed'] or not new['completed'] or len(new['rows']) != 14:
        raise ValueError('Complete paired development evidence required')
    if json.loads((reference/'cases.json').read_text()) != json.loads((candidate/'cases.json').read_text()):
        raise ValueError('Cases changed')
    source_before = json.loads((reference/'PROVENANCE.json').read_text())['source_sha256']
    source_after = json.loads((candidate/'PROVENANCE.json').read_text())['source_sha256']
    physical = [name for name in source_before if name.startswith(
        ('training/navrl/', 'training/neural_navigation/'))]
    if any(source_before[name] != source_after.get(name) for name in physical):
        raise ValueError('Physical runtime changed; fresh reference executions are required')
    with zipfile.ZipFile(OLD_MODEL/'policy_bc.zip') as a, zipfile.ZipFile(MODEL/'policy_bc.zip') as b:
        if a.read('policy.pth') != b.read('policy.pth'):
            raise ValueError('BC initialization changed; fresh BC executions are required')
    actor = torch.jit.load(str(MODEL/'bc_portable/actor.pt'), map_location='cpu').eval()
    previous = {label: [r for r in old['rows'] if r['variant'] == label]
        for label in ['frozen_reference', 'bc_only', 'recovery_ppo']}
    replay, changes = [], []
    for row in previous['bc_only']:
        name = row['case_id']
        path = reference/(name+'_bc_only.npz')
        with np.load(path, allow_pickle=False) as trace:
            # Check the complete exported directional actor, including docking,
            # on every actual prior BC decision. This is reused evidence.
            with torch.no_grad():
                actual = actor(torch.from_numpy(trace['observations'])).numpy()
            np.testing.assert_array_equal(actual, ACTIONS[trace['choices']])
            replay.append(dict(case_id=name,frames=len(actual),all_actions_equal=True,
                source_trace_sha256=digest(path),not_new_independent_execution=True))
            with np.load(candidate/(name+'_steady_base_ppo.npz'), allow_pickle=False) as final:
                checks = {key: bool(np.array_equal(trace[key], final[key], equal_nan=True))
                    for key in ['choices', 'commands', 'labels', 'scans', 'observations']}
                changes.append(dict(case_id=name,arrays_equal=checks,all_equal=all(checks.values())))
    def metrics(rows):
        return dict(tasks=len(rows),safe_successes=sum(r['safe_success'] for r in rows),
            collisions=sum(r['collisions'] for r in rows),timeouts=sum(r['timeout'] for r in rows),
            false_arrivals=sum(r['false_arrival'] for r in rows))
    result = dict(completed=True, deployment_approved=False,
        reference_report_sha256=digest(reference/'RESULTS.json'),
        candidate_report_sha256=digest(candidate/'RESULTS.json'),
        final_policy_sha256=digest(MODEL/'policy.zip'),
        reused_reference_metrics={k:metrics(v) for k,v in previous.items()},
        new_candidate_metrics=metrics(new['rows']),
        paired_gates={k:compare(v,new['rows']) for k,v in previous.items()},
        physical_runtime_sources_unchanged=len(physical),bc_weights_bytewise_equal=True,
        new_bc_export_full_reference_replay=replay,bc_to_final_trajectory_comparison=changes,
        all_bc_to_final_trajectories_equal=all(r['all_equal'] for r in changes),
        limitations=['Reference and BC mission results are reused, not new executions.',
            'Fixed reused development cases are not unseen-map or real deployment acceptance.',
            'Fewer random frozen-base deviations or parameter changes alone do not prove learned recovery.'])
    (candidate/'SUMMARY.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=['primary', 'auxiliary'], required=True)
    parser.add_argument('--training-invocation', required=True)
    parser.add_argument('--previous-auxiliary-invocation')
    args = parser.parse_args()
    os.chdir(ROOT)
    if args.phase == 'auxiliary':
        if not args.previous_auxiliary_invocation:
            parser.error('Exact preceding auxiliary invocation required')
        wait_for('rl-training-stall-probes-20261016.service', args.previous_auxiliary_invocation)
        prior = json.loads((BASE/'recovery_holdout_ppo_20261016/RESULTS.json').read_text())
        if (not prior['completed'] or len(prior['rows']) != 3
                or prior['bundle_manifest_sha256'] != digest(OLD_MODEL/'portable/manifest.json')):
            raise ValueError('Preceding complete held-out conditions required')
        run('experiments.probe_recovery_prefix',
            '--bundle','models/stop_feedback_20261015/ppo_4000_s221015/initial_portable',
            '--prefix',BASE/'w11_early_rotation_protocol_20261016/prefix.json',
            '--specifications',BASE/'w11_early_rotation_protocol_20261016/windows.json',
            '--output',BASE/'w11_early_rotation_probe_20261016','--cpu-cores','.15')
    wait_for('rl-steady-base-ppo-20261016.service', args.training_invocation)
    check_training()
    if args.phase == 'primary':
        run('experiments.audit_prefix_recovery_rollouts','--model-run',MODEL,
            '--output',BASE/'steady_base_sampling_audit_2000_20261016')
        run('experiments.analyze_recovery_learning','--model-run',MODEL,
            '--output',BASE/'steady_base_learning_analysis_2000_20261016','--cpu-cores','.35')
        run('experiments.evaluate_feedback_recovery','--bundle',MODEL/'portable',
            '--specifications',BASE/'frozen_recovery_cases.json',
            '--output',BASE/'steady_base_prefix_recovery_2000_20261016',
            '--label','steady_base_ppo','--cpu-cores','.35')
        run('experiments.evaluate_docking_navigation','--candidate',MODEL/'portable',
            '--cases',BASE/'feedback_development_cases.json',
            '--output',BASE/'steady_base_navigation_2000_20261016',
            '--candidate-name','steady_base_ppo','--cpu-cores','.35')
        summarize()
    else:
        run('experiments.evaluate_feedback_recovery','--bundle',MODEL/'portable',
            '--specifications',BASE/'recovery_holdout_reference_20261016/prefixes.json',
            '--output',BASE/'steady_base_holdout_2000_20261016',
            '--label','steady_base_ppo','--cpu-cores','.15')
    print(json.dumps(dict(phase=args.phase,completed=True)),flush=True)


if __name__ == '__main__':
    main()
