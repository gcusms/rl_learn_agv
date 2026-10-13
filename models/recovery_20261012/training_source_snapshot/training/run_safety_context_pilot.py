"""Train, export and independently evaluate the context-aware PPO candidate."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--parent', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--steps', type=int, default=100000)
    p.add_argument('--envs', type=int, default=4)
    p.add_argument('--seed', type=int, default=92001)
    p.add_argument('--failure-pool',type=Path)
    p.add_argument('--resume-policy',type=Path)
    p.add_argument('--manifest', type=Path, action='append', required=True)
    a = p.parse_args()
    project = Path(__file__).resolve().parents[1]
    a.output = a.output.resolve()
    if a.output.exists(): p.error('Fresh output required')
    if not 1 <= a.envs <= 4: p.error('Use 1-4 workers')
    a.output.mkdir(parents=True)
    candidate = a.output/'candidate'
    portable = a.output/'portable'
    commands = [
        [sys.executable, str(project/'training/train_safety_context.py'), '--parent', str(a.parent.resolve()),
         '--output', str(candidate), '--steps', str(a.steps), '--envs', str(a.envs), '--seed', str(a.seed)],
        [sys.executable, str(project/'training/export_navigation_actor.py'), '--policy', str(candidate/'policy.zip'),
         '--components', str(candidate), '--output', str(portable)]]
    if a.failure_pool:commands[0].extend(['--failure-pool',str(a.failure_pool.resolve())])
    if a.resume_policy:commands[0].extend(['--resume-policy',str(a.resume_policy.resolve())])
    for i, manifest in enumerate(a.manifest):
        commands.append([sys.executable, str(project/'training/evaluate_sensor_navigation.py'), '--bundle', str(portable),
                         '--manifest', str(manifest.resolve()), '--workers', str(a.envs),
                         '--output', str(a.output/f'runtime_{i}.json'), '--trace-dir', str(a.output/f'traces_{i}')])
    status = dict(status='running', requested_steps=a.steps, seed=a.seed, commands=commands,
                  deployment_approved=False, final_acceptance_used=False, started_unix=time.time())
    def save(): (a.output/'status.json').write_text(json.dumps(status, indent=2))
    save()
    try:
        for index, command in enumerate(commands):
            status['phase'] = 'training' if index == 0 else 'export' if index == 1 else f'runtime_{index-2}'
            save()
            with (a.output/f'{status["phase"]}.log').open('w') as log:
                subprocess.run(command, cwd=project, stdout=log, stderr=subprocess.STDOUT, check=True)
        reports = [json.loads((a.output/f'runtime_{i}.json').read_text()) for i in range(len(a.manifest))]
        status.update(status='complete', development_gate_passed=all(r['completed'] and r['safe_success']==r['tasks'] for r in reports),
                      summaries=[dict(tasks=r['tasks'], safe_success=r['safe_success'], collisions=r['collisions']) for r in reports],
                      policy_sha256=hashlib.sha256((candidate/'policy.zip').read_bytes()).hexdigest(),
                      finished_unix=time.time())
        save()
        print(json.dumps(status), flush=True)
    except Exception as error:
        status.update(status='failed', error=str(error)); save(); raise


if __name__ == '__main__': main()
