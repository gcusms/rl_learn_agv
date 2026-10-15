"""Evaluate declared heading variants only if the original probes all fail."""
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[3]
PROTOCOL = Path(__file__).resolve().parent
BASE = Path('reports/stop_feedback_20261015')
ORIGINAL = BASE / 'proactive_augmented_protocol_20261016'
sys.path.insert(0, str(ROOT / ORIGINAL))
from after_validation import wait_for


def main():
    os.chdir(ROOT)
    plan = json.loads((PROTOCOL / 'PLAN.json').read_text())
    wait_for(plan['source_unit'], plan['source_invocation'])
    report = json.loads((BASE/'w11_early_left_probes_20261016/RESULTS.json').read_text())
    if not report['completed'] or len(report['rows']) != 2:
        raise ValueError('Both original physical outcomes are required')
    successful = [r['specification']['id'] for r in report['rows'] if r['safe_success']]
    decision = dict(at=datetime.now().astimezone().isoformat(),
        original_probe_safe=successful, extra_probes_started=not successful,
        main_experiment_unchanged=True, runtime_teacher_added=False,
        goal_achieved=False, deployment_approved=False)
    (PROTOCOL/'DECISION.json').write_text(json.dumps(decision, indent=2)+'\n')
    if successful:
        print(json.dumps(decision), flush=True)
        return
    command = [sys.executable, '-m', 'experiments.probe_recovery_prefix',
        '--bundle', 'models/stop_feedback_20261015/ppo_4000_s221015/initial_portable',
        '--prefix', str(ORIGINAL/'w11_t90_probe_prefix.json'),
        '--specifications', str(PROTOCOL/'probe_windows.json'),
        '--output', str(BASE/'w11_heading_refinement_probes_20261016'),
        '--cpu-cores', '.15']
    print(json.dumps(dict(starting=command)), flush=True)
    subprocess.run(command, check=True)


if __name__ == '__main__':
    main()
