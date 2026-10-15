"""Pair the recovery with its exact parent and test all frozen extra conditions."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[3]
PROTOCOL = Path(__file__).resolve().parent
BASE = Path('reports/stop_feedback_20261015')
PARENT = Path('models/stop_feedback_20261015/proactive_augmented_bc_800_s231015')
MODEL = Path('models/stop_feedback_20261015/w11_recovery_bc800_s231015')
sys.path.insert(0, str(ROOT/BASE/'proactive_augmented_protocol_20261016'))
from after_validation import wait_for


def run(module, *arguments):
    command = [sys.executable, '-m', module, *map(str, arguments)]
    print(json.dumps(dict(starting=command)), flush=True)
    subprocess.run(command, check=True)


def read_completed(path, expected):
    report = json.loads(path.read_text())
    if not report['completed'] or len(report['rows']) != expected:
        raise ValueError('All complete physical outcomes required: '+str(path))
    return report


def main():
    os.chdir(ROOT)
    plan = json.loads((PROTOCOL/'PAIRED_PLAN.json').read_text())
    wait_for(plan['dependency_unit'], plan['dependency_invocation'])
    early = read_completed(BASE/'w11_recovery_bc800_early_20261016/RESULTS.json', 1)
    new_gate = read_completed(BASE/'w11_recovery_bc800_navigation_20261016/RESULTS.json', 18)
    prefixes = read_completed(BASE/'w11_recovery_bc800_prefixes_20261016/RESULTS.json', 2)
    related = read_completed(BASE/'w11_recovery_bc800_w11_20261016/RESULTS.json', 3)
    old_gate_path = BASE/'proactive_augmented_bc800_gate_20261016/RESULTS.json'
    if hashlib.sha256(old_gate_path.read_bytes()).hexdigest() != plan['parent_full_reset_report_sha256']:
        raise ValueError('The frozen paired parent evidence changed')
    old_gate = read_completed(old_gate_path, 18)
    run('experiments.evaluate_feedback_recovery', '--bundle', PARENT/'bc_portable',
        '--specifications', PROTOCOL/'w11_t90_evaluation.json',
        '--output', BASE/'w11_recovery_parent800_early_20261016',
        '--label', 'parent_bc800', '--cpu-cores', '.15')
    run('experiments.evaluate_docking_navigation', '--candidate', MODEL/'bc_portable',
        '--baseline', PARENT/'bc_portable', '--baseline-name', 'parent_bc800',
        '--cases', PROTOCOL/'fresh_cases.json',
        '--output', BASE/'w11_recovery_bc800_fresh_20261016',
        '--candidate-name', 'w11_augmented_bc800', '--cpu-cores', '.15')
    parent_early = read_completed(BASE/'w11_recovery_parent800_early_20261016/RESULTS.json', 1)
    fresh = read_completed(BASE/'w11_recovery_bc800_fresh_20261016/RESULTS.json', 24)
    from experiments.paired_navigation_gate import compare
    old_extra = [r for r in fresh['rows'] if r['variant'] == 'parent_bc800']
    new_extra = [r for r in fresh['rows'] if r['variant'] == 'w11_augmented_bc800']
    if len(old_extra) != 12 or len(new_extra) != 12:
        raise ValueError('All twelve extra paired conditions required')
    result = dict(completed=True, additional_ppo_steps=0,
        parent_to_candidate_full_reset=compare(old_gate['rows'], new_gate['rows']),
        parent_to_candidate_fresh=compare(old_extra, new_extra),
        same_90s_physical_prefix=dict(parent=parent_early['rows'][0], candidate=early['rows'][0]),
        original_prefixes=prefixes['rows'], correlated_w11_histories=related['rows'],
        parent_full_reset_reused_not_reexecuted=True,
        candidate_90s_continuation_reused_not_independent_repeat=True,
        original_90s_related_repeat_not_counted_as_additional_success=True,
        fresh_conditions_predeclared_not_unseen_maps=True,
        automatically_promoted=False, goal_achieved=False, deployment_approved=False)
    (PROTOCOL/'PAIRED_RESULTS.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(completed=True, paired_results=str(PROTOCOL/'PAIRED_RESULTS.json'),
                         automatically_promoted=False)), flush=True)


if __name__ == '__main__':
    main()
