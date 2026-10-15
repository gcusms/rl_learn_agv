"""Compare a second recorded failure family without calling scripts policy wins."""
import argparse
import json
import os
from pathlib import Path

from validate_bc import BASE, MODEL, PROTOCOL, ROOT, run, wait_for


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--holdout-invocation',required=True)
    args=parser.parse_args()
    os.chdir(ROOT)
    wait_for(args.holdout_invocation,'rl-proactive-bc-holdout-20261016.service')
    result=json.loads((BASE/'proactive_bc_full_reset_holdout_20261016/RESULTS.json').read_text())
    if not result['completed'] or len(result['rows'])!=8:
        raise ValueError('All four paired reset conditions, including collisions, must be recorded')
    run('experiments.evaluate_feedback_recovery','--bundle',MODEL/'bc_portable',
        '--specifications',PROTOCOL/'w11_prefixes.json',
        '--output',BASE/'proactive_bc_w11_recovery_20261016',
        '--label','proactive_bc','--cpu-cores','.15')
    run('experiments.evaluate_feedback_recovery',
        '--bundle','models/stop_feedback_20261015/ppo_4000_s221015/initial_portable',
        '--specifications',PROTOCOL/'w11_early_reference_prefixes.json',
        '--output',BASE/'proactive_w11_early_reference_20261016',
        '--label','frozen_reference','--cpu-cores','.15')
    print(json.dumps(dict(w11_comparison_executions_complete=True)),flush=True)


if __name__=='__main__':
    main()
