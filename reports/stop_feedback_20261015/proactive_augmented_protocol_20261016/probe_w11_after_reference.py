"""Use only the auxiliary budget after its exact existing comparison ends."""
import argparse
import os
from after_validation import BASE, PROTOCOL, ROOT, run, wait_for


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--comparison-invocation',required=True)
    args=parser.parse_args()
    os.chdir(ROOT)
    wait_for('rl-proactive-bc-w11-20261016.service',args.comparison_invocation)
    run('experiments.probe_recovery_prefix',
        '--bundle','models/stop_feedback_20261015/ppo_4000_s221015/initial_portable',
        '--prefix',PROTOCOL/'w11_t90_probe_prefix.json',
        '--specifications',PROTOCOL/'w11_t90_probe_windows.json',
        '--output',BASE/'w11_early_left_probes_20261016','--cpu-cores','.15')


if __name__=='__main__':
    main()
