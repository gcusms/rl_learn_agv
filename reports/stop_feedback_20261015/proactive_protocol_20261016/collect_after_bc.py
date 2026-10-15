"""Wait for the exact BC validation job, then collect the frozen training probes."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[3]
PROTOCOL = Path(__file__).resolve().parent


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--validation-invocation',required=True)
    args=parser.parse_args()
    os.chdir(ROOT)
    status=PROTOCOL/'COLLECT_STATUS.json'
    status.write_text(json.dumps(dict(phase='waiting_for_bc_validation',training_started=False),indent=2)+'\n')
    while True:
        state=dict(line.split('=',1) for line in subprocess.check_output([
            'systemctl','--user','show','rl-proactive-bc-navigation-20261016.service',
            '-p','InvocationID','-p','ActiveState','-p','Result'],text=True).splitlines())
        if state['InvocationID'] and state['InvocationID']!=args.validation_invocation:
            raise RuntimeError('BC validation invocation changed')
        if state['ActiveState'] in ('inactive','failed'):
            if state['ActiveState']!='inactive' or state['Result']!='success':
                raise RuntimeError('BC validation process failed; do not rerun automatically')
            break
        time.sleep(5)
    status.write_text(json.dumps(dict(phase='collecting_training_probes',training_started=False),indent=2)+'\n')
    command=[sys.executable,'-m','experiments.generate_proactive_demonstrations',
        '--bundle','models/stop_feedback_20261015/ppo_4000_s221015/initial_portable',
        '--cases',str(PROTOCOL/'ppo_training_cases.json'),
        '--output','reports/stop_feedback_20261015/proactive_training_probes_20261016',
        '--cpu-cores','.35']
    print(json.dumps(dict(starting=command)),flush=True)
    subprocess.run(command,check=True)
    status.write_text(json.dumps(dict(phase='collection_complete_pending_review',training_started=False),indent=2)+'\n')


if __name__=='__main__':
    main()
