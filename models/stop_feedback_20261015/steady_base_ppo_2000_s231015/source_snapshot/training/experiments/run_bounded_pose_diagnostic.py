"""Sequential training-failure and development replay, no learning/promotion."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--directory',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--training-log',type=Path,required=True)
    p.add_argument('--step',type=int,required=True)
    a=p.parse_args();root=a.directory.resolve();candidate=root/'candidate'
    if candidate.exists():p.error('Fresh candidate directory required')
    candidate.mkdir()
    shutil.copy2(a.checkpoint,candidate/'policy.zip')
    for name in ('hardware.json','localizer.pt','router.pt'):
        shutil.copy2(root/'portable'/name,candidate/name)
    status=dict(status='running',phase='training_failure_replay',started_unix=time.time(),
        deployment_approved=False,training_steps=0)
    def save():(root/'status.json').write_text(json.dumps(status,indent=2))
    save()
    try:
        subprocess.run([sys.executable,'training/experiments/trace_masked_training.py',
            '--policy',str(candidate/'policy.zip'),'--training-log',str(a.training_log.resolve()),
            '--step',str(a.step),'--sensor-details-until','30',
            '--output',str(root/'training_replay')],check=True)
        status['phase']='development_six';save()
        subprocess.run([sys.executable,'training/evaluate_sensor_navigation.py',
            '--bundle',str(root/'portable'),'--manifest',str(root/'cases.json'),
            '--workers','1','--output',str(root/'runtime.json'),
            '--trace-dir',str(root/'traces')],check=True)
        report=json.loads((root/'runtime.json').read_text())
        status.update(status='complete',finished_unix=time.time(),
            summary={k:report[k] for k in ('tasks','safe_success','collisions')})
        save()
    except Exception as error:
        status.update(status='failed',error=str(error));save();raise


if __name__=='__main__':main()
