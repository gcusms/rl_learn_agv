"""Run a training-task diagnostic after a live pilot exports its final actor.

    No restart, learning, model promotion or acceptance-set access.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--unit',required=True)
    p.add_argument('--training-log',type=Path,required=True)
    p.add_argument('--step',type=int,required=True)
    p.add_argument('--expected-steps',type=int,required=True)
    a=p.parse_args();root=a.run.resolve();output=root/'diagnostics/final_training_replay'
    status_path=root/'diagnostics/final_replay_watch.json';status_path.parent.mkdir(exist_ok=True)
    if output.exists():p.error('Fresh final replay output required')
    deadline=time.monotonic()+3600
    status=dict(status='waiting_for_verified_final_export',unit=a.unit,started_unix=time.time())
    def save():status_path.write_text(json.dumps(status,indent=2))
    save()
    try:
        while True:
            manifest=root/'portable/manifest.json';config=root/'candidate/config.json';policy=root/'candidate/policy.zip'
            if manifest.exists() and config.exists() and policy.exists():
                exported=json.loads(manifest.read_text());record=json.loads(config.read_text())
                digest=hashlib.sha256(policy.read_bytes()).hexdigest()
                if record.get('completed_steps')==a.expected_steps and exported.get('policy_sha256')==digest:
                    break
            live=subprocess.run(['systemctl','--user','show',a.unit,'--property=MainPID','--value'],
                capture_output=True,text=True,check=True).stdout.strip()
            if not live or live=='0':raise RuntimeError('Pilot is terminal without a verified final export; no restart attempted')
            if time.monotonic()>deadline:raise RuntimeError('Timed out waiting for export; pilot left untouched')
            time.sleep(20)
        status.update(status='running',policy_sha256=digest);save()
        subprocess.run([sys.executable,'training/experiments/trace_masked_training.py',
            '--policy',str(policy),'--training-log',str(a.training_log.resolve()),
            '--step',str(a.step),'--sensor-details-until','30','--output',str(output)],check=True)
        summary=json.loads((output/'SUMMARY.json').read_text())
        if summary['trials'][0]['policy_sha256']!=digest:raise ValueError('Final replay actor changed')
        status.update(status='complete',result=summary['trials'][0],finished_unix=time.time());save()
    except Exception as error:
        status.update(status='failed',error=str(error),finished_unix=time.time());save();raise


if __name__=='__main__':main()
