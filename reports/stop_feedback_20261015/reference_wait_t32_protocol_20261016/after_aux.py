"""Use the existing auxiliary slot for one original-guard BC continuation."""
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from zoneinfo import ZoneInfo


ROOT=Path(__file__).resolve().parents[3]
OUT=Path(__file__).resolve().parent


def stamp():
    return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()


def write(name,data):
    p=OUT/name;t=p.with_suffix(p.suffix+'.tmp')
    t.write_text(json.dumps(data,indent=2)+'\n');t.replace(p)


def verify(plan):
    for name,expected in plan['inputs'].items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=expected:
            raise ValueError('Frozen counterfactual input changed: '+name)


def main():
    os.chdir(ROOT)
    plan=json.loads((OUT/'PLAN.json').read_text());verify(plan)
    while True:
        current=dict(v.split('=',1) for v in subprocess.check_output([
            'systemctl','--user','show',plan['dependency_unit'],'-p','MainPID',
            '-p','ActiveState','-p','Result','-p','InvocationID'],text=True).splitlines())
        if current['InvocationID'] not in ('',plan['dependency_invocation']):
            raise ValueError('The exact preceding diagnostic was replaced')
        if current['ActiveState'] in ('inactive','failed'):
            if (current['MainPID']!='0' or current['ActiveState']!='inactive'
                    or current['Result']!='success'):
                raise ValueError('Preceding diagnostic did not complete successfully')
            break
        write('STATUS.json',dict(status='waiting',at=stamp(),dependency=current,
            physical_simulation_started=False));time.sleep(5)
    verify(plan)
    previous=json.loads((ROOT/plan['dependency_report']).read_text())
    if (not previous['completed'] or len(previous['candidate_rows'])!=6
            or len(previous['reference_rows'])!=6
            or not previous['original_admission_failed']):
        raise ValueError('Complete preceding diagnosis and original rejection required')
    write('ADMISSION.json',dict(at=stamp(),diagnostic_only=True,
        source_baseline_reused_not_recounted=True,model_only_after32s=True,
        original_protection_unchanged=True,new_continuations=1))
    write('STATUS.json',dict(status='running',at=stamp(),stage='one_original_guard_bc_continuation'))
    command=[sys.executable,'-m','experiments.evaluate_feedback_recovery',
        '--bundle',plan['bundle'],'--specifications',str(OUT/'prefix.json'),
        '--output',plan['output'],'--label','reference_bc_control_wait_t32','--cpu-cores','.15']
    print(json.dumps(dict(starting=command)),flush=True);subprocess.run(command,check=True)
    result=json.loads((ROOT/plan['output']/'RESULTS.json').read_text())
    if (not result['completed'] or len(result['rows'])!=1
            or not result['rows'][0]['prefix_sensor_and_physical_parity_verified']):
        raise ValueError('One complete sensor-and-physical-parity verified continuation required')
    write('RESULTS.json',dict(completed=True,at=stamp(),selection=plan['selection'],
        continuation=result['rows'][0],source_control_timeout=plan['source_baseline_complete_timeout'],
        different_episode_preceding_actions_not_attributed_to_bc=True,
        source_baseline_reused_not_counted_again=True,
        original_guard_unchanged=True,new_physical_continuations=1,new_ppo_steps=0,
        no_automatic_expansion_training_or_promotion=True,goal_achieved=False,deployment_approved=False))
    write('STATUS.json',dict(status='complete',at=stamp(),automatically_promoted=False))


if __name__=='__main__':
    try:main()
    except Exception as error:
        write('STATUS.json',dict(status='failed',at=stamp(),error=repr(error),automatically_restarted=False))
        raise
