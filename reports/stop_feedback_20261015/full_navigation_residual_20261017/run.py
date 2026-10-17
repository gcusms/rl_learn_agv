"""Serialize fit and its conditional full physical validation in fresh children."""
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys
from zoneinfo import ZoneInfo

ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent


def write(value):
    value['at']=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()
    p=HERE/'CYCLE_STATUS.json';tmp=p.with_suffix('.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');tmp.replace(p)


def main():
    admission=json.loads((HERE/'ADMISSION_TERMINAL.json').read_text())
    assert admission['completed'] and admission['admitted']
    for stage in ('fit','validate'):
        if stage=='validate':
            fit=json.loads((HERE/'FIT_RESULTS.json').read_text())
            if not all(fit[k] for k in ('completed','fit_passed','export_parity_passed','strict_runtime_admission_passed')):
                write(dict(status='complete',reason='Fit/export admission failed; no physical expansion',goal_achieved=False));return
        child=subprocess.Popen([sys.executable,'-u',str(HERE/(stage+'.py'))],cwd=ROOT)
        stat=Path('/proc')/str(child.pid)/'stat'
        tick=stat.read_text().rsplit(')',1)[1].split()[19] if stat.exists() else None
        write(dict(status='running',stage=stage,child_pid=child.pid,child_start_ticks=tick,
            one_model_or_physics_child_at_a_time=True,goal_achieved=False,deployment_approved=False))
        status=child.wait()
        if status:
            write(dict(status='failed',stage=stage,exit_code=status,no_auto_restart=True,
                goal_achieved=False,deployment_approved=False))
            raise RuntimeError(f'{stage} child exited with status {status}; inspect stage failure artifact')
    write(dict(status='complete',reason='Declared candidate cycle finished; inspect outcome gates',
        goal_achieved=False,deployment_approved=False))


if __name__=='__main__':
    try:main()
    except Exception as error:
        write(dict(status='failed',error=repr(error),no_auto_restart=True,goal_achieved=False));raise
