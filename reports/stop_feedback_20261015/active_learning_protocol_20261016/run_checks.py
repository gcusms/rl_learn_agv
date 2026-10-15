from pathlib import Path
import os, subprocess, sys
root=Path(__file__).resolve().parents[3]
os.chdir(root)
commands=[
 ['experiments.audit_task_geometry','--cases','reports/stop_feedback_20261015/proactive_augmented_protocol_20261016/fresh_cases.json','--hardware','models/stop_feedback_20261015/baseline/portable/hardware.json','--output','reports/stop_feedback_20261015/proactive_fresh_geometry_20261016','--cpu-cores','.05'],
 ['experiments.audit_active_recovery','--model-run','models/stop_feedback_20261015/proactive_augmented_ppo_8000_s231015','--through-steps','500','--output','reports/stop_feedback_20261015/proactive_actual500_audit_20261016','--cpu-cores','.05']]
for command in commands:
 print('Starting '+command[0],flush=True)
 subprocess.run([sys.executable,'-m',*command],check=True)
