"""Run two bounded legal-action diagnostics only when the main slot is free."""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import hashlib,json,os,subprocess,sys

ROOT=Path(__file__).resolve().parents[3]
OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'reports/stop_feedback_20261015/proactive_augmented_protocol_20261016'))
from after_validation import wait_for


def main():
 os.chdir(ROOT)
 plan=json.loads((OUT/'PLAN.json').read_text())
 def verify_inputs():
  for path,expected in plan['inputs'].items():
   if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=expected:raise ValueError('Diagnostic source changed: '+path)
 verify_inputs()
 print(json.dumps(dict(waiting_for=plan['dependency_unit'],invocation=plan['dependency_invocation'],no_simulation_started=True)),flush=True)
 wait_for(plan['dependency_unit'],plan['dependency_invocation'])
 verify_inputs()
 source=json.loads(Path(plan['dependency_completed_report']).read_text())
 if not source['completed'] or len(source['rows'])!=24:raise ValueError('All24 existing main validation rows must finish before reusing its CPU slot')
 state=dict(line.split('=',1) for line in subprocess.check_output(['systemctl','--user','show',plan['dependency_unit'],'-p','Id','-p','ActiveState','-p','Result','-p','InvocationID','-p','MainPID'],text=True).splitlines())
 if (state['ActiveState']!='inactive' or state['Result']!='success'
     or state['InvocationID'] not in ('',plan['dependency_invocation']) or state['MainPID']!='0'):
  raise RuntimeError('The existing main CPU slot has not been verifiably released')
 admission=dict(admitted=True,at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),dependency_state=state,completed_main_rows=24,soft_cpu_cores=.35,second_auxiliary_simulation_started=False,ppo_started=False,deployment_approved=False)
 (OUT/'ADMISSION.json').write_text(json.dumps(admission,indent=2)+'\n')
 summaries=[]
 for spec in plan['probes']:
  output=Path(spec['output'])
  command=[sys.executable,'-m','experiments.probe_recovery_prefix','--bundle',plan['reference_bundle'],'--prefix',str(OUT/'prefix.json'),'--specifications',str(OUT/spec['specifications']),'--output',str(output),'--cpu-cores','.35']
  print(json.dumps(dict(starting=command)),flush=True)
  subprocess.run(command,check=True)
  report=json.loads((output/'RESULTS.json').read_text())
  if not report['completed'] or len(report['rows'])!=1 or not report['not_a_trained_policy']:raise ValueError('One complete diagnostic outcome required')
  row=report['rows'][0]
  if not row['prefix_sensor_and_physical_parity_verified']:raise ValueError('Physical initialization parity missing')
  summary={k:row[k] for k in ['safe_success','collisions','physical_guards','false_arrival','timeout','post_release_seconds','final_xy_error_m','trace_sha256']}
  summary.update(id=spec['id'],output=str(output),requested_frames=len(row['intervention']),legal_intervention_frames=sum(x['requested_choice_legal'] for x in row['intervention']),diagnostic_only=True)
  summaries.append(summary)
  if row['collisions'] or row['physical_guards'] or row['false_arrival']:
   print(json.dumps(dict(remaining_probes_skipped='Unsafe complete diagnostic; no second trial under the same plan')),flush=True)
   break
 final=dict(completed=True,all_predeclared_probes_executed=len(summaries)==len(plan['probes']),rows=summaries,physical_diagnostics_only=True,trained_policy_score=False,new_ppo_steps=0,goal_achieved=False,deployment_approved=False)
 (OUT/'RESULTS.json').write_text(json.dumps(final,indent=2)+'\n')
 print(json.dumps(final),flush=True)


if __name__=='__main__':main()
