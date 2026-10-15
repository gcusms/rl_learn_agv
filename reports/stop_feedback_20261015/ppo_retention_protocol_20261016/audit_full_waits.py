"""Compare issued waiting only alongside the complete physical navigation outcomes."""
from pathlib import Path
import hashlib,json,os,sys
from experiments import audit_decision_waits as audit
from neural_navigation.experiment_budget import ExperimentBudget
ROOT=Path(__file__).resolve().parents[3];OUT=Path(__file__).resolve().parent
os.chdir(ROOT);budget=ExperimentBudget(.05,memory_mib=1800,directory=OUT)
original_measure=audit.measure
def measure(*args,**kwargs):
 result=original_measure(*args,**kwargs);budget.tick();return result
audit.measure=measure
sources={
 'parent_bc800':('proactive_augmented_bc800_gate_20261016','parentbc800_wait_audit_20261016'),
 'ppo8000':('proactive_augmented_ppo8000_navigation_20261016','ppo8000_wait_audit_20261016')}
reports={};hashes={}
for label,(source,output) in sources.items():
 path=Path('reports/stop_feedback_20261015')/source/'RESULTS.json'
 destination=Path('reports/stop_feedback_20261015')/output
 if not destination.exists():
  sys.argv=['audit_decision_waits','--evaluation',str(path),'--output',str(destination)]
  audit.main()
 reports[label]=json.loads((destination/'RESULTS.json').read_text())
 hashes[label]=hashlib.sha256(path.read_bytes()).hexdigest()
old={r['case_id']:r for r in reports['parent_bc800']['rows']}
new={r['case_id']:r for r in reports['ppo8000']['rows']}
assert old.keys()==new.keys()
common=[key for key in old if old[key]['safe_success'] and new[key]['safe_success']]
metrics=['unfinished_task_executed_idle_with_motion_s','only_stop_s','longest_unfinished_executed_wait_s','releases_without_further_motion']
rows=[dict(case_id=key,parent={k:old[key][k] for k in metrics},ppo={k:new[key][k] for k in metrics}) for key in common]
result=dict(completed=True,physical_tasks_per_model=18,source_sha256=hashes,summary={k:v['summary'] for k,v in reports.items()},common_safe_successes=len(common),common_success_waits=rows,lost_successes=[k for k in old if old[k]['safe_success'] and not new[k]['safe_success']],gained_successes=[k for k in old if not old[k]['safe_success'] and new[k]['safe_success']],resource=budget.statistics(),new_physical_runs=0,limitations=['A short wait ending in collision is not a successful recovery.','Only-stop intervals remain necessary-stop statistics; do not force motion from them.','Only the common-safe subset permits a waiting comparison without a different terminal outcome.'],goal_achieved=False,deployment_approved=False)
(OUT/'WAIT_COMPARISON.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(dict(completed=True,common_safe_successes=len(common),lost_successes=result['lost_successes'],gained_successes=result['gained_successes'],resource=result['resource']),indent=2))
