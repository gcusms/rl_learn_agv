"""Isolate route projection during exact sensor replay, without execution.

The diagnostic copy declares its route complete solely to measure what the
unchanged laser, dynamic, body and memory checks would otherwise permit.
These counterfactual commands are never sent to physics or installed in a
runtime. This is not a proposal to bypass route protection.
"""
import copy
from datetime import datetime
import hashlib
import json
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent
BASE=HERE.parent
OUT=BASE/'sensor01_route_projection_probe_20261017'


def write(path,value):
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
    temporary.replace(path)


def main():
    assert not OUT.exists();OUT.mkdir()
    prior=BASE/'sensor01_route_diagnosis_20261017'
    report=json.loads((prior/'RESULTS.json').read_text())
    assert report['completed'] and report['exact625_observation_frames']==83
    inputs=json.loads((prior/'PLAN.json').read_text())['inputs']
    for name,expected in inputs.items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==expected,name
    job_path=BASE/'v2_behavior_retention_validation_20261017/jobs/observed6_pre_ppo_holdout_sensor_01_candidate.json'
    job=json.loads(job_path.read_text());case=job['spec']
    inputs[str(Path(__file__).relative_to(ROOT))]=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    write(OUT/'PLAN.json',dict(inputs=inputs,no_counterfactual_command_execution=True,
        route_diagnostic_override_not_a_runtime_change=True,
        original_guard_all_action_parity_required_before_counterfactual=True,
        soft_cpu_cores=.35,hard_memory_mib=3072,rss_budget_mib=2800,
        goal_achieved=False,deployment_approved=False))
    import numpy as np
    import torch
    from neural_navigation.aligned_release_runtime import AlignedReleaseRuntime
    from neural_navigation.experiment_budget import ExperimentBudget
    from neural_navigation.masked_actions import COMMANDS
    torch.set_num_threads(1)
    budget=ExperimentBudget(.35,2800,OUT)
    trace=np.load(job['trace'],allow_pickle=False)
    runtime=AlignedReleaseRuntime(Path(job['bundle']),ROOT/'training'/case['scene']/'decision_map.yaml',
        case['start'],case['goal'],'shortcut')
    budget.tick();rows=[]
    for index,label in enumerate(trace['labels']):
        result=runtime.step(trace['scans'][index],trace['odometry'][index],trace['measured_velocity'][index],
            float(label[0]),capture_observation=True)
        np.testing.assert_array_equal(np.asarray(result['policy_observation'],np.float32),trace['observations'][index])
        np.testing.assert_allclose(result['command'],trace['commands'][index],rtol=0,atol=2e-8)
        if label[0]>=13.:
            core=runtime.core;stamp=float(core.elapsed)
            snapshot=copy.deepcopy(runtime.guard)
            points,uncertainty=snapshot.memory.update(core)
            class PreparedMemory:
                def update(self,unused):return points,uncertainty
                def reset(self):pass
            snapshot.memory=PreparedMemory()
            hypothetical=copy.copy(core)
            hypothetical.diagnostics=lambda:dict(core.diagnostics(),route_complete=True)
            checks=[]
            for action,command in enumerate(COMMANDS):
                executed,reason=copy.copy(snapshot).check(core,command,stamp,stamp)
                np.testing.assert_allclose(executed,runtime.context.guard_probe_frame['executed'][action],rtol=0,atol=2e-8)
                assert reason==runtime.context.guard_probe_frame['reasons'][action]
                counterfactual,cause=copy.copy(snapshot).check(hypothetical,command,stamp,stamp)
                checks.append(dict(action=action,command=command.tolist(),original_executed=executed.tolist(),
                    original_reason=reason,diagnostic_executed=counterfactual.tolist(),diagnostic_reason=cause,
                    diagnostic_unchanged_command=bool(np.allclose(counterfactual,command,rtol=0,atol=1e-9))))
            rows.append(dict(frame=index,time_s=stamp,original_route_complete=core.diagnostics()['route_complete'],
                original_legal=np.flatnonzero(trace['masks'][index]).tolist(),checks=checks,
                actual_chosen_action=int(trace['choices'][index])))
        budget.tick()
    for name,expected in inputs.items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==expected,name
    outcome=dict(completed=True,at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
        exact625_replay_frames=len(trace['labels']),original_12action_guard_probe_parity_frames=len(rows),rows=rows,
        changed_runtime_or_policy=False,counterfactual_commands_executed=False,
        original_scan_prediction_body4cm_memory_and_blind_checks_unchanged=True,
        new_physical_episodes=0,new_ppo_steps=0,new_optimizer_updates=0,
        resource=budget.statistics(),goal_achieved=False,deployment_approved=False)
    write(OUT/'RESULTS.json',outcome)
    print(json.dumps({k:v for k,v in outcome.items() if k!='rows'},ensure_ascii=False),flush=True)


if __name__=='__main__':
    try:main()
    except Exception as error:
        if OUT.exists():write(OUT/'FAILURE.json',dict(error=repr(error),no_auto_restart=True))
        raise
