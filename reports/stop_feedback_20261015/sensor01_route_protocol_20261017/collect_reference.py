"""Try one admissible demonstration from a successful sibling sensor case.

This is an offline reference-sequence teacher, not autonomous model success.
Every target-frame motion must pass the original target runtime mask. Abort
before physics if any reference motion is inadmissible. No guard relaxation.
"""
from datetime import datetime
import hashlib
import json
from pathlib import Path
from unittest.mock import patch
from zoneinfo import ZoneInfo

ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent
BASE=HERE.parent
OUT=BASE/'sensor01_reference_demonstration_20261017'


def write(path,value):
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
    temporary.replace(path)


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def read(path):return json.loads(path.read_text())


class InadmissibleReference(RuntimeError):pass


def main():
    assert not OUT.exists();OUT.mkdir()
    validation=BASE/'v2_behavior_retention_validation_20261017'
    job_path=validation/'jobs/observed6_pre_ppo_holdout_sensor_01_candidate.json'
    source_job_path=validation/'jobs/observed6_pre_ppo_holdout_sensor_00_candidate.json'
    job=read(job_path);source_job=read(source_job_path);bundle=Path(job['bundle'])
    case=job['spec'];source_case=source_job['spec']
    for key in set(case)|set(source_case):
        if key not in ('id','seed','extra_scan_dropout'):
            assert case.get(key)==source_case.get(key),key
    assert read(BASE/'sensor01_route_projection_probe_20261017/RESULTS.json')['completed']
    parent=read(validation/'RESULTS.json');assert parent['completed']
    source_row=next(r for r in parent['groups']['observed6']['rows'] if r['case_id']==source_case['id'])
    assert source_row['safe_success'] and not source_row['collisions']
    source_trace=Path(source_job['trace']);assert digest(source_trace)==source_row['trace_sha256']
    manifest=read(bundle/'manifest.json')
    paths=[Path(__file__),job_path,source_job_path,source_trace,validation/'RESULTS.json',bundle/'manifest.json']
    paths += [bundle/name for name in manifest['files']]
    for folder in ('neural_navigation','navrl','experiments'):
        paths+=list((ROOT/'training'/folder).rglob('*.py'))
    paths += [p for p in (ROOT/'training'/case['scene']).iterdir() if p.is_file()]
    hashes={str(p.relative_to(ROOT)):digest(p) for p in set(paths)}
    write(OUT/'PLAN.json',dict(at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),inputs=hashes,
        target_case=case,source_case=source_case,source_row=source_row,
        maximum_new_physical_episodes=1,reference_actual_executed_commands=True,
        teacher_uses_reference_sequence_and_current_target_mask_not_physics_truth=True,
        abort_before_execution_on_any_inadmissible_reference=True,
        after_reference_end_use_unchanged_nominal_actor=True,
        actual_execution_updates_all_history_and_feedback=True,
        positive_training_requires_complete_safe_episode_and_separate_support_admission=True,
        no_policy_safety_memory_routing_action_body4cm_or_deadline_changes=True,
        development_demonstration_not_independent_model_test=True,
        new_ppo_steps=0,new_optimizer_updates=0,soft_cpu_cores=.35,
        rss_budget_mib=2800,hard_service_memory_mib=3072,
        goal_achieved=False,deployment_approved=False))
    import numpy as np
    import torch
    from experiments.evaluate_docking_navigation import evaluate
    from neural_navigation.aligned_release_runtime import AlignedReleaseRuntime
    from neural_navigation.experiment_budget import ExperimentBudget
    from neural_navigation.masked_actions import ACTIONS,COMMANDS,masks_from_observation
    torch.set_num_threads(1)
    budget=ExperimentBudget(.35,2800,OUT)
    source=np.load(source_trace,allow_pickle=False)
    distance=np.max(abs(source['commands'][:,None,:]-COMMANDS[None,:,:]),axis=2)
    reference=distance.argmin(1);assert np.max(distance[np.arange(len(reference)),reference])<2e-8
    created=[]

    class TeacherPolicy:
        def __init__(self,actor,core):self.actor=actor;self.core=core;self.entries=[];self.observations=[]
        def predict(self,observation):
            observation=np.asarray(observation,np.float32)
            nominal_action=self.actor.predict(observation)
            difference=np.max(abs(ACTIONS-nominal_action),axis=1)
            nominal=int(difference.argmin());assert difference[nominal]<2e-6
            index=len(self.entries);desired=int(reference[index]) if index<len(reference) else nominal
            mask=masks_from_observation(observation[:282]);assert mask[nominal]
            with torch.no_grad():
                supported=bool(self.actor.actor.support(torch.from_numpy(observation[None])).item())
            entry=dict(frame=index,time_s=float(self.core.elapsed),nominal=nominal,reference=desired,
                admissible=bool(mask[desired]),trainable_support=supported,
                teacher_changed_choice=desired!=nominal,source_sequence_active=index<len(reference),
                legal_actions=np.flatnonzero(mask).tolist())
            self.entries.append(entry);self.observations.append(observation.copy())
            if not mask[desired]:
                write(OUT/'BLOCKED_REFERENCE.json',entry)
                raise InadmissibleReference(f'Reference action {desired} blocked at target frame {index}')
            return ACTIONS[desired].copy()

    class Runtime(AlignedReleaseRuntime):
        def __init__(self,*args,**kwargs):
            super().__init__(*args,**kwargs)
            teacher=TeacherPolicy(self.policy.policy,self.core)
            self.policy.policy=teacher;created.append(teacher)

    write(OUT/'STATUS.json',dict(status='offline_reference_episode',case_id=case['id']))
    outcome=None;trace=None;blocked=None
    try:
        with patch('neural_navigation.goal_pose_runtime.resolve_runtime',return_value=Runtime):
            outcome,trace=evaluate(case,bundle,budget)
    except InadmissibleReference as error:
        blocked=str(error)
    assert len(created)==1;teacher=created[0];entries=teacher.entries
    full_safe=bool(outcome and outcome['safe_success'] and not any(outcome[k] for k in ('collisions','false_arrival','physical_guards','timeout')))
    artifact=OUT/'demonstration.npz'
    if trace is not None:
        assert len(entries)==len(trace['choices'])
        np.testing.assert_array_equal(trace['choices'],[e['reference'] for e in entries])
        trace.update(nominal_choices=np.array([e['nominal'] for e in entries]),
            trainable_support=np.array([e['trainable_support'] for e in entries]),
            teacher_changed_choice=np.array([e['teacher_changed_choice'] for e in entries]))
        np.savez_compressed(artifact,**trace)
        outcome.update(trace_path=str(artifact.relative_to(ROOT)),trace_sha256=digest(artifact))
    else:
        artifact=OUT/'inadmissible_partial_observations.npz'
        np.savez_compressed(artifact,observations=np.array(teacher.observations))
    write(OUT/'TEACHER_CHOICES.json',entries)
    changed=[e for e in entries if e['teacher_changed_choice']]
    for name,expected in hashes.items():assert digest(ROOT/name)==expected,name
    budget.tick()
    result=dict(completed=True,at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
        row=outcome,blocked_reference=blocked,attempted_frames=len(entries),
        full_safe_demonstration=full_safe,teacher_changed_choices=len(changed),
        changed_choices_outside_trainable_support=sum(not e['trainable_support'] for e in changed),
        not_a_model_success=True,not_automatically_admitted_for_training=True,
        original_footprint4cm_actions_dynamics_masks_and_deadline_preserved=True,
        new_physical_episodes=1,new_ppo_steps=0,new_optimizer_updates=0,
        all_input_hashes_verified=len(hashes),resource=budget.statistics(),
        goal_achieved=False,deployment_approved=False)
    write(OUT/'RESULTS.json',result)
    write(OUT/'STATUS.json',dict(status='complete',full_safe_demonstration=full_safe,blocked_reference=blocked))
    print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__=='__main__':
    try:main()
    except Exception as error:
        if OUT.exists():write(OUT/'FAILURE.json',dict(error=repr(error),no_auto_restart=True))
        raise
