"""Pin a fresh diagnostic revision after a verified setup-format exception."""
import ast
from datetime import datetime
import hashlib
import json
from pathlib import Path
import subprocess
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
OLD = OUT.parent/'post_aux_diagnostics_20261016'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def state(unit):
    return dict(line.split('=',1) for line in subprocess.check_output([
        'systemctl','--user','show',unit,'-p','Id','-p','MainPID','-p','ActiveState',
        '-p','Result','-p','InvocationID','-p','ExecMainStatus'],text=True).splitlines())


def main():
    if (OUT/'PLAN.json').exists():
        raise ValueError('Do not overwrite the corrective plan')
    original_plan = json.loads((OLD/'PLAN.json').read_text())
    for name, expected in original_plan['inputs'].items():
        if digest(ROOT/name) != expected:
            raise ValueError('Original pinned input changed: '+name)
    unit = 'rl-post-aux-diagnostics-20261016.service'
    failed = state(unit)
    if (failed['MainPID']!='0' or failed['ActiveState']!='failed'
            or failed['Result']!='exit-code' or failed['ExecMainStatus']!='1'
            or failed['InvocationID']!='dc8d17c030c14011aeb75353cfe1dd94'):
        raise ValueError('The exact previous process must be confirmed terminal')
    log = subprocess.check_output(['journalctl','--user','-u',unit,'-n','70',
                                   '--no-pager','-o','short-iso'],text=True)
    if "ValueError: Only an explicitly recorded training failure can be rehearsed" not in log:
        raise ValueError('The known task-format failure was not verified')
    if any((OLD/'collision_replay').iterdir()):
        raise ValueError('Unexpected prior physical outputs require separate investigation')
    for suffix in ['early_transfer_diagnostic_navigation_20261016',
                   'early_transfer_diagnostic_reference_20261016']:
        if (OUT.parent/suffix).exists():
            raise ValueError('Do not duplicate or overwrite a started navigation evaluation')
    auxiliary = {name:state(name) for name in [
        'rl-rotation-escape-t106-20261016.service',
        'rl-early-transfer-quick-20261016.service']}
    if any(s['MainPID']!='0' or s['ActiveState']!='inactive' or s['Result']!='success'
           for s in auxiliary.values()):
        raise ValueError('Previous auxiliary jobs are not complete')
    for filename in ['replay_collision.py','after_quick.py']:
        ast.parse((OUT/filename).read_text())
    copies = ['recorded_collision_episode7.npz','collision_specification.json',
              'candidate_cases.json','reference_additional_cases.json','reused_reference_rows.json']
    for filename in copies:
        if digest(OUT/filename)!=digest(OLD/filename):
            raise ValueError('Recorded data or declared tasks changed')
    old_source = (OLD/'replay_collision.py').read_text()
    previous = "    physical = ScheduledRecoveryCurriculum(\n"
    insertion = ("    # Match the actual trainer's recorded-case schedule. The prefix itself\n"
                 "    # has an ordinary evaluation id and is explicitly reset by its wrapper;\n"
                 "    # it must not be relabelled or passed through training_case().\n"
                 "    training_cases = json.loads((ROOT/'reports/stop_feedback_20261015/proactive_protocol_20261016/ppo_training_cases.json').read_text())\n"
                 "    physical = ScheduledRecoveryCurriculum(\n")
    expected_source = old_source.replace(previous,insertion,1).replace(
        "recorded_cases=[training_case(spec['case'])],recorded_episodes_per_cycle=1)",
        "recorded_cases=[training_case(case) for case in training_cases],recorded_episodes_per_cycle=1)",1)
    if (OUT/'replay_collision.py').read_text()!=expected_source:
        raise ValueError('The fix must only restore the original trainer curriculum list')
    if digest(OUT/'after_quick.py')!=digest(OLD/'after_quick.py'):
        raise ValueError('The evaluation order and conditions must remain unchanged')
    (OUT/'ORIGINAL_FAILURE.log').write_text(log)
    failure = dict(at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
        original_service=failed, previous_auxiliary_states=auxiliary,
        original_status=json.loads((OLD/'STATUS.json').read_text()),
        exception='training_case rejected ordinary prefix-case id motion_profile_015',
        failed_before_curriculum_constructor=True, original_physical_steps=0,
        original_files_preserved=True, original_plan_sha256=digest(OLD/'PLAN.json'),
        source_diff='Initialize ScheduledRecoveryCurriculum with the same12 recorded training cases as the actual trainer; prefix reset and all25 sampled actions remain unchanged.',
        reused_earlier_result_as_new_result=False, no_observation_timeout_restart=True)
    (OUT/'ORIGINAL_FAILURE.json').write_text(json.dumps(failure,indent=2)+'\n')
    plan = dict(original_plan)
    plan.update(created_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        corrective_revision=True, original_failed_plan=str((OLD/'PLAN.json').relative_to(ROOT)),
        original_failed_invocation=failed['InvocationID'],
        correction_changes_physics_policy_masks_rewards_or_tasks=False,
        original_initialization_output_preserved=True)
    plan['inputs'] = dict(original_plan['inputs'])
    for p in [OLD/'PLAN.json',OUT/'prepare_fix.py',OUT/'replay_collision.py',
              OUT/'after_quick.py',OUT/'ORIGINAL_FAILURE.log',OUT/'ORIGINAL_FAILURE.json',
              *(OUT/filename for filename in copies)]:
        plan['inputs'][str(p.relative_to(ROOT))] = digest(p)
    (OUT/'PLAN.json').write_text(json.dumps(plan,indent=2)+'\n')
    print(json.dumps(dict(prepared=True,original_failure_confirmed=True,
        preserved_original_plan=True,pinned_inputs=len(plan['inputs']),
        unchanged_sampled_actions=25,unchanged_candidate_tasks=6,
        unchanged_additional_reference_tasks=3,additional_physical_simulations_started=0)))


if __name__=='__main__':
    main()
