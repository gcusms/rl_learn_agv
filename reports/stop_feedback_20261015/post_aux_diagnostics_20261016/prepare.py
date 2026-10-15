"""Freeze one recorded collision and an explicitly non-promoting BC study."""
from datetime import datetime
import hashlib
import json
from pathlib import Path
import subprocess
from zoneinfo import ZoneInfo

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
BASE = ROOT / 'reports/stop_feedback_20261015'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, data):
    path.write_text(json.dumps(data, indent=2)+'\n')


def main():
    if (OUT / 'PLAN.json').exists():
        raise ValueError('Do not replace a frozen diagnostic plan')
    model = ROOT / 'models/stop_feedback_20261015/w11_retained2000_s231015'
    observations, masks, choices, starts, records = [], [], [], [], []
    sources = []
    for path in sorted((model/'rollouts').glob('*.npz'))[:5]:
        with np.load(path, allow_pickle=False) as data:
            observations.extend(data['observations'][:,0])
            masks.extend(data['action_masks'][:,0])
            choices.extend(data['sampled_actions'][:,0,0])
            starts.extend(data['episode_starts'][:,0])
            records.extend(json.loads(str(value)) for value in data['execution_diagnostics_json'][:,0])
        sources.append(path)
    beginning = np.flatnonzero(starts)
    first, end = int(beginning[6]), int(beginning[7])
    selected = records[first:end]
    if end-first != 25 or not selected[-1]['done']:
        raise ValueError('The complete, seventh25-frame training collision is required')
    if len({row['case_json'] for row in selected}) != 1:
        raise ValueError('Recorded collision crosses task boundaries')
    original_case = json.loads(selected[0]['case_json'])
    if original_case['source_training_case_id'] != 'physical_prefix_early_avoidance_t6':
        raise ValueError('Wrong physical training prefix')
    spec_path = BASE/'w11_success_augmented_protocol_20261016/ppo_training_prefixes.json'
    specification = next(s for s in json.loads(spec_path.read_text()) if s['id']=='early_avoidance_t6')
    write(OUT/'collision_specification.json', specification)
    archived = OUT/'recorded_collision_episode7.npz'
    np.savez_compressed(archived,
        observations=np.asarray(observations[first:end], np.float32),
        masks=np.asarray(masks[first:end], bool),
        choices=np.asarray(choices[first:end], np.int64),
        diagnostics_json=np.asarray([json.dumps(row) for row in selected]),
        metadata_json=np.asarray(json.dumps(dict(source_model=str(model.relative_to(ROOT)),
            original_training_episode=7, original_case=original_case,
            diagnostic_only_not_an_evaluation=True, expected_collision_count=1))))

    gate_path = BASE/'w11_success_augmented_protocol_20261016/gate_cases.json'
    gate = json.loads(gate_path.read_text())
    primary_ids = ['motion_profile_015','recovery_holdout_00','recovery_holdout_01']
    primary = [next(c for c in gate if c['id']==key) for key in primary_ids]
    training_path = BASE/'proactive_protocol_20261016/ppo_training_cases.json'
    training = json.loads(training_path.read_text())
    extra_ids = ['train_failure_proactive_variation_02',
                 'train_failure_proactive_variation_06',
                 'train_failure_proactive_variation_07']
    extra = [dict(next(c for c in training if c['id']==key),
                  pilot_category='diagnostic_original_training_rehearsal') for key in extra_ids]
    write(OUT/'candidate_cases.json', primary+extra)
    write(OUT/'reference_additional_cases.json', extra)
    reference = BASE/'w11_recovery_bc800_navigation_20261016/RESULTS.json'
    report = json.loads(reference.read_text())
    if not report['completed'] or len(report['rows']) != 18:
        raise ValueError('Complete frozen BC reference required')
    old_rows = [row for row in report['rows'] if row['case_id'] in primary_ids]
    if len(old_rows) != 3:
        raise ValueError('Three exact reference outcomes required')
    write(OUT/'reused_reference_rows.json', dict(source=str(reference.relative_to(ROOT)),
        source_sha256=digest(reference), rows=old_rows,
        repeated_as_new_success=False))

    differences = BASE/'early_decision_transfer_20261016/FIT_DIFFERENCES.json'
    fit = json.loads(differences.read_text())
    if (not fit['completed'] or len(fit['disagreements'])!=6
            or fit['motion_to_wait'] or fit['wait_to_motion']
            or not fit['original_admission_gate_still_failed']):
        raise ValueError('Declared legal motion disagreements required before diagnostic admission')
    files = set(sources+[spec_path, gate_path, training_path, reference, differences])
    files.update(OUT/name for name in ['prepare.py','replay_collision.py','after_quick.py',
        'collision_specification.json','recorded_collision_episode7.npz',
        'candidate_cases.json','reference_additional_cases.json','reused_reference_rows.json'])
    files.add(model/selected[0]['initialization_trace'])
    for folder in ['models/stop_feedback_20261015/baseline/portable',
                   'models/stop_feedback_20261015/w11_recovery_bc800_s231015/bc_portable',
                   'models/stop_feedback_20261015/early_transfer_bc800_s231015/bc_portable']:
        files.update(p for p in (ROOT/folder).iterdir() if p.is_file() and p.suffix!='.so')
    for folder in ['training/neural_navigation','training/navrl']:
        files.update((ROOT/folder).glob('*.py'))
    for name in ['prefix_recovery_training.py','recovery_sampling.py','fixed_training_replay.py',
                 'stop_feedback_navigation.py','quality_masked_navigation.py',
                 'train_localization_recovery.py','evaluate_docking_navigation.py',
                 'evaluate_feedback_recovery.py','masked_navigation.py','recovery_curriculum.py']:
        files.add(ROOT/'training/experiments'/name)
    plan = dict(created_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        dependency_unit='rl-early-transfer-quick-20261016.service',
        dependency_invocation='d593629d83d8422c9a87d63b6347a5aa',
        hypothesis='Recorded sensor/action replay can separate contact geometry from action-label disagreements; complete model-only navigation can test whether the six legal motion label changes harm the original training tasks.',
        original_quick_admission_failed_and_not_overridden=True,
        diagnostic_study_admitted=True, candidate_not_promoted=True,
        admission_evidence='Six mismatches are legal motion alternatives,all55waitlabels remain,and7new demonstratedstop-to-motiontargets arecorrect. This warrants diagnosis,not a claim of improved navigation.',
        collision=dict(recorded_steps=25, prefix_steps=30, fixed_sample_replay=True,
            policy_evaluation=False, new_ppo_steps=0),
        navigation=dict(candidate_tasks=6, additional_reference_tasks=3,
            reused_reference_rows=3, model_only_from_reset=True,
            original_training_tasks_and_rehearsals_only=True,
            original14_and_heading_suite_not_fully_covered=True,
            extra8_conditions_not_run=True),
        resource=dict(soft_cpu_cores=.15,hard_cpu_percent=20,nice=19,
            memory_high_gib=2.5,memory_max_gib=3,
            reuse_existing_auxiliary_slot_after_exact_dependency_finishes=True,
            no_third_physical_simulation=True),
        inputs={str(p.relative_to(ROOT)):digest(p) for p in sorted(files)},
        no_automatic_expansion_training_or_promotion=True,
        goal_achieved=False,deployment_approved=False)
    write(OUT/'PLAN.json', plan)
    print(json.dumps(dict(prepared=True,pinned_inputs=len(files),
        collision_sampled_steps=25,candidate_cases=6,additional_reference_cases=3,
        original_admission_still_failed=True,new_ppo_steps=0)))


if __name__=='__main__':
    main()
