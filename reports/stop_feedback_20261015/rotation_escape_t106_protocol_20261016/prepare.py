"""Prepare an isolated, truthfully labelled frozen-policy guard diagnostic."""
from copy import deepcopy
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
SOURCE = Path('models/stop_feedback_20261015/w11_recovery_bc800_s231015/bc_portable')
TARGET = Path('models/stop_feedback_20261015/rotation_escape_t106_diagnostic_20261016/portable')
BASE = Path('reports/stop_feedback_20261015')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def revise_training_claims(value):
    if isinstance(value, dict):
        for key, item in value.items():
            if key in ('actions_masks_guard_unchanged', 'unchanged_actions_masks_guard'):
                value[key] = False
            else:
                revise_training_claims(item)
    elif isinstance(value, list):
        for item in value:
            revise_training_claims(item)


def main():
    os.chdir(ROOT)
    if TARGET.exists() or (OUT / 'PLAN.json').exists():
        raise ValueError('Fresh diagnostic output and plan required')
    shutil.copytree(SOURCE, TARGET)
    hardware = json.loads((SOURCE / 'hardware.json').read_text())
    if hardware['near_field_memory'].get('clearance_increasing_rotation', False):
        raise ValueError('Expected a reference with rotation escape disabled')
    changed = deepcopy(hardware)
    changed['near_field_memory']['clearance_increasing_rotation'] = True
    write(TARGET / 'hardware.json', changed)
    change = dict(
        path='hardware.near_field_memory.clearance_increasing_rotation', before=False, after=True,
        changes_action_mask=True, actor_weights_changed=False, fixed_body_margin_unchanged=True,
        original_training_used_rotation_escape=False, diagnostic_only=True)
    preservation = json.loads((SOURCE / 'navigation_preservation.json').read_text())
    revise_training_claims(preservation)
    preservation['diagnostic_guard_override'] = change
    write(TARGET / 'navigation_preservation.json', preservation)
    manifest = json.loads((SOURCE / 'manifest.json').read_text())
    revise_training_claims(manifest)
    manifest.update(navigation_preservation_contract=preservation,
        diagnostic_guard_override=change, diagnostic_parent_manifest_sha256=digest(SOURCE / 'manifest.json'),
        deployment_approved=False, exact_action_parity=False)
    manifest['historical_comparison_inputs_before_guard_change'] = manifest.pop('comparison_inputs', None)
    manifest['files'] = {name: digest(TARGET / name) for name in manifest['files']}
    write(TARGET / 'manifest.json', manifest)
    card = json.loads((SOURCE / 'MODEL_CARD.json').read_text())
    revise_training_claims(card)
    card.update(manifest_sha256=digest(TARGET / 'manifest.json'), diagnostic_guard_override=change,
        method='Frozen BC actor with diagnostic rotation-escape mask; no additional training')
    card['limitations'] += [
        'Guard configuration differs from training and from all original-mask scores.',
        'Any fixed-window probe is diagnostic and cannot be counted as a learned-policy success.']
    write(TARGET / 'MODEL_CARD.json', card)
    for name in ('actor.pt', 'localizer.pt', 'router.pt'):
        if digest(SOURCE / name) != digest(TARGET / name):
            raise ValueError('Frozen model weights changed: ' + name)
    prefix = json.loads((BASE / 'late_recovery_feasibility_20261016/prefix.json').read_text())
    write(OUT / 'prefix.json', prefix)
    write(OUT / 'model_specifications.json', [prefix])
    write(OUT / 'probe_specifications.json', [dict(id='right_rotation_3p2s',
        start_s=106., end_s=109.2, choice=5)])
    legacy = Path('/home/ubuntu/rl_learn/rl_learn_agv/training/runs/safety_context_20261011/rotation_escape_probe/RESULTS.json')
    write(OUT / 'LEGACY_NEGATIVE_EVIDENCE.json', dict(source=str(legacy), source_sha256=digest(legacy),
        report=json.loads(legacy.read_text()),
        relevance='An older 300D full-reset probe had 0/5 safe; the flag is not a globally proven fix.'))
    names = [OUT / 'run.py', OUT / 'prepare.py', OUT / 'prefix.json', OUT / 'model_specifications.json',
        OUT / 'probe_specifications.json', OUT / 'LEGACY_NEGATIVE_EVIDENCE.json',
        BASE / 'guard_geometry_audit_20261016/RESULTS.json', Path(prefix['trace']),
        BASE / 'w11_recovery_bc800_w11_20261016/RESULTS.json',
        Path('training/neural_navigation/runtime.py'), Path('training/neural_navigation/safety_memory.py'),
        Path('training/experiments/evaluate_feedback_recovery.py'),
        Path('training/experiments/evaluate_docking_navigation.py'),
        Path('training/experiments/probe_feedback_alternatives.py'),
        Path('training/experiments/probe_recovery_prefix.py')]
    names += list(SOURCE.iterdir()) + list(TARGET.iterdir())
    names = sorted(set(path for path in names if path.is_file()))
    plan = dict(created_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
        source_bundle=str(SOURCE), diagnostic_bundle=str(TARGET),
        dependency_unit='rl-w11-paired-v2-after-validation-20261016.service',
        dependency_invocation='5011449a519943149d7d682002f8cb1b',
        dependency_completed_report=str(BASE / 'w11_recovery_bc800_fresh_20261016/RESULTS.json'),
        dependency_paired_report=str(BASE / 'w11_success_augmented_protocol_20261016/PAIRED_RESULTS.json'),
        model_output=str(BASE / 'rotation_escape_t106_model_20261016'),
        probe_output=str(BASE / 'rotation_escape_t106_probe_20261016'),
        inputs={str(path.relative_to(ROOT) if path.is_absolute() else path): digest(path) for path in names},
        soft_cpu_cores=.15, maximum_simultaneous_auxiliary_simulations=1,
        declared_probe_window_s=[106., 109.2], declared_probe_choice=5,
        no_action_mask_bypass=True, current_training_or_deployment_changed=False,
        new_ppo_steps=0, goal_achieved=False, deployment_approved=False)
    write(OUT / 'PLAN.json', plan)
    print(json.dumps(dict(prepared=True, inputs=len(plan['inputs']), diagnostic_bundle=str(TARGET))))


if __name__ == '__main__':
    main()
