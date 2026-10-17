"""After the bounded physical probe, admit only a complete safe source.

No optimizer or physics runs here. A successful expert probe is still an
expert result. Its fixed-prefix initialization and pre-intervention waiting
are excluded from new targets. If no complete safe source exists, emit that
negative finding rather than imitate a failed or merely legal maneuver.
"""
from datetime import datetime
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import time
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
BASE = ROOT / 'reports/stop_feedback_20261015'
PROBE = BASE / 'actual106_confirmed_free_probe_verified_20261016'
MODEL = ROOT / 'models/stop_feedback_20261015/actual95_transfer_bc400_s231016'
DEPENDENCY = 'rl-actual106-confirmed-free-probe-verified-20261016.service'


def stamp():
    return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(name, data):
    target = OUT / name
    temporary = target.with_suffix(target.suffix+'.tmp')
    temporary.write_text(json.dumps(data, indent=2)+'\n')
    temporary.replace(target)


def state():
    value = subprocess.check_output(['systemctl', '--user', 'show', DEPENDENCY,
        '-p', 'ActiveState', '-p', 'Result', '-p', 'MainPID', '-p', 'InvocationID'], text=True)
    return dict(line.split('=', 1) for line in value.splitlines())


def main():
    if (OUT / 'PLAN.json').exists():
        raise FileExistsError('Do not restart a source-admission job')
    preceding = state()
    paths = [Path(__file__), PROBE / 'run.py', PROBE / 'PLAN.json',
        MODEL / 'policy_bc.zip', MODEL / 'demonstrations.npz',
        MODEL / 'bc_portable/actor.pt', MODEL / 'bc_portable/hardware.json',
        ROOT / 'training/experiments/recovery_demonstrations.py',
        BASE / 'actual106_memory_context_20261016/replay.py']
    inputs = {str(p.relative_to(ROOT)): digest(p) for p in paths}
    plan = dict(at=stamp(), inputs=inputs, predecessor=DEPENDENCY,
        predecessor_invocation=preceding['InvocationID'], maximum_sources=1,
        exclude_all_initialization_and_waiting_before_first_expert_window=True,
        only_complete_safe_physical_arrival=True,
        require_post106_frame_sampled_truth_gap_at_least_m=.04,
        stop_labels_remain_in_old_dataset=True, no_failed_positive_examples=True,
        no_optimization_or_physical_simulation=True,
        new_ppo_steps=0, new_optimizer_updates=0, deployment_approved=False, goal_achieved=False)
    write('PLAN.json', plan)
    while True:
        current = state()
        if current['InvocationID'] not in ('', plan['predecessor_invocation']):
            raise RuntimeError('The preceding physical probe was restarted')
        if current['ActiveState'] in ('inactive', 'failed'):
            if current['ActiveState'] != 'inactive' or current['Result'] != 'success' or current['MainPID'] != '0':
                raise RuntimeError('The physical source process did not finish successfully')
            break
        write('STATUS.json', dict(at=stamp(), status='waiting_for_complete_physical_probe',
            predecessor=current, source_admitted=False, optimization_started=False))
        time.sleep(5)
    for name, expected in inputs.items():
        assert digest(ROOT / name) == expected, name
    source_path = PROBE / 'RESULTS.json'
    source = json.loads(source_path.read_text())
    assert source['completed']
    report = dict(completed=True, at=stamp(), source_report_sha256=digest(source_path),
        source_admitted=False, new_ppo_steps=0, new_optimizer_updates=0,
        new_physical_steps=0, goal_achieved=False, deployment_approved=False)
    if source.get('skipped'):
        report['reason'] = 'Physical source skipped by its own predecessor admission gate.'
        write('RESULTS.json', report)
        return
    safe = [r for r in source['rows'] if r['safe_success']]
    if not safe:
        report['reason'] = 'No complete safe actual106 recovery; do not turn failed maneuvers into positive labels.'
        write('RESULTS.json', report)
        return
    if safe[0]['no_expert_actions']:
        report['reason'] = 'The unchanged actor recovered with the memory component; investigate full-time memory regression before adding imitation targets.'
        report['model_only_safe_recovery'] = True
        write('RESULTS.json', report)
        return
    row = safe[0]
    assert row['physical_arrival'] and not row['false_arrival'] and row['protocol_valid']
    assert not row['timeout'] and not row['collisions'] and not row['physical_guards']
    assert row['prefix_sensor_and_physical_parity_verified'] and row['frozen_prefix_frames'] == 530
    trace_path = PROBE / 'physical' / (row['case_id']+'.npz')
    assert digest(trace_path) == row['trace_sha256']
    import numpy as np
    import torch
    from experiments.recovery_demonstrations import convert_observations
    from experiments.stop_recovery_policy import StopRecoveryPPO
    from neural_navigation.experiment_budget import ExperimentBudget
    from neural_navigation.masked_actions import ACTIONS, COMMANDS, masks_from_observation
    torch.set_num_threads(1)
    budget = ExperimentBudget(.15, memory_mib=2048, directory=OUT)
    with np.load(trace_path, allow_pickle=False) as z:
        data = {k:z[k].copy() for k in z.files}
    times = data['labels'][:, 0]
    choices = data['choices'].astype(np.int64)
    masks = data['masks'].astype(bool)
    np.testing.assert_array_equal(masks_from_observation(data['observations'][:, :282]), masks)
    assert masks[np.arange(len(choices)), choices].all()
    first_intervention = min(w['start_s'] for w in row['specification']['windows'])
    expert = np.zeros(len(times), bool)
    for window in row['specification']['windows']:
        expert |= (times >= window['start_s']-1e-8) & (times < window['end_s']-1e-8)
    np.testing.assert_array_equal(data['expert_controlled'], expert)
    np.testing.assert_array_equal(data['model_controlled'], (times >=106.-1e-8) & ~expert)
    entries = row['intervention']
    assert any(e['requested_choice_legal'] and e['requested_choice'] for e in entries)
    # The true geometry is only a label-quality audit, never an actor input.
    spec = importlib.util.spec_from_file_location('memory_audit_geometry', paths[-1])
    geometry = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(geometry)
    case = json.loads((PROBE / 'prefix.json').read_text())['case']
    scene = json.loads((ROOT / 'training' / case['scene'] / 'physical_scene.json').read_text())
    hardware = json.loads((MODEL / 'bc_portable/hardware.json').read_text())
    half = np.asarray(hardware['robot_footprint_m'])/2
    walls = np.asarray(scene['walls'], float)
    radii = np.asarray([h['radius_m'] for h in scene['dynamic_obstacles']])
    gaps = []
    for i in np.flatnonzero(times>=106.-1e-8):
        gaps.append(geometry.physical_gap(data['labels'][i,1:4], half, walls,
                                         data['human_positions'][i], radii))
        budget.tick()
    minimum_gap = float(min(gaps))
    report['minimum_post106_frame_sampled_truth_gap_m'] = minimum_gap
    report['frame_sampled_gap_is_not_a_continuous_hardware_safety_certificate'] = True
    if minimum_gap < .04:
        report['reason'] = 'Reached without collision, but sampled true clearance fell below unchanged4cm source admission margin.'
        write('RESULTS.json', report)
        return
    obs = convert_observations(data['observations'], masks, data['commands'], times)
    model = StopRecoveryPPO.load(MODEL / 'policy_bc.zip', device='cpu')
    model.policy.set_training_mode(False)
    with torch.no_grad():
        supported = model.policy.recovery_support(torch.from_numpy(obs)).numpy()
    keep = supported & (times >= first_intervention-1e-8)
    assert keep.any() and np.count_nonzero(choices[keep])
    away = np.linalg.norm(obs[:,218:220]*12.,axis=1)>.3
    np.testing.assert_allclose(data['commands'][away], COMMANDS[choices[away]], atol=2e-8, rtol=0)
    actor = torch.jit.load(str(MODEL / 'bc_portable/actor.pt'), map_location='cpu').eval()
    checked = 0
    for i in np.flatnonzero(data['model_controlled'] & away):
        with torch.no_grad():
            predicted = actor(torch.from_numpy(data['observations'][i:i+1])).numpy()[0]
        np.testing.assert_array_equal(predicted, ACTIONS[choices[i]])
        checked += 1
        budget.tick()
    with np.load(MODEL / 'demonstrations.npz', allow_pickle=False) as z:
        old = {k:z[k].copy() for k in ['observations','masks','choices']}
    assert len(old['choices']) == 2421 and int((old['choices']==0).sum()) == 57
    combined = {k:np.concatenate([old[k], a[keep]]) for k,a in
                [('observations',obs),('masks',masks),('choices',choices)]}
    seen, conflicts = {}, []
    for i,(x,mask,label) in enumerate(zip(combined['observations'],combined['masks'],combined['choices'])):
        key = hashlib.sha256(x.tobytes()+mask.tobytes()).digest()
        before = seen.setdefault(key,(int(label),i))
        if before[0] != int(label): conflicts.append([before[1],i])
    if conflicts:
        report.update(reason='Exact sensor/mask input has conflicting labels; review before fitting.', conflicts=conflicts)
        write('RESULTS.json',report)
        return
    np.savez_compressed(OUT/'demonstrations.npz',**combined)
    for name, expected in inputs.items(): assert digest(ROOT/name)==expected,name
    report.update(source_admitted=True, source_id=row['case_id'], source_trace=str(trace_path.relative_to(ROOT)),
        source_trace_sha256=digest(trace_path), source_is_expert_not_learned_model_success=True,
        source_has_experimental_memory_handover=True, source_first_intervention_s=first_intervention,
        old_examples=len(old['choices']), old_necessary_stop_labels_preserved=57,
        new_examples=int(keep.sum()), new_movement_targets=int(np.count_nonzero(choices[keep])),
        new_stop_targets=int((choices[keep]==0).sum()), total_examples=len(combined['choices']),
        original_prefix_and_pre_intervention_waiting_excluded=True, exact_input_conflicts=0,
        independently_checked_actor_driving_frames=checked,
        demonstrations_sha256=digest(OUT/'demonstrations.npz'),resource=budget.statistics(),
        followup='Only a bounded separate fitting experiment is admissible; full-time memory and learned-control regressions are still required.')
    write('RESULTS.json',report)
    write('STATUS.json',dict(at=stamp(),status='source_admitted_for_review_only',optimization_started=False))


if __name__=='__main__':
    os.chdir(ROOT)
    try:
        main()
    except Exception as error:
        write('FAILURE.json',dict(at=stamp(),error=repr(error),automatically_restarted=False,
            source_admitted=False,goal_achieved=False,deployment_approved=False))
        raise
