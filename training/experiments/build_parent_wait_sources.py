"""Admit only complete-safe parent recovery suffixes; preserve old sources."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import torch

from experiments.build_release_recovery_sources import training_indices
from experiments.long_risk_actor import digest
from experiments.release_actor import ReleaseActor
from experiments.release_fit_weights import fit_weights
from neural_navigation.experiment_budget import ExperimentBudget
from neural_navigation.masked_actions import ACTIONS, COMMANDS, masks_from_observation
from neural_navigation.release_observation import augment, contract
from neural_navigation.sensors import LidarSpec


ROOT = Path(__file__).resolve().parents[2]


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def main(args):
    parent, model, source, output = [getattr(args, name).resolve()
        for name in ['parent', 'model', 'source', 'output']]
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    budget = ExperimentBudget(.15, memory_mib=2048, directory=output)
    meta = json.loads((parent / 'RESULTS.json').read_text())
    report_path = source / 'RESULTS.json'
    report = json.loads(report_path.read_text())
    assert meta['completed'] and report['completed']
    assert digest(parent / 'demonstrations.npz') == meta['dataset_sha256']
    accepted = [r for r in report['rows'] if r.get('complete_safe_source_admitted', False)]
    rejected = [r for r in report['rows'] if r not in accepted]
    if not accepted:
        write(output / 'RESULTS.json', dict(completed=True, skipped=True,
            reason='No complete safe parent continuation; failed/partial traces are not labels.',
            frames=0, new_ppo_steps=0, new_optimizer_updates=0, goal_achieved=False))
        return
    failed_path = ROOT / 'reports/stop_feedback_20261015/retained_sensor_release_broad_20261016/RESULTS.json'
    failed = json.loads(failed_path.read_text())
    assert failed['completed']
    failed_rows = failed['groups']['actual95_actual90']['rows']
    excluded = sorted(set(meta['excluded_failed_trace_sha256s']
        + [r['trace_sha256'] for r in failed_rows + rejected]))
    files = [Path(__file__), parent / 'RESULTS.json', parent / 'demonstrations.npz',
        report_path, source / 'PLAN.json', failed_path, model / 'checkpoint.pt', model / 'parent_actor.pt',
        model / 'portable/actor.pt', model / 'portable/hardware.json',
        ROOT / 'training/experiments/build_release_recovery_sources.py',
        ROOT / 'training/experiments/release_fit_weights.py',
        ROOT / 'training/experiments/release_actor.py']
    files += [ROOT / r['trace_path'] for r in accepted]
    hashes = {str(p.relative_to(ROOT)):digest(p) for p in files}
    write(output / 'PLAN.json', dict(at=datetime.now(timezone.utc).isoformat(), inputs=hashes,
        parent_source_frames=meta['frames'], observation_contract=contract(),
        admitted_complete_safe_sources=[r['case_id'] for r in accepted],
        rejected_source_cases=[r['case_id'] for r in rejected],
        failed_initialization_prefixes_never_positive_labels=True,
        original_arrays_preserved=True, new_ppo_steps=0, new_physical_episodes=0))
    actor = ReleaseActor(torch.jit.load(str(model / 'parent_actor.pt'),map_location='cpu')).eval()
    actor.load_state_dict(torch.load(model / 'checkpoint.pt',map_location='cpu',weights_only=False)['actor'])
    portable = torch.jit.load(str(model / 'portable/actor.pt'),map_location='cpu').eval()
    lidar = LidarSpec(json.loads((model / 'portable/hardware.json').read_text())['lidar'])
    with np.load(parent / 'demonstrations.npz',allow_pickle=False) as archive:
        original = {k:archive[k].copy() for k in archive.files}
    pieces = {key:[value] for key,value in original.items()}
    sources = list(meta['sources'])
    next_id = int(original['source_ids'].max()) + 1
    for row in accepted:
        assert row['safe_success'] and row['physical_arrival'] and row['protocol_valid']
        assert not any(row[key] for key in ['timeout','false_arrival','collisions','physical_guards'])
        assert row['prefix_sensor_and_physical_parity_verified'] and row['reference_success_is_not_candidate_success']
        path = ROOT / row['trace_path']
        assert digest(path) == row['trace_sha256'] and digest(path) not in excluded
        with np.load(path,allow_pickle=False) as archive:
            trace = {key:archive[key].copy() for key in archive.files}
        observations, choices, masks = [trace[key] for key in ['observations','choices','masks']]
        controlled = trace['reference_parent_controlled'].astype(bool)
        n = row['frozen_prefix_frames']
        np.testing.assert_array_equal(controlled,np.arange(len(choices))>=n)
        np.testing.assert_array_equal(masks_from_observation(observations[:,:282]),masks)
        assert masks[np.arange(len(choices)),choices].all()
        assert observations[n,434]*5 >= 2-1e-6 and choices[n] != 0
        with torch.no_grad():
            tensor = torch.from_numpy(observations)
            expected = actor.parent(tensor[:,:439]).numpy()
            np.testing.assert_array_equal(expected[controlled],ACTIONS[choices[controlled]])
            np.testing.assert_array_equal(portable(tensor).numpy()[controlled],
                ACTIONS[trace['candidate_choices_on_reference_context'][controlled]])
            support = actor.support(tensor).numpy()
        for obs,scan in zip(observations,trace['scans']):
            np.testing.assert_array_equal(augment(obs[:481],scan,lidar),obs)
        selected = training_indices(trace,row,support)
        selected = selected[controlled[selected]]
        assert len(selected)>0 and selected.min()==n and selected.max()==len(choices)-2
        np.testing.assert_allclose(trace['commands'][selected],COMMANDS[choices[selected]],rtol=0,atol=2e-8)
        extension = dict(observations=observations[selected],choices=choices[selected],masks=masks[selected],
            source_ids=np.full(len(selected),next_id,np.int64),sensor_times=trace['labels'][selected,0])
        for key,value in extension.items():
            pieces[key].append(value)
        sources.append(dict(case_id=row['case_id'],source_report=str(report_path.relative_to(ROOT)),
            trace_path=row['trace_path'],trace_sha256=row['trace_sha256'],frames=len(selected),
            complete_episode_frames=len(choices),excluded_initialization_prefix_frames=n,
            excluded_inactive_arrival_frames=[len(choices)-1],safe_success=True,
            collision_free_complete_episode=True,source_is_offline_teacher=True,
            source_is_verified_frozen_parent=True,reference_parent_and_sensor_inputs_replayed=True,
            not_a_full_model_only_mission=True))
        next_id += 1
        budget.tick()
    arrays = {key:np.concatenate(value) for key,value in pieces.items()}
    for key,value in original.items():
        assert arrays[key][:len(value)].tobytes()==value.tobytes()
    with torch.no_grad():
        tensor = torch.from_numpy(arrays['observations'])
        features, baseline = actor.features(tensor)
        active = actor.support(tensor).numpy()
    nominal = np.abs(baseline.numpy()[:,None]-ACTIONS[None]).max(2).argmin(1)
    assert not ((~active)&(nominal!=arrays['choices'])).any()
    _,weighting = fit_weights(features.numpy()[active],arrays['choices'][active],
        arrays['observations'][active,434]*5,mode='recovery_phase')
    np.savez_compressed(output / 'demonstrations.npz',**arrays)
    for name,expected in hashes.items():
        assert digest(ROOT/name)==expected,name
    budget.tick()
    result = dict(completed=True,at=datetime.now(timezone.utc).isoformat(),frames=len(tensor),
        parent_frames=meta['frames'],added_frames=len(tensor)-meta['frames'],sources=sources,
        dataset_sha256=digest(output/'demonstrations.npz'),original_prefix_byte_exact=True,
        failed_latest_trace_excluded=True,excluded_failed_trace_sha256s=excluded,
        original_fresh_four_now_development=True,failed_initialization_prefixes_excluded=True,
        training_weight_preflight=weighting,new_training_steps=0,new_physical_steps=0,new_ppo_steps=0,
        resource=budget.statistics(),goal_achieved=False,deployment_approved=False)
    write(output/'RESULTS.json',result)
    (output/'driver_snapshot.py').write_text(Path(__file__).read_text())
    print(json.dumps({k:v for k,v in result.items() if k!='sources'}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['parent','model','source','output']:
        parser.add_argument('--'+name,type=Path,required=True)
    main(parser.parse_args())
