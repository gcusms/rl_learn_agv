"""Join verified safe reference suffixes without relabeling failed histories."""
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
from neural_navigation.release_observation import augment
from neural_navigation.sensors import LidarSpec


ROOT=Path(__file__).resolve().parents[2]


def write(path,value):
    path.write_text(json.dumps(value,indent=2)+'\n')


def histogram(values):
    keys,counts=np.unique(values,return_counts=True)
    return {int(key):int(count) for key,count in zip(keys,counts)}


def main(args):
    parent,replay,model,output=[getattr(args,name).resolve() for name in ['parent','replay','model','output']]
    output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1)
    budget=ExperimentBudget(.15,memory_mib=2048,directory=output)
    meta=json.loads((parent/'RESULTS.json').read_text());report=json.loads((replay/'RESULTS.json').read_text())
    assert meta['completed'] and report['completed'] and report['new_physical_episodes']==0
    assert len(report['rows'])==3 and all(r['complete_safe_source_admitted'] for r in report['rows'])
    assert digest(parent/'demonstrations.npz')==meta['dataset_sha256']
    files=[Path(__file__),parent/'RESULTS.json',parent/'demonstrations.npz',replay/'RESULTS.json',
        replay/'PLAN.json',model/'checkpoint.pt',model/'parent_actor.pt',model/'portable/hardware.json',
        ROOT/'training/experiments/build_release_recovery_sources.py',
        ROOT/'training/experiments/release_fit_weights.py',ROOT/'training/experiments/release_actor.py']
    files += [ROOT/r[k] for r in report['rows'] for k in ['trace_path','reference_actor','physical_source_report','physical_source_trace']]
    hashes={str(p.relative_to(ROOT)):digest(p) for p in set(files)}
    write(output/'PLAN.json',dict(at=datetime.now(timezone.utc).isoformat(),inputs=hashes,
        full_safe_original_physical_evidence_and_causal_replay_required=True,
        source_initialization_prefixes_excluded=True,old_arrays_byte_exact=True,
        terminal_execution_not_relabelled=True,new_ppo_steps=0,new_physical_episodes=0))
    actor=ReleaseActor(torch.jit.load(str(model/'parent_actor.pt'),map_location='cpu')).eval()
    actor.load_state_dict(torch.load(model/'checkpoint.pt',map_location='cpu',weights_only=False)['actor'])
    lidar=LidarSpec(json.loads((model/'portable/hardware.json').read_text())['lidar'])
    with np.load(parent/'demonstrations.npz',allow_pickle=False) as data:
        original={k:data[k].copy() for k in data.files}
    pieces={key:[value] for key,value in original.items()};sources=list(meta['sources'])
    source_id=int(original['source_ids'].max())+1
    for row in report['rows']:
        assert row['safe_success'] and row['physical_arrival'] and row['protocol_valid']
        assert not any(row[k] for k in ['collisions','timeout','false_arrival','physical_guards'])
        path=ROOT/row['trace_path'];assert digest(path)==row['trace_sha256']
        assert digest(ROOT/row['physical_source_trace'])==row['physical_source_trace_sha256']
        assert row['original_masks_actions_and_execution_reproduced'] and row['independent_original_physical_success_reused']
        with np.load(path,allow_pickle=False) as data:
            trace={k:data[k].copy() for k in data.files}
        obs,choices,masks=[trace[k] for k in ['observations','choices','masks']]
        controlled=trace['reference_replay_controlled'].astype(bool)
        np.testing.assert_array_equal(controlled,np.arange(len(choices))>=row['frozen_prefix_frames'])
        np.testing.assert_array_equal(masks_from_observation(obs[:,:282]),masks)
        assert masks[np.arange(len(choices)),choices].all()
        reference=torch.jit.load(str(ROOT/row['reference_actor']),map_location='cpu').eval()
        with torch.no_grad():
            tensor=torch.from_numpy(obs)
            np.testing.assert_array_equal(reference(tensor[:,:439]).numpy()[controlled],ACTIONS[choices[controlled]])
            support=actor.support(tensor).numpy()
        for observation,scan in zip(obs,trace['scans']):
            np.testing.assert_array_equal(augment(observation[:481],scan,lidar),observation)
        selected=training_indices(trace,row,support)
        selected=selected[controlled[selected]]
        assert len(selected)>0 and selected.min()==row['frozen_prefix_frames']
        np.testing.assert_allclose(trace['commands'][selected],COMMANDS[choices[selected]],rtol=0,atol=2e-8)
        extension=dict(observations=obs[selected],choices=choices[selected],masks=masks[selected],
            source_ids=np.full(len(selected),source_id,np.int64),sensor_times=trace['labels'][selected,0])
        for key,value in extension.items():pieces[key].append(value)
        sources.append(dict(case_id=row['case_id'],source_report=str((replay/'RESULTS.json').relative_to(ROOT)),
            trace_path=row['trace_path'],trace_sha256=row['trace_sha256'],frames=len(selected),
            complete_episode_frames=len(choices),excluded_initialization_prefix_frames=row['frozen_prefix_frames'],
            excluded_inactive_arrival_frames=[len(choices)-1],safe_success=True,collision_free_complete_episode=True,
            source_is_offline_teacher=True,reference_is_previous_learning_model=True,
            original_safe_physics_and_sensor_only_replay_verified=True,
            original_physical_trace_sha256=row['physical_source_trace_sha256'],not_new_physical_success=True))
        source_id+=1;budget.tick()
    arrays={key:np.concatenate(value) for key,value in pieces.items()}
    for key,value in original.items():assert arrays[key][:len(value)].tobytes()==value.tobytes()
    with torch.no_grad():
        tensor=torch.from_numpy(arrays['observations']);features,baseline=actor.features(tensor)
        active=actor.support(tensor).numpy()
    nominal=np.abs(baseline.numpy()[:,None]-ACTIONS[None]).max(2).argmin(1)
    disagreements=np.flatnonzero((~active)&(nominal!=arrays['choices']))
    coverage=dict(active_frames=int(active.sum()),all_labels=histogram(arrays['choices']),
        parent_active_labels=histogram(original['choices'][active[:len(original['choices'])]]),
        active_labels=histogram(arrays['choices'][active]),
        active_resumed_motion_labels=histogram(arrays['choices'][active&(arrays['observations'][:,434]>=.4-1e-7)&(arrays['choices']!=0)]),
        inactive_target_disagreements=disagreements.tolist())
    write(output/'SOURCE_ADMISSION.json',coverage)
    if len(disagreements):
        write(output/'RESULTS.json',dict(completed=True,skipped=True,
            reason='Reference behavior differs outside trainable support; no frames silently discarded.',
            inactive_target_disagreements=disagreements.tolist(),new_ppo_steps=0,goal_achieved=False))
        return
    _,weighting=fit_weights(features.numpy()[active],arrays['choices'][active],
        arrays['observations'][active,434]*5,mode='recovery_phase')
    np.savez_compressed(output/'demonstrations.npz',**arrays)
    for name,expected in hashes.items():assert digest(ROOT/name)==expected,name
    budget.tick()
    result=dict(completed=True,at=datetime.now(timezone.utc).isoformat(),frames=len(tensor),parent_frames=meta['frames'],
        added_frames=len(tensor)-meta['frames'],sources=sources,dataset_sha256=digest(output/'demonstrations.npz'),
        original_prefix_byte_exact=True,failed_latest_trace_excluded=True,
        excluded_failed_trace_sha256s=meta['excluded_failed_trace_sha256s'],original_fresh_four_now_development=True,
        source_initialization_prefixes_excluded=True,training_coverage=coverage,training_weight_preflight=weighting,
        new_training_steps=0,new_physical_steps=0,new_ppo_steps=0,resource=budget.statistics(),goal_achieved=False,deployment_approved=False)
    write(output/'RESULTS.json',result);(output/'driver_snapshot.py').write_text(Path(__file__).read_text())
    print(json.dumps({k:v for k,v in result.items() if k!='sources'}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['parent','replay','model','output']:parser.add_argument('--'+name,type=Path,required=True)
    main(parser.parse_args())
