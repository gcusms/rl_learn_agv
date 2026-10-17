"""Add a complete previously successful trace to the earlier-release dataset."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import torch

from experiments.build_release_recovery_sources import training_indices
from experiments.long_risk_actor import digest
from experiments.release_fit_weights import fit_weights
from experiments.release_actor import ReleaseActor
from neural_navigation.experiment_budget import ExperimentBudget
from neural_navigation.masked_actions import ACTIONS, COMMANDS, masks_from_observation
from neural_navigation.release_observation import augment, contract
from neural_navigation.sensors import LidarSpec


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'reports/stop_feedback_20261015'


def main(args):
    output = args.output.resolve();output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    budget = ExperimentBudget(.15, memory_mib=1536, directory=output)
    parent = BASE / 'sensor_release_sources_20261016'
    reference = ROOT / 'models/stop_feedback_20261015/release_phase300_s231016'
    good_path = BASE / 'release_phase_validation_20261016/RESULTS.json'
    failed_path = BASE / 'sensor_release_validation_20261016/RESULTS.json'
    meta, good, failed = [json.loads(p.read_text()) for p in [parent / 'RESULTS.json', good_path, failed_path]]
    assert all(r['completed'] for r in [meta,good,failed])
    assert digest(parent / 'demonstrations.npz') == meta['dataset_sha256']
    case_id = 'release_phase_frozen_noise_05_2'
    old = next(r for r in good['rows'] if r['case_id'] == case_id and r['variant'] == 'recovery_phase')
    new = next(r for r in failed['rows'] if r['case_id'] == case_id and r['variant'] == 'candidate')
    assert old['safe_success'] and (not new['safe_success']
        or new['seconds'] > old['seconds']*1.2 or new['travel_m'] > old['travel_m']*1.2)
    assert old['physical_arrival'] and old['protocol_valid'] and old['learned_model_only'] and old['no_teacher']
    assert not any(old[k] for k in ['collisions','timeout','false_arrival','physical_guards'])
    path = ROOT / old['trace_path'];assert digest(path) == old['trace_sha256']
    files = [Path(__file__), parent / 'RESULTS.json', parent / 'demonstrations.npz', good_path,
        failed_path, path, reference / 'checkpoint.pt', reference / 'parent_actor.pt',
        reference / 'portable/actor.pt', reference / 'portable/hardware.json']
    hashes = {str(p.relative_to(ROOT)):digest(p) for p in files}
    (output / 'PLAN.json').write_text(json.dumps(dict(inputs=hashes,
        only_previously_complete_safe_episode_admitted=True,
        new_failure_not_a_positive_example=True, previous_arrays_preserved=True,
        observation_contract=contract(), new_physical_episodes=0, new_optimizer_updates=0),indent=2)+'\n')
    with np.load(parent / 'demonstrations.npz',allow_pickle=False) as data:
        original={k:data[k].copy() for k in data.files}
    with np.load(path,allow_pickle=False) as data:
        trace={k:data[k].copy() for k in data.files}
    obs, choices, masks=trace['observations'],trace['choices'],trace['masks']
    np.testing.assert_array_equal(masks_from_observation(obs[:,:282]),masks)
    assert masks[np.arange(len(masks)),choices].all()
    actor=ReleaseActor(torch.jit.load(str(reference/'parent_actor.pt'),map_location='cpu')).eval()
    actor.load_state_dict(torch.load(reference/'checkpoint.pt',map_location='cpu',weights_only=False)['actor'])
    saved=torch.jit.load(str(reference/'portable/actor.pt'),map_location='cpu').eval()
    with torch.no_grad():
        x=torch.from_numpy(obs)
        np.testing.assert_array_equal(saved(x).numpy(),ACTIONS[choices])
        np.testing.assert_array_equal(actor(x).numpy(),ACTIONS[choices])
        support=actor.support(x).numpy()
    lidar=LidarSpec(json.loads((reference/'portable/hardware.json').read_text())['lidar'])
    for observation,scan in zip(obs,trace['scans']):
        np.testing.assert_array_equal(augment(observation[:481],scan,lidar),observation)
    selected=training_indices(trace,old,support)
    np.testing.assert_allclose(trace['commands'][selected],COMMANDS[choices[selected]],rtol=0,atol=2e-8)
    extension=dict(observations=obs[selected],choices=choices[selected],masks=masks[selected],
        source_ids=np.full(len(selected),int(original['source_ids'].max())+1,np.int64),
        sensor_times=trace['labels'][selected,0])
    arrays={k:np.concatenate([v,extension[k]]) for k,v in original.items()}
    for k,v in original.items():assert arrays[k][:len(v)].tobytes()==v.tobytes()
    with torch.no_grad():
        x=torch.from_numpy(arrays['observations']);features,baseline=actor.features(x);active=actor.support(x).numpy()
    nominal=np.abs(baseline.numpy()[:,None]-ACTIONS[None]).max(2).argmin(1)
    assert not ((~active)&(nominal!=arrays['choices'])).any()
    _,weighting=fit_weights(features.numpy()[active],arrays['choices'][active],
        arrays['observations'][active,434]*5,mode='recovery_phase')
    sources=list(meta['sources'])
    sources.append(dict(case_id=case_id,source_report=str(good_path.relative_to(ROOT)),trace_path=old['trace_path'],
        trace_sha256=old['trace_sha256'],frames=len(selected),complete_episode_frames=len(choices),
        excluded_inactive_arrival_frames=[len(choices)-1],safe_success=True,collision_free_complete_episode=True,
        source_is_offline_teacher=False,retention_source_after_regression_diagnosis=True))
    excluded=sorted(set(meta['excluded_failed_trace_sha256s']+[new['trace_sha256']]))
    assert old['trace_sha256'] not in excluded
    np.savez_compressed(output/'demonstrations.npz',**arrays)
    for name,expected in hashes.items():assert digest(ROOT/name)==expected,name
    budget.tick()
    result=dict(completed=True,at=datetime.now(timezone.utc).isoformat(),frames=len(x),
        parent_frames=meta['frames'],added_frames=len(selected),sources=sources,
        dataset_sha256=digest(output/'demonstrations.npz'),original_prefix_byte_exact=True,
        failed_latest_trace_excluded=True,excluded_failed_trace_sha256s=excluded,
        original_fresh_four_now_development=True,training_weight_preflight=weighting,
        new_training_steps=0,new_physical_steps=0,new_ppo_steps=0,resource=budget.statistics(),
        goal_achieved=False,deployment_approved=False)
    (output/'RESULTS.json').write_text(json.dumps(result,indent=2)+'\n')
    (output/'driver_snapshot.py').write_text(Path(__file__).read_text())
    print(json.dumps({k:v for k,v in result.items() if k!='sources'},indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    main(parser.parse_args())
