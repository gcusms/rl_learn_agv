"""Find actual supported motion states that a stopping candidate fails to use.

Only completed, safe, unchanged-guard reference trajectories supply motion
labels. This is an input replay, not a physical recovery or a training run.
"""
from datetime import datetime
import gc
import hashlib
import json
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import torch

from experiments.stop_recovery_policy import StopRecoveryPPO, masks_tensor


ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
BASE = ROOT / 'reports/stop_feedback_20261015'
MODELS = ROOT / 'models/stop_feedback_20261015'
FOLDERS = ['w11_recovery_bc800_navigation_20261016',
           'w11_recovery_bc800_prefixes_20261016',
           'w11_recovery_bc800_w11_20261016',
           'w11_recovery_bc800_fresh_20261016']
CANDIDATES = {
    'reference': 'w11_recovery_bc800_s231015/policy_bc.zip',
    'control': 'w11_control2000_s231015/policy.zip',
    'retained': 'w11_retained2000_s231015/policy.zip',
    'diagnostic_bc': 'early_transfer_bc800_s231015/policy_bc.zip',
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(name, value):
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')


def main():
    if (OUT / 'RESULTS.json').exists():
        raise FileExistsError('Do not replace a completed state audit')
    torch.set_num_threads(1)
    inputs = {str(Path(__file__).relative_to(ROOT)): digest(Path(__file__))}
    observations, masks_all, choices_all, metadata = [], [], [], []
    for folder_name in FOLDERS:
        folder = BASE / folder_name
        path = folder / 'RESULTS.json'
        report = json.loads(path.read_text())
        if not report['completed']:
            raise ValueError('Only complete source reports are allowed')
        inputs[str(path.relative_to(ROOT))] = digest(path)
        for row in report['rows']:
            if (row['variant'] != 'w11_augmented_bc800' or not row['safe_success']
                    or row['collisions'] or row['false_arrival'] or row['physical_guards']
                    or not row.get('protocol_valid', True)):
                continue
            suffix = '' if row.get('recovery_test') else '_'+row['variant']
            path = folder / (row['case_id']+suffix+'.npz')
            if digest(path) != row['trace_sha256']:
                raise ValueError('Source trace changed')
            inputs[str(path.relative_to(ROOT))] = row['trace_sha256']
            with np.load(path, allow_pickle=False) as data:
                obs = data['observations']
                if obs.shape[1] != 439:
                    raise ValueError('Original directional feedback input required')
                train = np.c_[obs[:, :300], obs[:, 303:]].astype(np.float32)
                masks = data['masks'].astype(bool)
                choices = data['choices'].astype(int)
                commands = data['commands']
                times = data['labels'][:, 0]
                controlled = data['model_controlled'].astype(bool) if row.get('recovery_test') else np.ones(len(obs), bool)
                encoded = np.c_[np.ones(len(obs), bool), obs[:, 235:244]>.5, obs[:, 267:269]>.5]
                np.testing.assert_array_equal(encoded, masks)
                valid = (controlled & masks[:, 1:].any(-1) & (obs[:,435]>.5)
                    & (obs[:,438]>.5) & (np.linalg.norm(obs[:,218:220]*12.,axis=-1)>.3))
                for i in np.flatnonzero(valid):
                    metadata.append(dict(trace=str(path.relative_to(ROOT)),
                        report=str((folder/'RESULTS.json').relative_to(ROOT)),
                        trace_sha256=row['trace_sha256'],case_id=row['case_id'],
                        original_case_id=row.get('original_case_id',row['case_id']),
                        recovery_test=bool(row.get('recovery_test')),frame=int(i),sensor_time_s=float(times[i]),
                        actual_choice=int(choices[i]),actual_command=commands[i].tolist(),
                        legal_actions=np.flatnonzero(masks[i]).tolist(),
                        prior_legal_wait_s=float(obs[i,434]*5.),
                        nearest_lidar_m=float(obs[i,144:216].min()*12.),
                        source_safe_seconds=row['seconds'],source_safe_travel_m=row['travel_m']))
                observations.append(train[valid]);masks_all.append(masks[valid]);choices_all.append(choices[valid])
    obs_np = np.concatenate(observations)
    masks_np = np.concatenate(masks_all)
    choices_np = np.concatenate(choices_all)
    obs = torch.from_numpy(obs_np)
    masks = torch.from_numpy(masks_np)
    np.testing.assert_array_equal(masks_tensor(obs).numpy(),masks_np)
    predictions = {}
    for name, relative in CANDIDATES.items():
        path = MODELS / relative
        inputs[str(path.relative_to(ROOT))] = digest(path)
        model = StopRecoveryPPO.load(path, device='cpu')
        model.policy.set_training_mode(False)
        with torch.no_grad():
            distribution = model.policy.get_distribution(obs, masks)
            probability = distribution.distribution.probs.numpy().copy()
            predictions[name] = dict(choice=probability.argmax(-1),stop=probability[:,0])
            if name == 'reference':
                features = model.policy.mlp_extractor.policy_net(obs)
                baseline, supported = model.policy.action_net.parts(features)
                base = baseline.masked_fill(~masks, -1e8).numpy()
                base_choice = base.argmax(-1)
                base_stop_gap = base[:,0]-base[:,1:].max(-1)
                supported = supported.numpy()
                del features, baseline, base
            del distribution
        del model
        gc.collect()
    reference_replay = predictions['reference']['choice'] == choices_np
    motion = (choices_np != 0) & reference_replay & supported
    base_stop_motion = motion & (base_choice == 0)
    summaries = {}
    for name, prediction in predictions.items():
        summaries[name] = dict(greedy_stop_on_reference_motion=int((motion & (prediction['choice']==0)).sum()),
            greedy_stop_on_base_stop_reference_motion=int((base_stop_motion & (prediction['choice']==0)).sum()))
    selected = np.flatnonzero(base_stop_motion)
    selected = sorted(selected, key=lambda i:(
        -int(predictions['diagnostic_bc']['choice'][i]==0),
        -sum(int(predictions[n]['choice'][i]==0) for n in ('control','retained')),
        len(metadata[i]['legal_actions']),-float(base_stop_gap[i])))
    ranked = []
    per_trace = {}
    for i in selected:
        meta = metadata[i]
        if any(abs(meta['sensor_time_s']-t)<2. for t in per_trace.get(meta['trace'],[])):
            continue
        if len(per_trace.get(meta['trace'],[]))>=2:
            continue
        per_trace.setdefault(meta['trace'],[]).append(meta['sensor_time_s'])
        ranked.append(dict(meta,base_choice=int(base_choice[i]),base_stop_gap=float(base_stop_gap[i]),
            recovery_support=bool(supported[i]),model_inference={n:dict(choice=int(p['choice'][i]),
                stop_probability=float(p['stop'][i])) for n,p in predictions.items()}))
        if len(ranked)>=16:
            break
    # Replay the genuine late failed prefixes without inventing success labels.
    late = []
    folder = BASE / 'w11_retained2000_w11_20261016'
    report_path = folder/'RESULTS.json'
    report = json.loads(report_path.read_text())
    if not report['completed']:
        raise ValueError('Late failure source is incomplete')
    inputs[str(report_path.relative_to(ROOT))] = digest(report_path)
    for row in report['rows']:
        path=folder/(row['case_id']+'.npz')
        if digest(path)!=row['trace_sha256']:
            raise ValueError('Actual failure source changed')
        inputs[str(path.relative_to(ROOT))]=row['trace_sha256']
        with np.load(path,allow_pickle=False) as data:
            idx=np.flatnonzero(data['model_controlled'].astype(bool))[0]
            raw=data['observations'][idx]
            late.append(dict(case_id=row['case_id'],trace=str(path.relative_to(ROOT)),frame=int(idx),
                sensor_time_s=float(data['labels'][idx,0]),
                observation=np.r_[raw[:300],raw[303:]].astype(np.float32),
                mask=data['masks'][idx].astype(bool)))
    for name,relative in CANDIDATES.items():
        model=StopRecoveryPPO.load(MODELS/relative,device='cpu')
        model.policy.set_training_mode(False)
        with torch.no_grad():
            o=torch.from_numpy(np.stack([x['observation'] for x in late]));m=torch.from_numpy(np.stack([x['mask'] for x in late]))
            probs=model.policy.get_distribution(o,m).distribution.probs.numpy()
            active=model.policy.recovery_support(o).numpy()
            for index,item in enumerate(late):
                item.setdefault('models',{})[name]=dict(choice=int(probs[index].argmax()),
                    stop_probability=float(probs[index,0]),recovery_support=bool(active[index]))
        del model;gc.collect()
    for item in late:
        item['legal_actions']=np.flatnonzero(item.pop('mask')).tolist()
        item.pop('observation')
    write('RESULTS.json',dict(completed=True,at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
        analyzed_reference_frames=len(metadata),reference_replay_mismatches=int((~reference_replay).sum()),
        supported_actual_motion_frames=int(motion.sum()),
        base_stop_supported_actual_motion_frames=int(base_stop_motion.sum()),summaries=summaries,
        ranked_successful_motion_states=ranked,actual_failed_late_states=late,inputs=inputs,
        architecture_hard_logit_bound=False,new_physical_steps=0,new_ppo_steps=0,
        scope='Same-input policy replay on safe reference motion and actual failed prefixes; candidates are not physical recovery proof.',
        goal_achieved=False,deployment_approved=False))
    print(json.dumps(dict(frames=len(metadata),base_stop_motion=int(base_stop_motion.sum()),
        summaries=summaries,ranked_states=len(ranked),actual_failed_late_states=late),indent=2))


if __name__ == '__main__':
    main()
