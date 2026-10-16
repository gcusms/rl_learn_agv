"""Separate route projection, dynamic vetoes and native contact margin."""
from datetime import datetime
import hashlib
import json
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

from neural_navigation.decision_feedback import GROUPS, GROUP_DIM, RISK_SLICE, STAT_SLICE
from neural_navigation.experiment_budget import ExperimentBudget


ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
MODELS = ROOT/'models/stop_feedback_20261015/w11_retained2000_s231015'
REPLAY = OUT.parent/'post_aux_diagnostics_fixed_20261016/collision_replay'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    destination = OUT/'COLLISION_FORENSICS.json'
    if destination.exists():
        raise ValueError('Do not overwrite completed forensic evidence')
    budget = ExperimentBudget(.05,memory_mib=1024,directory=OUT)
    report_path = REPLAY/'RESULTS.json'
    replay = json.loads(report_path.read_text())
    if not replay['completed'] or not replay['all25sampled_states_masks_commands_and_rewards_reproduced']:
        raise ValueError('Complete exact physical replay required')
    inputs = {str(report_path.relative_to(ROOT)):digest(report_path)}
    trace_path = REPLAY/'TRACE.npz'
    if digest(trace_path)!=replay['raw_trace_sha256']:
        raise ValueError('Replay input changed')
    with np.load(trace_path,allow_pickle=False) as z:
        obs = z['observations'].copy();masks=z['masks'].astype(bool)
        replay_choices = z['choices'].copy()
    inputs[str(trace_path.relative_to(ROOT))]=digest(trace_path)
    only_stop = ~masks[:,1:].any(-1)
    first_only_stop = int(np.flatnonzero(only_stop)[0])
    groups = obs[:,300:300+GROUP_DIM].reshape(-1,12,len(GROUPS))
    risks = obs[:,300+RISK_SLICE.start:300+RISK_SLICE.stop]
    states=[]
    for i in range(max(0,first_only_stop-2),len(obs)):
        frame = replay['frames'][i]
        states.append(dict(sensor_timestamp_s=frame['sensor_timestamp_s'],
            sampled_choice=frame['sampled_choice'],legal_actions=frame['legal_actions'],
            executed_reason=frame['executed_reason'],
            action_rejection_groups={str(a):GROUPS[int(groups[i,a].argmax())] for a in range(12)},
            forecast_risk_by_action=risks[i].tolist(),
            predictor_input_valid=bool(obs[i,300+STAT_SLICE.start+4]>.5),
            actual_robot_velocity=frame['actual_velocity'],
            nearest_physical_human_gap_m=min(frame['robot_to_human_gap_m'])))
    contacts=replay['native_contacts']
    distances=np.array([c['signed_distance_m'] for c in contacts])
    if not len(distances):
        raise ValueError('Native contact records are missing')
    physical = dict(native_contact_records=len(contacts),
        minimum_signed_distance_m=float(distances.min()),
        maximum_signed_distance_m=float(distances.max()),
        nonpositive_native_contact_distances=int((distances<=0).sum()),
        first_contact=replay['first_native_contact'],
        physical_guard_activations=replay['physical_guards'],
        original_collision_counter_unchanged=replay['collisions'],
        interpretation='Robot/pedestrian native contact buffer activated at positive separation; original conservative failure label retained.')
    observed=[];choices=[];starts=[];records=[];log_probs=[]
    for path in sorted((MODELS/'rollouts').glob('*.npz')):
        with np.load(path,allow_pickle=False) as z:
            observed.extend(z['observations'][:,0]);choices.extend(z['sampled_actions'][:,0,0])
            starts.extend(z['episode_starts'][:,0]);log_probs.extend(z['old_log_prob'][:,0])
            records.extend(json.loads(str(s)) for s in z['execution_diagnostics_json'][:,0])
        inputs[str(path.relative_to(ROOT))]=digest(path)
        budget.tick()
    starts=np.flatnonzero(starts);observed=np.asarray(observed);choices=np.asarray(choices,int)
    a,b=int(starts[0]),int(starts[1]);c,d=int(starts[6]),int(starts[7])
    np.testing.assert_array_equal(choices[c:d],replay_choices)
    np.testing.assert_array_equal(observed[a],observed[c])
    episodes_path=MODELS/'episodes.jsonl'
    episodes=[json.loads(s) for s in episodes_path.read_text().splitlines()]
    if not episodes[0]['success'] or episodes[0]['collisions'] or not episodes[6]['collisions']:
        raise ValueError('Original successful and failed training episodes changed')
    warmups=[MODELS/records[i]['initialization_trace'] for i in [a,c]]
    with np.load(warmups[0],allow_pickle=False) as first,np.load(warmups[1],allow_pickle=False) as seventh:
        for key in ['scans','odometry','measured_velocity','physical_poses','times','choices','commands']:
            np.testing.assert_allclose(first[key],seventh[key],rtol=0,atol=2e-8,equal_nan=True)
    contrast=dict(same30_frame_physical_and_sensor_history_verified=True,
        first_observation436_exactly_equal=True,
        successful_training_episode=dict(number=1,actual_steps=b-a,
            first_choice=int(choices[a]),first_sample_probability=float(np.exp(log_probs[a])),
            physical_success=True,collisions=0),
        failed_training_episode=dict(number=7,actual_steps=d-c,
            first_choice=int(choices[c]),first_sample_probability=float(np.exp(log_probs[c])),
            physical_success=False,collisions=replay['collisions']),
        single_action_causal_trial=False,
        limitation='Both complete outcomes are recorded on-policy training exposure. Policy weights and later samples differed; the first action alone is not an isolated causal intervention.')
    for p in [episodes_path,*warmups]:
        inputs[str(p.relative_to(ROOT))]=digest(p)
    budget.tick()
    result=dict(completed=True,at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
        first_only_stop_sensor_s=replay['frames'][first_only_stop]['sensor_timestamp_s'],
        only_stop_frames=int(only_stop.sum()),frame_diagnostics=states,
        contact_semantics=physical,actual_training_contrast=contrast,
        conclusions=['The emitted stop reason is not the rejection reason for every possible motion.',
            'At8.8s translation is projected for incomplete routing, while both pure rotations are independently vetoed by dynamic prediction.',
            'The current guard already preserves geometrically/predictively admissible pure rotation during incomplete routing; no unconditional rotation permission is justified.',
            'At the final forced-stop states the recovery policy has no permitted moving action; earlier decision quality and arrival at these states must be evaluated.',
            'A complete safe earlier training trajectory exists from exactly the same physical initialization, but is not an independent navigation score.'],
        inputs_sha256=inputs,source_sha256=digest(Path(__file__)),
        additional_environment_steps=0,optimizer_updates=0,
        runtime_masks_rewards_or_failure_labels_changed=False,
        resource=budget.statistics(),goal_achieved=False,deployment_approved=False)
    destination.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(completed=True,first_only_stop_sensor_s=result['first_only_stop_sensor_s'],
        contact_semantics=physical,actual_training_contrast=contrast,
        resource=result['resource']),indent=2),flush=True)


if __name__=='__main__':
    main()
