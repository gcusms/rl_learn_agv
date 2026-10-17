"""Attribute the completed v2 regression before changing policy or protection.

Replay real observations with their matching model contract. Reproduce the
original short forecast before comparing an existing registration alternative.
Alternative masks are diagnostic only, not permission to execute an action.
"""
from collections import Counter
from datetime import datetime
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
BASE = HERE.parent


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def intervals(flag, times, terminal):
    steps = np.diff(np.r_[times, terminal])
    indices = np.flatnonzero(flag)
    rows = []
    for part in np.split(indices, np.flatnonzero(np.diff(indices)>1)+1):
        if len(part):
            rows.append(dict(first_frame=int(part[0]), last_frame=int(part[-1]),
                start_s=float(times[part[0]]), seconds=float(steps[part].sum()), frames=len(part)))
    return rows


def summarize(z, row):
    times = z['labels'][:, 0]
    dt = np.diff(np.r_[times, row['seconds']])
    assert np.all(dt >= -1e-8)
    only = ~z['masks'][:, 1:].any(1)
    wait = (z['choices'] == 0) & ~only
    moving = z['choices'] != 0
    groups = z['observations'][:,303:399].reshape(-1,12,8).argmax(2)
    components = dict(only_stop_s=float(dt[only].sum()),
        voluntary_stop_s=float(dt[wait].sum()), nonzero_choice_s=float(dt[moving].sum()))
    assert abs(sum(components.values())-row['seconds']) < 1e-7
    return dict(seconds=row['seconds'], travel_m=row['travel_m'], components=components,
        only_stop_intervals=intervals(only,times,row['seconds']),
        voluntary_stop_intervals=intervals(wait,times,row['seconds']),
        only_stop_nonzero_rejection_groups={str(k):v for k,v in Counter(groups[only,1:].ravel().tolist()).items()},
        maximum_localization_xy_error_m=float(np.linalg.norm(z['labels'][:,1:3]-z['labels'][:,4:6],axis=1).max()),
        maximum_route_endpoint_error_m=float(np.linalg.norm(z['route_diagnostics'][:,2:4]-[7.125,5.725],axis=1).max()),
        route_reasons=dict(zip(*[a.tolist() for a in np.unique(z['route_reasons'],return_counts=True)])))


def main():
    assert not (HERE/'RESULTS.json').exists()
    post_path = BASE/'guard_aligned_mixed_ppo_resume_validation_20261017/RESULTS.json'
    old_path = BASE/'native625_local_speedup_retention_20261017/RESULTS.json'
    post, old = [json.loads(p.read_text()) for p in (post_path,old_path)]
    assert post['completed'] and post['new_physical_episodes'] == 30
    case_id = 'pre_ppo_holdout_dynamic_01'
    candidate = next(r for r in post['groups']['observed6']['rows'] if r['case_id']==case_id)
    baseline = next(r for r in old['frozen_rows'] if r['case_id']==case_id and r['variant']=='parent')
    collision_rows = [r for r in old['frozen_rows'] if r['case_id']=='pre_ppo_holdout_sensor_01']
    rows = [('dynamic01_bc4500',baseline),('dynamic01_v2_ppo1993',candidate)]
    rows += [('old_collision_'+r['variant'],r) for r in collision_rows]
    hardware_path = ROOT/'models/stop_feedback_20261015/guard_aligned_migration_s231017/portable/hardware.json'
    hardware = json.loads(hardware_path.read_text())
    paths = [Path(__file__),post_path,old_path,hardware_path]
    paths += [ROOT/r['trace_path'] for _,r in rows]
    for name in ['motion_prediction','predictive_constraint','contour_history','contour_translation',
                 'contour_velocity','track_uncertainty','directional_risk','body_motion','runtime',
                 'masked_actions','experiment_budget']:
        p=ROOT/'training/neural_navigation'/(name+'.py')
        if p.exists(): paths.append(p)
    hashes = {str(p.relative_to(ROOT)):digest(p) for p in paths}
    loaded = {}
    for name,row in rows:
        path = ROOT/row['trace_path']
        assert digest(path)==row['trace_sha256']
        with np.load(path,allow_pickle=False) as z:
            loaded[name] = {k:z[k].copy() for k in z.files}
    a,b = loaded['dynamic01_bc4500'],loaded['dynamic01_v2_ppo1993']
    n=min(len(a['choices']),len(b['choices']))
    first=int(np.flatnonzero(a['choices'][:n]!=b['choices'][:n])[0])
    parity={k:bool(np.array_equal(a[k][:first+1],b[k][:first+1],equal_nan=True)) for k in
        ['masks','scans','odometry','measured_velocity','labels','human_positions','route_diagnostics']}
    assert all(parity.values())
    assert np.array_equal(a['observations'][:first+1,:439],b['observations'][:first+1,:439])
    assert np.array_equal(a['observations'][:first+1,481:],b['observations'][:first+1,481:])
    assert np.array_equal(a['commands'][:first],b['commands'][:first])
    attribution = dict(first_divergence_frame=first,sensor_time_s=float(a['labels'][first,0]),
        baseline_action=int(a['choices'][first]),candidate_action=int(b['choices'][first]),
        common_sensor_physical_mask_and_route_inputs=parity,
        unchanged_observation_prefix_and_quality_suffix=True,
        risk_history_versions_differ_not_identical_model_inputs=True,
        baseline=summarize(a,baseline),candidate=summarize(b,candidate))
    attribution['extra_seconds_by_category'] = {k:attribution['candidate']['components'][k]-v
        for k,v in attribution['baseline']['components'].items()}
    write(HERE/'ATTRIBUTION.json',dict(completed=True,attribution=attribution,inputs=hashes,
        new_ppo_steps=0,new_physical_episodes=0,goal_achieved=False))

    from neural_navigation.predictive_constraint import PredictiveConstraint, registered_contract
    from neural_navigation.experiment_budget import ExperimentBudget
    budget=ExperimentBudget(.35,memory_mib=1200,directory=HERE)
    lidar_config=hardware['lidar']
    lidar=SimpleNamespace(**lidar_config)
    lidar.mount=np.asarray(lidar.mount,float)
    lidar.angles=np.linspace(lidar.angle_min,lidar.angle_max,lidar.beams)
    predictors=[]
    for name,row in rows:
        z=loaded[name]
        original=PredictiveConstraint(hardware['near_field_memory']['predictive_action_constraint'],
            hardware['robot_footprint_m'],.3)
        alternate=PredictiveConstraint(registered_contract(True),hardware['robot_footprint_m'],.3)
        counts=Counter();examples=[];maximum_contact_error=0.
        for i,stamp in enumerate(z['labels'][:,0]):
            core=SimpleNamespace(elapsed=float(stamp),odom=z['odometry'][i],scan=z['scans'][i],
                measured_velocity=z['measured_velocity'][i],lidar=lidar,odometry_sanity_failure=False)
            original.prepare(core);alternate.prepare(core)
            recorded=z['constraint_diagnostics'][i,3:15]
            np.testing.assert_array_equal(np.isfinite(original.contacts),np.isfinite(recorded))
            finite=np.isfinite(recorded)
            if finite.any():
                maximum_contact_error=max(maximum_contact_error,float(abs(original.contacts[finite]-recorded[finite]).max()))
                np.testing.assert_allclose(original.contacts[finite],recorded[finite],rtol=0,atol=1e-8)
            assert original.supported_tracks==int(z['constraint_diagnostics'][i,0])
            original_veto=np.array([original.rejection(core,c) is not None for c in original.commands])
            alternate_veto=np.array([alternate.rejection(core,c) is not None for c in alternate.commands])
            assert int(original_veto.sum())==int(z['constraint_diagnostics'][i,1])
            counts.update(frames=1,original_only_stop=int(original_veto[1:].all()),
                registered_only_stop=int(alternate_veto[1:].all()),
                original_vetoes=int(original_veto.sum()),registered_vetoes=int(alternate_veto.sum()),
                removed_vetoes=int((original_veto&~alternate_veto).sum()),
                added_vetoes=int((~original_veto&alternate_veto).sum()),
                original_stop_contact=int(np.isfinite(original.contacts[0])),
                registered_stop_contact=int(np.isfinite(alternate.contacts[0])))
            if ((name=='dynamic01_v2_ppo1993' and (i in [33,66,75,100,125,133,206,230,250]))
                    or name.startswith('old_collision_') and i>=len(z['choices'])-6):
                tracks=[]
                for t in original.tracks:
                    # Physical circles explain a replay only; never enter a predictor.
                    points=t['points'];radial=np.abs(np.linalg.norm(points[:,None,:]-z['human_positions'][i][None,:,:],axis=2)-.25)
                    tracks.append(dict(id=int(t['id']),source=t['source'],age_s=float(t['age']),hits=int(t['hits']),
                        speed_m_s=float(np.linalg.norm(t['velocity'])),
                        contour_position_uncertainty_m=float(t['contour_position_uncertainty_m']),
                        reachable_speed_hypothesis_m_s=float(t['reachable_speed_hypothesis_m_s']),
                        mean_circle_surface_mismatch_m=radial.mean(axis=0).tolist(),
                        circle_label_only_not_confirmed_association=True))
                examples.append(dict(frame=i,time_s=float(stamp),recorded_action=int(z['choices'][i]),
                    original_contacts=[float(v) if np.isfinite(v) else None for v in original.contacts],
                    registered_contacts=[float(v) if np.isfinite(v) else None for v in alternate.contacts],
                    original_vetoed_actions=np.flatnonzero(original_veto).tolist(),
                    registered_vetoed_actions=np.flatnonzero(alternate_veto).tolist(),original_tracks=tracks))
            if i%25==0:
                write(HERE/'STATUS.json',dict(status='actual_sensor_forecast_replay',source=name,
                    completed_frames=i+1,total_frames=len(z['choices'])))
            budget.tick()
        predictors.append(dict(source=name,source_collisions=row['collisions'],counts=dict(counts),
            original_contact_maximum_error=maximum_contact_error,examples=examples))
        write(HERE/'PREDICTOR_REPLAY_PARTIAL.json',dict(rows=predictors))
    write(HERE/'PREDICTOR_REPLAY.json',dict(completed=True,rows=predictors,
        baseline_forecast_reproduced=True,alternative_changes_permissions_if_adopted=True,
        no_physical_safety_or_generalization_claim=True,actual_object_positions_only_offline_labels=True,
        alternative_not_enabled=True,new_physical_episodes=0,new_ppo_steps=0,resource=budget.statistics()))

    import gc
    import torch
    from neural_navigation.masked_actions import ACTIONS
    torch.set_num_threads(1)
    model_specs=[('bc4500',ROOT/'models/stop_feedback_20261015/normal_suite_wait_pair_v2_s231016/wait_continuation/portable',a),
        ('aligned_migration',ROOT/'models/stop_feedback_20261015/guard_aligned_migration_s231017/portable',b),
        ('v2_ppo1993',ROOT/post['model_path']/'portable',b)]
    model_rows=[]
    for name,bundle,z in model_specs:
        path=bundle/'actor.pt';hashes[str(path.relative_to(ROOT))]=digest(path)
        actor=torch.jit.load(str(path),map_location='cpu').eval()
        choices=[];supports=[];logits=[]
        with torch.no_grad():
            for start in range(0,len(z['choices']),32):
                x=torch.from_numpy(z['observations'][start:start+32])
                outputs=actor(x).numpy()
                distances=np.max(abs(outputs[:,None]-ACTIONS[None]),axis=2)
                assert np.max(np.min(distances,axis=1))<2e-6
                choices.extend(distances.argmin(1).tolist())
                supports.extend(actor.support(x).tolist())
                logits.extend(actor.logits(x).numpy().tolist())
                budget.tick()
        choices=np.asarray(choices);supports=np.asarray(supports)
        if name in ['bc4500','v2_ppo1993']:
            np.testing.assert_array_equal(choices,z['choices'])
        waits=(z['choices']==0)&z['masks'][:,1:].any(1)
        model_rows.append(dict(model=name,actual_model_contract_inputs=True,
            recorded_action_replay_exact=name!='aligned_migration',
            at_first_common_state=dict(action=int(choices[first]),supported=bool(supports[first]),
                legal_actions=np.flatnonzero(z['masks'][first]).tolist(),logits=logits[first]),
            recorded_voluntary_wait_frames=int(waits.sum()),
            diagnostic_replay_motion_on_these_states=int(((choices!=0)&waits).sum()),
            replay_is_fixed_states_not_closed_loop=True))
        del actor;gc.collect()
    for name,expected in hashes.items():assert digest(ROOT/name)==expected,name
    result=dict(completed=True,at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
        attribution=attribution,model_replay=model_rows,predictor_replay='PREDICTOR_REPLAY.json',
        inputs=hashes,resource=budget.statistics(),
        next_change_must_distinguish_early_avoidable_and_late_only_stop_states=True,
        original_retention_efficiency_safety_late106_and_fresh_gates_preserved=True,
        alternative_not_enabled=True,new_ppo_steps=0,new_optimizer_updates=0,
        new_physical_episodes=0,goal_achieved=False,deployment_approved=False)
    write(HERE/'RESULTS.json',result);write(HERE/'STATUS.json',dict(status='complete',goal_achieved=False))
    print(json.dumps(dict(extra_seconds=attribution['extra_seconds_by_category'],model_replay=model_rows),ensure_ascii=False),flush=True)


if __name__=='__main__':
    try:
        main()
    except Exception as error:
        write(HERE/'FAILURE.json',dict(error=repr(error),no_auto_restart=True))
        raise
