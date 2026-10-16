"""Exact replay; separate current laser, remembered points and static geometry."""
import argparse
import copy
import json
from pathlib import Path
import numpy as np
import torch
import mujoco
from navrl.geometry import ROOT
from neural_navigation.stress_env import StressNavigationEnv
from neural_navigation.masked_actions import COMMANDS
from neural_navigation.static_footprint import static_footprint_clear
from neural_navigation.safety_memory import NearFieldMemory
from experiments.time_consistent_tracking import TimeConsistentFreshRuntime
from experiments.train_quality_navigation import digest


def main():
    parser=argparse.ArgumentParser()
    for name in ('bundle','trace','output'):parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():parser.error('Fresh output required')
    torch.set_num_threads(1);trace=json.loads(args.trace.read_text());case=trace['case'];scene=ROOT/case['scene']
    env=StressNavigationEnv(case=case,components=args.bundle,scene_dir=scene,route_mode='shortcut',max_steps=700)
    try:
        env.reset(seed=case['seed'],options={k:case[k] for k in ('start','goal','initial_prior_offset') if k in case})
        runtime=TimeConsistentFreshRuntime(args.bundle,scene/'decision_map.yaml',
            np.asarray(case['start'])+np.asarray(case.get('initial_prior_offset',[0,0,0])),case['goal'],'shortcut')
        confirmed_memory=NearFieldMemory(dict(runtime.guard.memory.config,
            free_evidence_margin_m=.03,free_evidence_frames=3))
        diagnosis=None
        for index,row in enumerate(trace['samples']):
            result=runtime.step(env.scan,env.observed_odometry,env.measured_velocity,env.elapsed,capture_observation=True)
            np.testing.assert_allclose(result['policy_observation'],row['policy_observation'],atol=2e-6,rtol=0)
            np.testing.assert_allclose(result['command'],row['command'],atol=2e-8,rtol=0)
            if result['stop_reason']!=row['stop_reason']:raise ValueError('Replay feedback differs')
            confirmed_memory.update(runtime.core)
            if not any(row['mask'][1:]):
                core=runtime.core;guard=runtime.guard;probes=[]
                for command in COMMANDS:
                    variants={}
                    for label in ('native_memory','current_laser_same_escape_shadow','rotation_escape_shadow',
                                  'confirmed_free_memory_shadow','confirmed_free_rotation_shadow'):
                        shadow=copy.deepcopy(guard)
                        if label=='current_laser_same_escape_shadow':
                            class EmptyMemory:
                                def update(self,unused):return np.empty((0,2)),np.empty(0)
                                def reset(self):pass
                            shadow.memory=EmptyMemory()
                        if label in ('confirmed_free_memory_shadow','confirmed_free_rotation_shadow'):
                            shadow.memory=copy.deepcopy(confirmed_memory)
                        if label in ('rotation_escape_shadow','confirmed_free_rotation_shadow'):
                            shadow.clearance_increasing_rotation=True
                        executed,reason=shadow.check(core,command,env.elapsed,env.elapsed)
                        variants[label]=dict(accepted=bool(np.allclose(executed,command,atol=1e-9,rtol=0)),reason=reason)
                    probes.append(dict(command=command.tolist(),**variants))
                entries=guard.memory.entries if guard.memory else {}
                # Memory is stored in odometry coordinates, not map coordinates.
                oc,os=np.cos(core.odom[2]),np.sin(core.odom[2]);odom_rotation=np.array([[oc,-os],[os,oc]])
                mc,ms=np.cos(core.estimate[2]),np.sin(core.estimate[2]);map_rotation=np.array([[mc,-ms],[ms,mc]])
                remembered=[]
                for point,stamp in entries.values():
                    body=(point-core.odom[:2])@odom_rotation
                    mapped=core.estimate[:2]+body@map_rotation.T
                    remembered.append(dict(odometry_point=np.asarray(point).tolist(),body_point=body.tolist(),
                        estimated_map_point=mapped.tolist(),age_s=float(env.elapsed-stamp),
                        estimated_map_distance_m=float(core.static_map.lookup(mapped,core.static_map.distance,outside=0.))))
                valid=np.isfinite(core.scan)&(core.scan>=core.lidar.range_min)&(core.scan<core.lidar.range_max)
                beam_indices=np.flatnonzero(valid);angles=core.lidar.angles[valid]+core.lidar.mount[3]
                endpoints=core.scan[valid,None]*np.c_[np.cos(angles),np.sin(angles)]+core.lidar.mount[:2]
                gaps=np.linalg.norm(endpoints-np.clip(endpoints,-guard.half,guard.half),axis=1)
                # Simulator ray IDs/ranges are scoring only, after all decisions.
                mount=core.lidar.mount;yaw=env.truth[2]
                origin=np.array([env.truth[0]+np.cos(yaw)*mount[0]-np.sin(yaw)*mount[1],
                    env.truth[1]+np.sin(yaw)*mount[0]+np.cos(yaw)*mount[1],mount[2]])
                ray_angles=core.lidar.angles+yaw+mount[3]
                directions=np.c_[np.cos(ray_angles),np.sin(ray_angles),np.zeros(len(ray_angles))].reshape(-1)
                ids=np.zeros(len(ray_angles),np.int32);ranges=np.zeros(len(ray_angles))
                mujoco.mj_multiRay(env.model,env.data,origin,directions,np.array([1,0,0,0,0,0],np.uint8),
                    1,env.robot,ids,ranges,len(ray_angles),core.lidar.range_max)
                closest=[]
                for offset in np.argsort(gaps)[:8]:
                    beam=int(beam_indices[offset]);angle=core.lidar.angles[beam]+mount[3]
                    noiseless=mount[:2]+ranges[beam]*np.array([np.cos(angle),np.sin(angle)])
                    closest.append(dict(beam=beam,measured_body_gap_m=float(gaps[offset]),
                        measured_range_m=float(core.scan[beam]),noiseless_range_score_only_m=float(ranges[beam]),
                        noiseless_body_gap_score_only_m=float(np.linalg.norm(noiseless-np.clip(noiseless,-guard.half,guard.half))),
                        hit_geom_score_only=mujoco.mj_id2name(env.model,mujoco.mjtObj.mjOBJ_GEOM,int(ids[beam])) if ids[beam]>=0 else None))
                diagnosis=dict(frame=index,timestamp=float(env.elapsed),estimate=core.estimate.tolist(),
                    truth_score_only=env.truth.tolist(),pose_error_score_only_m=float(np.linalg.norm(core.estimate[:2]-env.truth[:2])),
                    estimated_static_body_clear=bool(static_footprint_clear(core.static_map,core.estimate,guard.half*2,.04)),
                    true_static_body_clear_score_only=bool(static_footprint_clear(core.static_map,env.truth,guard.half*2,.04)),
                    quality_flags=row['policy_observation'][294:],probes=probes,remembered_points=remembered,
                    closest_measured_returns=closest,confirmed_free_memory_entries=len(confirmed_memory.entries),
                    confirmed_free_shadow_config=dict(free_evidence_margin_m=.03,free_evidence_frames=3),
                    measured_velocity=core.measured_velocity.tolist(),localization_std=core.localizer.std.tolist(),
                    scan_valid_beam_indices=beam_indices.tolist(),scan_valid_ranges=core.scan[valid].tolist(),
                    odometry_pose=core.odom.tolist())
                break
            v,w=result['command'];env.step(np.array([v/(.2 if v>=0 else .08),w/.5]))
            if env.collision_events or env.hard_guard_activations:raise ValueError('Physical failure before rejected frame')
        if diagnosis is None:raise ValueError('No all-rejected frame reproduced')
        report=dict(trace_sha256=digest(args.trace),reproduced_actual_observation_command_and_reason=True,
            diagnosis=diagnosis,deployment_approved=False,
            limitation='Memory removal is diagnostic shadow only, never executed. A shadow accepted command is not proven safe; remembered rear hazards may be real. Truth/ray IDs only score geometry after decisions. Corrected odometry-to-map memory conversion; original v1 world-point map distances were invalid and must not be interpreted.')
        args.output.write_text(json.dumps(report,indent=2,allow_nan=False))
        print(json.dumps({k:v for k,v in diagnosis.items() if k!='remembered_points'}),flush=True)
    finally:env.close()


if __name__=='__main__':main()
