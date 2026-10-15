"""Check whether the repeated-rejection cost fires under sampled PPO actions.

Uses a previously recorded training failure, not a frozen acceptance task.
Reward scoring can use simulated truth; policy inputs remain sensor-only.
"""
import argparse
import copy
from collections import Counter
import json
from pathlib import Path
import numpy as np
import torch
from stable_baselines3 import PPO
from experiments.recovery_curriculum import RejectedTranslationCost
from neural_navigation.runtime import SensorNavigationRuntime
from neural_navigation.stress_env import StressNavigationEnv
from neural_navigation.training_distance import TrainingDistanceGraph
from navrl.geometry import ROOT


class CapturePolicy:
    def __init__(self,model,deterministic):
        self.model,self.deterministic=model,deterministic
    def predict(self,observation):
        self.observation=observation.copy()
        return self.model.predict(observation,deterministic=self.deterministic)[0]


def inspect_probes(runtime,timestamp):
    """Read-only denser checks; accepted probes are never executed."""
    core=runtime.core
    checks=[]
    for v in (0.,.01,.025,.05,.1,.2):
        for w in (-.5,-.25,-.1,0.,.1,.25,.5):
            if v==0 and w==0:continue
            proposal=np.array([v,w])
            command,reason=copy.deepcopy(runtime.guard).check(core,proposal,timestamp,timestamp)
            checks.append(dict(proposal=proposal.tolist(),executed=command.tolist(),reason=reason,
                fully_accepted=bool(np.allclose(command,proposal,atol=1e-9,rtol=0))))
    return dict(timestamp=timestamp,estimate=core.estimate.tolist(),
        current_inflated_map_free=bool(core.static_map.lookup(core.estimate[:2],core.static_map.free,outside=False)),
        current_map_clearance_m=float(core.static_map.lookup(core.estimate[:2])),
        original_context_probe_count=9,additional_grid=checks,
        limitation='Read-only present-frame checks, not future dynamic safety or exhaustive continuous feasibility.')


def run(source,bundle,model,deterministic,max_steps,inspect=False):
    scene=Path(source['scene']);case=source['interventions'];options=source['options']
    env=StressNavigationEnv(case=case,components=bundle,scene_dir=scene,route_mode='shortcut',max_steps=max_steps)
    rows=[];counts=Counter();cost=RejectedTranslationCost();snapshots=[];inspected=set()
    try:
        env.reset(seed=source['seed'],options=options)
        prior=np.asarray(options['start'])+np.asarray(options.get('initial_prior_offset',[0,0,0]))
        runtime=SensorNavigationRuntime(bundle,scene/'decision_map.yaml',prior,options['goal'],'shortcut')
        capture=CapturePolicy(model,deterministic);runtime.policy=capture
        field=TrainingDistanceGraph(runtime.core.static_map).field(options['goal'])
        previous=field.value(env.truth[:2]);arrived=False
        for _ in range(max_steps):
            result=runtime.step(env.scan,env.observed_odometry,env.measured_velocity,env.elapsed)
            if inspect and not any(result['action_feasibility']) and result['stop_reason'] not in inspected:
                # Diagnose each distinct rejecting reason at most once.
                snapshots.append(inspect_probes(runtime,env.elapsed))
                inspected.add(result['stop_reason'])
            if result['stop_reason']=='estimated_goal_reached':
                arrived=bool(np.linalg.norm(env.truth[:2]-env.goal)<=.2);break
            command=np.asarray(result['command'])
            executed=np.array([command[0]/(.2 if command[0]>=0 else .08),command[1]/.5])
            env.step(executed)
            distance=field.value(env.truth[:2]);progress=previous-distance;previous=distance
            info=dict(proposed_action=result['proposed_action'],commanded_velocity=command,
                execution_guard_reason=result['stop_reason'],training_actions_overridden=result['safety_override'],
                fixed_geodesic_progress_m=progress)
            penalty=cost.step(capture.observation,info,env.dt)
            counts[result['stop_reason'] or 'none']+=1
            proposal=np.asarray(result['proposed_action'])
            rows.append(dict(timestamp=env.elapsed,proposal=proposal.tolist(),command=command.tolist(),
                guard_reason=result['stop_reason'],feasibility=result['action_feasibility'],
                geodesic_progress_m=progress,rejection_seconds=cost.duration,rejection_penalty=penalty,
                original_projection_cost=float(.25*np.sum((proposal-executed)**2))))
            if env.collision_events or env.hard_guard_activations:break
        return dict(deterministic=deterministic,steps=len(rows),arrived=arrived,
            collisions=env.collision_events,guard_reasons=dict(counts),
            penalty_steps=sum(x['rejection_penalty']>0 for x in rows),
            penalty_sum=sum(x['rejection_penalty'] for x in rows),
            projection_cost_sum=sum(x['original_projection_cost'] for x in rows),
            feasibility_snapshots=snapshots,samples=rows)
    finally:env.close()


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--episodes',type=Path,required=True)
    p.add_argument('--policy',type=Path,required=True)
    p.add_argument('--bundle',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--max-steps',type=int,default=700)
    p.add_argument('--mode',choices=('both','sampled','deterministic'),default='both')
    p.add_argument('--inspect-probes',action='store_true')
    a=p.parse_args()
    if a.output.exists():p.error('Fresh audit output required')
    episodes=[json.loads(x) for x in a.episodes.read_text().splitlines() if x]
    failure=next((x for x in episodes if not x['verified_mission_success']),None)
    if failure is None:p.error('Recorded failed training episode required')
    source=json.loads(failure['training_case_json'])
    torch.set_num_threads(1);model=PPO.load(a.policy,device='cpu')
    reports=[]
    modes=(True,False) if a.mode=='both' else (a.mode=='deterministic',)
    for deterministic in modes:
        torch.manual_seed(92201)
        report=run(source,a.bundle.resolve(),model,deterministic,a.max_steps,a.inspect_probes)
        reports.append(report);print(json.dumps({k:v for k,v in report.items() if k not in ('samples','feasibility_snapshots')}),flush=True)
    a.output.write_text(json.dumps(dict(source_training_episode=failure,reports=reports,
        limitation='Audits reward activation under sampled actions; not a training update or an acceptance result.'),indent=2))


if __name__=='__main__':main()
