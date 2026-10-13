"""Fixed diagnostic cases. Classical reference paths are offline metrics only."""
import argparse
import hashlib
import json
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from navrl.geometry import ROOT,wrap
from neural_navigation.room_env import RoomNavigationEnv
from navigation_comparison import ComparisonEnv
from neural_navigation.stress_env import StressNavigationEnv


class FixedCase(gym.Wrapper):
    def __init__(self, case, components, max_steps,route_mode):
        torch.set_num_threads(1)
        self.case = case
        max_steps=int(case.get('max_steps',max_steps))
        dynamic = case['category'] in ('triggered_obstacle','continuous_mission')
        scene=ROOT/case.get('scene','assets/navigation_comparison' if dynamic else 'assets/room_navigation_scripted')
        if dynamic:env=ComparisonEnv(components=components,scene_dir=scene,max_steps=max_steps,route_mode=route_mode)
        elif any(k in case for k in ['obstacle','dropout_windows','odom_scale','extra_scan_dropout']):
            env=StressNavigationEnv(case=case,components=components,scene_dir=scene,max_steps=max_steps,route_mode=route_mode)
        else:env=RoomNavigationEnv(components=components,scene_dir=scene,dynamic=False,max_steps=max_steps,route_mode=route_mode)
        super().__init__(env)
        self.completed = False
        self.initialized = False
        self.mission_targets=0;self.leg_started_step=0

    def reset(self, **kwargs):
        if self.initialized:
            return self.observation.copy(), {}
        self.initialized = True
        options = {k: self.case[k] for k in ('start', 'goal','initial_prior_offset') if k in self.case}
        self.observation, info = self.env.reset(seed=self.case['seed'], options=options)
        self.errors = [float(info['localization_error_m'])]
        self.yaw_errors=[abs(float(wrap(self.env.estimate[2]-self.env.truth[2])))]
        self.low_alignment_steps=0
        self.step_times = []
        self.turning_steps=0;self.waiting_steps=0;self.forward_speeds=[]
        self.route_checks = 0
        self.route_failures = 0
        self.started = time.perf_counter()
        # Evaluator-only reference: never inserted into robot path/observations.
        try:
            reference = self.env.static_map.plan(self.env.truth[:2], self.env.goal)
            self.reference_length = float(np.linalg.norm(np.diff(reference, axis=0), axis=1).sum())
        except ValueError:
            self.reference_length = None
        return self.observation.copy(), info

    def step(self, action):
        if self.completed:
            return self.observation.copy(), 0., False, False, {}
        started = time.perf_counter()
        self.observation, reward, done, truncated, info = self.env.step(action)
        self.step_times.append(time.perf_counter()-started)
        self.errors.append(float(info['localization_error_m']))
        self.yaw_errors.append(abs(float(wrap(self.env.estimate[2]-self.env.truth[2]))))
        self.low_alignment_steps+=self.env.localizer.map_agreement<.35
        velocity=np.asarray(info['actual_velocity'])
        if abs(velocity[0])<.02:
            if abs(velocity[1])>.1:self.turning_steps+=1
            else:self.waiting_steps+=1
        elif velocity[0]>0:self.forward_speeds.append(float(velocity[0]))
        self.route_checks += 1
        if np.linalg.norm(self.env.path[-1]-self.env.goal) > .12:
            self.route_failures += 1
        continuous=self.case['category']=='continuous_mission'
        if continuous:
            if done and info['true_goal_distance_m']<=.30:
                self.mission_targets+=1
                if self.mission_targets<4 and not truncated:
                    self.observation=self.env.select_goal(self.mission_targets);done=False
                    self.leg_started_step=len(self.step_times)
            if len(self.step_times)-self.leg_started_step>=700 and not done:truncated=True
        if done or truncated:
            self.completed = True
            arrived = bool(done and info['true_goal_distance_m'] <= .30 and (not continuous or self.mission_targets==4))
            safe = self.env.collision_events == 0 and self.env.hard_guard_activations == 0
            event=info.get('stress_intervention')
            valid_protocol=not self.case.get('obstacle') or event is not None
            if 'reverse_after_event_s' in self.case:valid_protocol=valid_protocol and event is not None and 'reversal_time_s' in event
            if 'actor_pause_after_event_s' in self.case:valid_protocol=valid_protocol and event is not None and 'pause_completed_time_s' in event
            info['diagnostic_result'] = dict(
                case_id=self.case['id'], category=self.case['category'], arrived=arrived,
                safe_success=arrived and safe and valid_protocol, timeout=bool(truncated),protocol_valid=bool(valid_protocol),
                false_arrival=bool(done and not arrived),
                final_true_goal_distance_m=float(info['true_goal_distance_m']),
                collisions=int(self.env.collision_events), guards=int(self.env.hard_guard_activations),
                seconds=float(self.env.elapsed), travel_m=float(self.env.travel),
                localization_error_p95_m=float(np.percentile(self.errors,95)),
                localization_error_max_m=float(max(self.errors)),
                localization_yaw_error_p95_rad=float(np.percentile(self.yaw_errors,95)),
                localization_yaw_error_max_rad=float(max(self.yaw_errors)),
                low_alignment_fraction=self.low_alignment_steps/max(len(self.step_times),1),
                localization_recovery_attempts=self.env.localizer.recovery_attempts,
                localization_recovery_accepted=self.env.localizer.recovery_accepted,
                route_incomplete_fraction=self.route_failures/max(self.route_checks,1),
                environment_step_p95_ms=float(np.percentile(self.step_times,95)*1000),
                offline_static_reference_path_m=self.reference_length,
                router_cache_hits=self.env.router_cache_hits,
                router_cache_misses=self.env.router_cache_misses,mission_targets=self.mission_targets,
                turning_seconds=self.turning_steps*self.env.dt,waiting_seconds=self.waiting_steps*self.env.dt,
                mean_moving_forward_m_s=float(np.mean(self.forward_speeds)) if self.forward_speeds else 0.,
                stress_intervention=event,sensor_fault_steps=info.get('sensor_fault_steps',0))
        return self.observation.copy(), reward, done, truncated, info


def factory(case, components, max_steps,route_mode):
    return lambda: FixedCase(case, components, max_steps,route_mode)


def evaluate_batch(model,cases,components,max_steps,route_mode):
    env=SubprocVecEnv([factory(c,components,max_steps,route_mode) for c in cases],start_method='spawn')
    results={}
    try:
        obs=env.reset()
        for _ in range(max(int(c.get('max_steps',max_steps)) for c in cases)):
            # Use the same actor arithmetic as single-robot deployment. Changing
            # BLAS batch shape can perturb actions and amplify route-cell ties.
            action=np.stack([model.predict(row,deterministic=True)[0] for row in obs])
            obs,_,_,infos=env.step(action)
            for info in infos:
                if 'diagnostic_result' in info:
                    result=info['diagnostic_result'];results[result['case_id']]=result
                    print('CASE',json.dumps(result),flush=True)
            if len(results)==len(cases):break
    finally:env.close()
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--policy', type=Path, required=True)
    parser.add_argument('--components', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, default=ROOT/'config/navigation_diagnostic_v1.json')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-steps', type=int, default=700)
    parser.add_argument('--workers',type=int,default=12)
    parser.add_argument('--route-mode',choices=['raw','shortcut'],default='raw')
    args = parser.parse_args()
    if args.max_steps <= 0 or args.workers<=0: parser.error('max-steps and workers must be positive')
    if args.output.exists(): parser.error('Use a fresh report path')
    manifest = json.loads(args.manifest.read_text())
    for relative,expected in manifest.get('scene_sha256',{}).items():
        if hashlib.sha256((ROOT/relative).read_bytes()).hexdigest()!=expected:
            parser.error(f'Frozen scene changed: {relative}')
    cases = manifest['cases']
    if not cases or len({c['id'] for c in cases}) != len(cases): parser.error('Case IDs must be unique')
    torch.set_num_threads(1)
    model = PPO.load(args.policy, device='cpu')
    results = {}
    started = time.perf_counter()
    for offset in range(0,len(cases),args.workers):
        results.update(evaluate_batch(model,cases[offset:offset+args.workers],args.components.resolve(),args.max_steps,args.route_mode))
    categories = {}
    for category in sorted({c['category'] for c in cases}):
        subset = [r for r in results.values() if r['category'] == category]
        categories[category] = dict(tasks=len(subset), safe_success=sum(r['safe_success'] for r in subset),
                                    arrived=sum(r['arrived'] for r in subset), collisions=sum(r['collisions'] for r in subset),
                                    guards=sum(r['guards'] for r in subset), timeouts=sum(r['timeout'] for r in subset))
    report = dict(policy=str(args.policy.resolve()), policy_sha256=hashlib.sha256(args.policy.read_bytes()).hexdigest(),
                  components_sha256={name:hashlib.sha256((args.components/name).read_bytes()).hexdigest() for name in ['localizer.pt','router.pt']},
                  manifest_sha256=hashlib.sha256(args.manifest.read_bytes()).hexdigest(), max_steps=args.max_steps,
                  wall_seconds=time.perf_counter()-started, categories=categories, cases=list(results.values()),
                  completed=len(results)==len(cases), limitation=manifest['limitation'],route_mode=args.route_mode,
                  reference='Offline inflated-grid A* with line-of-sight smoothing; not a dynamic optimality bound',
                  action_inference='Deterministic actor evaluated separately for each robot; batch-independent shape',
                  timing='Environment step includes MuJoCo physics; not an embedded navigation inference benchmark')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print('SUMMARY', json.dumps(categories), flush=True)
    if not report['completed']: raise RuntimeError('Evaluation did not complete every fixed case')


if __name__ == '__main__':
    main()
