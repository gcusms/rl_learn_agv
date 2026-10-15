"""Same fine-action sensor interface for training and native-physics evaluation."""
import json
from pathlib import Path
import gymnasium as gym
import numpy as np
from neural_navigation.stress_env import StressNavigationEnv
from neural_navigation.quality_prediction_context import QualityPredictionContext
from neural_navigation.runtime import execute_policy_action
from neural_navigation.safety_context import SafetyContext
from prototypes.narrow_alignment import AlignmentNarrowRuntime
from prototypes.narrow_observation import augment,masks,ACTIONS,COMMANDS

class NarrowTrainingEnv(gym.Env):
    observation_space=gym.spaces.Box(-1,1,(318,),np.float32)
    action_space=gym.spaces.Discrete(20)
    reward_scale=1/60
    def __init__(self,scene,bundle,seed=73120,max_steps=180):
        self.scene=Path(scene);self.bundle=Path(bundle);self.rng=np.random.default_rng(seed)
        self.max_steps=max_steps;self.env=None;self.runtime=None;self.episodes=0
    def reset(self,*,seed=None,options=None):
        super().reset(seed=seed)
        if seed is not None:self.rng=np.random.default_rng(seed)
        options=options or {};start=options.get('start',[float(self.rng.uniform(1.35,1.65)),float(self.rng.uniform(1.98,2.02)),float(self.rng.uniform(-.08,.08))])
        goal=options.get('goal',[5.,2.]);episode_seed=int(options.get('case_seed',self.rng.integers(1,2**30)))
        case=dict(id='narrow_training',category='narrow_training',seed=episode_seed,start=start,goal=goal)
        if self.env is None:
            self.env=StressNavigationEnv(case=case,components=self.bundle,scene_dir=self.scene,route_mode='shortcut',max_steps=self.max_steps)
            self.runtime=AlignmentNarrowRuntime(self.bundle,self.scene/'decision_map.yaml',start,goal,'shortcut')
        self.env.case=case;self.env.reset(seed=episode_seed,options=dict(start=start,goal=goal))
        self.runtime.core.reset(start,goal);self.runtime.context=SafetyContext(self.runtime.context.schema)
        self.runtime.guard.reverse_started=None
        if self.runtime.guard.memory:self.runtime.guard.memory.reset()
        self.quality=QualityPredictionContext(self.runtime.guard.half*2)
        self.count=0;self.last_command=np.zeros(2);self.episodes+=1
        obs=self.observe();self.old_distance=np.linalg.norm(self.runtime.core.estimate[:2]-goal)
        return obs,dict(training_seed=episode_seed,start=start,goal=goal)
    def observe(self):
        core=self.runtime.core;env=self.env
        base,_=core.update(env.scan,env.observed_odometry,env.measured_velocity,env.elapsed)
        prefix=self.runtime.context.augment(base,core,self.runtime.guard,env.elapsed)
        self.observation=augment(self.quality.augment(prefix,core),core,self.runtime.guard,env.elapsed)
        # Recognition confidence is an observation, not a second safety mask.
        # augment() already probes every fine action against the live guard.
        return self.observation.copy()
    def action_masks(self):return masks(self.observation).copy()
    def step(self,action):
        index=int(action)
        if not self.action_space.contains(action) or not self.action_masks()[index]:raise ValueError('Masked fine action selected')
        result=execute_policy_action(self.runtime.core,self.runtime.guard,ACTIONS[index],self.env.elapsed)
        self.runtime.context.record(result);command=np.asarray(result['command'])
        changed=not np.allclose(command,COMMANDS[index],atol=2e-8,rtol=0) and result['stop_reason']!='estimated_goal_reached'
        if changed:raise RuntimeError('Sampled/executed action disagreement; stop training')
        self.count+=1
        if result['stop_reason']!='estimated_goal_reached':self.env.step(np.array([command[0]/.2,command[1]/.5]))
        obs=self.observe();distance=float(np.linalg.norm(self.runtime.core.estimate[:2]-self.env.goal))
        collision=bool(self.env.collision_events or self.env.hard_guard_activations)
        estimated=distance<.12;verified=bool(estimated and np.linalg.norm(self.env.truth[:2]-self.env.goal)<=.2 and not collision)
        done=bool(collision or estimated);truncated=bool(self.count>=self.max_steps and not done)
        # Straight training corridor: distance progress is appropriate here,
        # not a substitute for geodesic reward in a future turning curriculum.
        reward=15*(self.old_distance-distance)-.02-.005*float(np.sum((command-self.last_command)**2))
        if index==0 and self.action_masks()[1:].any():reward-=.01
        if collision:reward-=60
        if verified:reward+=60
        if estimated and not verified:reward-=60
        if truncated:reward-=2
        self.old_distance=distance;self.last_command=command.copy()
        info=dict(verified_mission_success=verified,collision_events=self.env.collision_events,
            physical_guards=self.env.hard_guard_activations,nonzero_choice_changed_by_guard=changed,
            choice=index,fine_choice=index>=12,commanded_velocity=command.tolist(),
            travel_m=float(self.env.travel),simulated_seconds=float(self.env.elapsed),native_reward=reward)
        return obs,float(reward*self.reward_scale),done,truncated,info
    def close(self):
        if self.env:self.env.close()
