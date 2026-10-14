"""Sensor-conditioned terminal policy: demonstration warmup then masked PPO.

The original actor remains frozen. A learned terminal actor selects the same
12 velocity choices inside the versioned terminal radius (new candidate: 0.10 m).
Sampling, PPO updates,
and export all use the same gate and masks; no teacher runs at deployment.
"""
import copy
import hashlib
import json
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from navrl.geometry import wrap
from neural_navigation.goal_pose import append_goal_features, contract as goal_contract
from neural_navigation.masked_actions import ACTIONS, COMMANDS, masks_from_observation
from neural_navigation.quality_prediction_context import contract as quality_contract
from experiments.goal_pose_navigation import GoalPoseRuntime


def docking_contract(radius=.8):
    if radius not in (.8,.10):raise ValueError('Unsupported terminal radius')
    return dict(schema='learned_terminal_actor_v1' if radius==.8 else 'learned_terminal_actor_v2',observation_dimensions=303,
        activation_distance_m=radius,requires_directional_goal=True,actions=12,
        frozen_navigation_actor=True,teacher_at_runtime=False,
        action_space_expanded=False,slow_rotation_calibrated=False)


def terminal_gate(x:torch.Tensor,radius:float=.8):
    return (torch.linalg.vector_norm(x[:,218:220]*12,dim=1)<=radius)&(x[:,302]>.5)


def terminal_features(x):
    # Physical goal coordinates need useful scale; the original /12 encoding
    # otherwise compresses all terminal errors into a tiny input range.
    return torch.cat([torch.clamp(x[:,218:220]*15,-1,1),x[:,220:224],
        x[:,300:302],x[:,224:227],x[:,282:300],x[:,144:216],
        x[:,235:244],x[:,267:269]],dim=1)


def docking_progress_reward(previous_distance,distance,previous_yaw,yaw):
    heading_gate=float(max(distance,previous_distance)<.15)
    return 12*(previous_distance-distance)+8*heading_gate*(previous_yaw-yaw)


class DockingPolicy(MaskableActorCriticPolicy):
    def __init__(self,*args,terminal_radius=.8,**kwargs):
        super().__init__(*args,**kwargs)
        self.terminal_radius=float(terminal_radius);docking_contract(self.terminal_radius)
        self.terminal_actor=torch.nn.Sequential(torch.nn.Linear(112,64),torch.nn.Tanh(),
            torch.nn.Linear(64,64),torch.nn.Tanh(),torch.nn.Linear(64,12))
        self.mlp_extractor.policy_net.requires_grad_(False)
        self.action_net.requires_grad_(False)
        lr_schedule=kwargs.get('lr_schedule',args[2] if len(args)>2 else None)
        self.optimizer=self.optimizer_class([p for p in self.parameters() if p.requires_grad],
            lr=lr_schedule(1),**self.optimizer_kwargs)

    def logits(self,obs):
        features=self.extract_features(obs)
        if not self.share_features_extractor:features=features[0]
        base=self.action_net(self.mlp_extractor.forward_actor(features))
        return torch.where(terminal_gate(obs,self.terminal_radius)[:,None],self.terminal_actor(terminal_features(obs)),base)

    def get_distribution(self,obs,action_masks=None):
        distribution=self.action_dist.proba_distribution(action_logits=self.logits(obs))
        if action_masks is not None:distribution.apply_masking(action_masks)
        return distribution

    def forward(self,obs,deterministic=False,action_masks=None):
        distribution=self.get_distribution(obs,action_masks)
        actions=distribution.get_actions(deterministic=deterministic)
        return actions,self.predict_values(obs),distribution.log_prob(actions)

    def evaluate_actions(self,obs,actions,action_masks=None):
        distribution=self.get_distribution(obs,action_masks)
        return self.predict_values(obs),distribution.log_prob(actions),distribution.entropy()


class PortableDockingActor(torch.nn.Module):
    def __init__(self,policy):
        super().__init__()
        self.net=copy.deepcopy(policy.mlp_extractor.policy_net)
        self.head=copy.deepcopy(policy.action_net)
        self.terminal=copy.deepcopy(policy.terminal_actor)
        self.terminal_radius=policy.terminal_radius
        self.register_buffer('actions',torch.from_numpy(ACTIONS.copy()))

    def forward(self,x):
        masks=torch.cat([torch.ones_like(x[:,:1],dtype=torch.bool),x[:,235:244]>.5,x[:,267:269]>.5],dim=1)
        logits=torch.where(terminal_gate(x,self.terminal_radius)[:,None],self.terminal(terminal_features(x)),self.head(self.net(x)))
        choice=torch.where(masks,logits,torch.full_like(logits,-1e8)).argmax(dim=1)
        return self.actions[choice]


def export_docking(model,components,output):
    import shutil
    output=Path(output);output.mkdir(exist_ok=False)
    rng=np.random.default_rng(114903)
    obs=rng.uniform(-1,1,(512,303)).astype(np.float32)
    obs[:256,218:220]=rng.uniform(-.035,.035,(256,2));obs[:256,302]=1.
    actor=torch.jit.trace(PortableDockingActor(model.policy).eval(),torch.from_numpy(obs))
    with torch.no_grad():
        expected=ACTIONS[model.predict(obs,action_masks=masks_from_observation(obs[:,:282]),deterministic=True)[0]]
        np.testing.assert_array_equal(actor(torch.from_numpy(obs)).numpy(),expected)
    actor.save(str(output/'actor.pt'))
    for name in ('hardware.json','router.pt','localizer.pt','localization_contract.json'):
        if (Path(components)/name).exists():shutil.copy2(Path(components)/name,output/name)
    for name,data in [('goal_pose_observation.json',goal_contract()),('quality_observation.json',quality_contract()),
                      ('docking_contract.json',docking_contract(model.policy.terminal_radius))]:
        (output/name).write_text(json.dumps(data,indent=2))
    from neural_navigation.masked_actions import action_contract
    manifest=dict(sensor_core='learned-terminal-goal-pose',
        required_runtime='experiments.docking_navigation.DockingRuntime',
        observation_contract=goal_contract(),quality_prefix_contract=quality_contract(),
        action_contract=action_contract(),docking_contract=docking_contract(model.policy.terminal_radius),
        requires_quality_feature_generator=True,exact_action_parity=True,
        comparison_inputs=len(obs),deployment_approved=False,
        files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in output.iterdir()})
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2))


class DockingRuntime(GoalPoseRuntime):
    def __init__(self,bundle,*args,**kwargs):
        bundle=Path(bundle);manifest=json.loads((bundle/'manifest.json').read_text())
        actual=manifest['docking_contract'];expected=docking_contract(actual.get('activation_distance_m'))
        if (manifest['required_runtime']!='experiments.docking_navigation.DockingRuntime'
            or actual!=expected or json.loads((bundle/'docking_contract.json').read_text())!=expected):
            raise ValueError('Docking contract mismatch')
        # Reuse the checked sensor path without weakening its bundle validation.
        self._initialize_goal_runtime(bundle,manifest,*args,**kwargs)

    def _initialize_goal_runtime(self,bundle,manifest,*args,**kwargs):
        from neural_navigation.runtime import SensorNavigationRuntime
        from neural_navigation.quality_prediction_context import QualityPredictionContext
        from experiments.goal_pose_navigation import GoalPoseInput
        if (manifest['observation_contract']!=goal_contract() or manifest['quality_prefix_contract']!=quality_contract()
            or json.loads((bundle/'goal_pose_observation.json').read_text())!=goal_contract()
            or json.loads((bundle/'quality_observation.json').read_text())!=quality_contract()):
            raise ValueError('Goal/quality contract mismatch')
        for name,digest in manifest['files'].items():
            if Path(name).name!=name or hashlib.sha256((bundle/name).read_bytes()).hexdigest()!=digest:
                raise ValueError('Bundle integrity failure')
        SensorNavigationRuntime.__init__(self,bundle,*args,**kwargs)
        if self.policy.observation_dimensions!=282:raise ValueError('Native V4 prefix required')
        self.policy.observation_dimensions=303
        self.policy=GoalPoseInput(self.policy,self.core,QualityPredictionContext(self.guard.half*2))


def demonstration_choice(obs,mask):
    """Training-only stop/rotate/approach reference using sensor observation.

    It is restricted to the existing legal choices, has no obstacle truth, and
    is checked through the same physics and guard as the learned actor.
    """
    dx,dy=np.asarray(obs[218:220])*12
    distance=float(np.hypot(dx,dy));bearing=float(np.arctan2(dy,dx))
    yaw=float(np.arctan2(obs[300],obs[301]));wz=float(obs[221]*.5)
    if distance<.10:
        # Start braking before alignment, accounting for observed turn rate.
        stop_angle=.10+abs(wz)*.20
        choice=0 if abs(yaw)<=stop_angle else (4 if yaw>0 else 5)
    elif abs(bearing)>.28:
        choice=4 if bearing>0 else 5
    else:
        speed=.05 if distance<.30 else (.1 if distance<.6 else .2)
        choice={.05:3,.1:2,.2:1}[speed]
    return choice if mask[choice] else 0


class DockingCurriculum(gym.Wrapper):
    """Short physical tasks, with a fixed time scale and truth-only rewards."""
    def __init__(self,env,gamma=.995):
        super().__init__(env)
        self.observation_space=gym.spaces.Box(-1,1,(303,),np.float32)
        self.gamma=gamma;self.reward_scale=env.reward_scale
        self.total_steps=0;self.curriculum_level=0;self.physical_since=None
        self.resets=0

    def _potential(self):
        base=self.env.unwrapped;distance=float(np.linalg.norm(base.active.truth[:2]-base.core.goal))
        yaw=float(abs(wrap(base.core.goal_yaw-base.active.truth[2])))
        return -12*distance-8*np.exp(-distance/.5)*yaw

    def reset(self,seed=None,options=None):
        # Short stop episodes otherwise recompile a large MuJoCo world every
        # second. Reuse physics for 16 resets, then sample another training map.
        base=self.env.unwrapped
        if base.active is None or self.resets%16==0:
            _,info=self.env.reset(seed=seed,options=dict(pose_alignment_task=True))
        else:
            gym.Env.reset(base,seed=seed);rng=base.np_random;m=base.maps[base.scene]
            cells=np.argwhere(m.clearance>.65)
            xy=m.xy(cells[int(rng.integers(len(cells)))])
            start=[*xy,float(rng.uniform(-np.pi,np.pi))]
            episode_seed=int(rng.integers(1,2**31-1))
            _,info=base.active.reset(seed=episode_seed,options=dict(start=start,goal=xy.tolist()))
            case=json.loads(base.training_case_json);case.update(seed=episode_seed,options=dict(start=start,goal=xy.tolist()))
            base.training_case_json=json.dumps(case)
            base.rejection_cost.reset();base.guard_overrides=0; base.guard_steps=0
            base.previous_executed=np.zeros(2)
        self.resets+=1
        base=self.env.unwrapped;core=base.core;e=base.active;rng=base.np_random
        options=options or {};level=int(options.get('curriculum_level',self.curriculum_level))
        # Balanced by episode duration (short caps), not long general navigation.
        approach=level>=2 and rng.random()<.45
        initial=np.array(json.loads(base.training_case_json)['options']['start'])
        goal=initial[:2].copy()
        if approach:
            for _ in range(30):
                distance=float(rng.uniform(.18,.65));bearing=float(rng.uniform(-.8,.8))
                proposal=initial[:2]+distance*np.array([np.cos(initial[2]+bearing),np.sin(initial[2]+bearing)])
                if base.maps[base.scene].lookup(proposal,base.maps[base.scene].clearance,outside=0)>.55 and base.maps[base.scene].visible(initial[:2],proposal):
                    goal=proposal;break
            else:approach=False
        ranges=[.25,np.pi/2,np.pi]
        offset=0. if rng.random()<.30 else float(rng.uniform(-ranges[min(level,2)],ranges[min(level,2)]))
        e.goal=goal.copy();core.reset(initial,[*goal,float(wrap(initial[2]+offset))])
        base.field=base.graphs[base.scene].field(goal)
        base.previous_distance=base.field.value(e.truth[:2]);base.max_steps=75
        from neural_navigation.runtime import EmergencyStop
        from neural_navigation.safety_context import SafetyContext
        base.guard=EmergencyStop(e.scene['robot_collision_box_m'][:2],memory_config=base.hardware.get('near_field_memory'))
        base.context=SafetyContext(base.hardware['policy_observation']['schema'])
        prefix,_=core.update(e.scan,e.observed_odometry,e.measured_velocity,e.elapsed)
        prefix=base.context.augment(prefix,core,base.guard,e.elapsed)
        base.previous_observation=prefix.copy()
        self.env.prediction.reset();obs=self.env.prediction.augment(prefix,core)
        case=json.loads(base.training_case_json);case['options']['goal']=goal.tolist()
        case.update(goal_yaw=core.goal_yaw,pose_alignment_task=not approach,
            docking_curriculum_level=level,docking_task='approach' if approach else 'alignment')
        base.training_case_json=json.dumps(case)
        self.previous_potential=self._potential();self.physical_since=None
        self.previous_physical_distance=float(np.linalg.norm(e.truth[:2]-core.goal))
        self.previous_physical_yaw=float(abs(wrap(core.goal_yaw-e.truth[2])))
        return append_goal_features(obs,core),dict(info,docking_task=case['docking_task'])

    def step(self,action):
        obs,reward,done,truncated,info=self.env.step(action)
        base=self.env.unwrapped;e=base.active;c=goal_contract()
        distance=float(np.linalg.norm(e.truth[:2]-base.core.goal))
        yaw=float(abs(wrap(base.core.goal_yaw-e.truth[2])))
        vx,wz=np.asarray(info['actual_velocity'])
        good=(distance<c['position_tolerance_m'] and yaw<c['yaw_tolerance_rad']
            and abs(vx)<c['stopped_linear_m_s'] and abs(wz)<c['stopped_angular_rad_s'])
        if not good:self.physical_since=None
        elif self.physical_since is None:self.physical_since=e.elapsed
        physically_settled=good and e.elapsed-self.physical_since>=c['settle_seconds']-1e-9
        if done and info['verified_mission_success'] and not physically_settled:
            reward-=50*self.reward_scale;done=False
            info=dict(info,verified_mission_success=False,mission_targets_reached=0)
        # Separate physical progress from near-goal angle progress. Multiplying
        # absolute angle error by exp(-distance/.5) rewarded moving away; a
        # gamma-scaled negative potential also gave stationary positive terms.
        # Here no movement gives zero progress, and increasing distance loses
        # reward even when the target orientation has a large error.
        progress=docking_progress_reward(self.previous_physical_distance,distance,self.previous_physical_yaw,yaw)
        self.previous_physical_distance=distance;self.previous_physical_yaw=yaw
        if info['safety_termination'] or (done and not info['verified_mission_success']):progress=min(progress,0.)
        aligned=distance<.15 and yaw<c['yaw_tolerance_rad']
        braking_cost=(2*(abs(vx)/.2)**2+2*(abs(wz)/.5)**2)*e.dt if aligned else 0.
        # Refund the fixed time cost only while physically stopped/aligned;
        # the small hold reward ends at the verified dwell terminal.
        holding_reward=.05+.1*e.dt if good else 0.
        reward+=(progress-braking_cost+holding_reward)*self.reward_scale
        info=dict(info,docking_progress_reward=progress,docking_braking_cost=braking_cost,
            docking_holding_reward=holding_reward,physical_goal_settled=bool(physically_settled),
            docking_task=json.loads(base.training_case_json)['docking_task'])
        self.total_steps+=1
        return append_goal_features(obs,base.core),reward,done,truncated,info

    def action_masks(self):return self.env.action_masks()
