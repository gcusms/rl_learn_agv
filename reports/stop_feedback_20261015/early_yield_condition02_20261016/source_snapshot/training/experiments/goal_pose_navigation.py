"""Opt-in 303-feature task wrapper, migration and portable runtime.

No heading steering controller is added: the categorical RL actor still selects
every movement. New goal features must be trained before deployment.
"""
import copy
import hashlib
import json
from pathlib import Path
import gymnasium as gym
import numpy as np
import torch
from sb3_contrib import MaskablePPO
from navrl.geometry import wrap
from experiments.masked_navigation import MaskedNavigationCurriculum,PortableMaskedActor
from experiments.quality_masked_navigation import QualityMaskedWrapper,QualityAnchoredPPO
from neural_navigation.goal_pose import contract,append_goal_features
from neural_navigation.masked_actions import ACTIONS,masks_from_observation,action_contract
from neural_navigation.quality_prediction_context import QualityPredictionContext,contract as quality_contract
from neural_navigation.runtime import SensorNavigationRuntime


class GoalPoseAnchoredPPO(QualityAnchoredPPO):
    observation_dim=303
    def eligible(self,observations,masks):
        clear=super().eligible(observations,masks)
        far=torch.linalg.vector_norm(observations[:,218:220]*12,dim=1)>.8
        # Retain clear navigation, while letting the actor learn near-goal turns.
        return clear&(far|(observations[:,302]<.5))


class GoalPoseWrapper(gym.Wrapper):
    def __init__(self,env,gamma=.99,alignment_probability=.35,alignment_max_steps=100,balanced_alignment=False):
        super().__init__(env)
        if env.observation_space.shape!=(300,) or not 0<gamma<=1 or not 0<=alignment_probability<=.5:
            raise ValueError('300-feature quality prefix, valid discount and at least half navigation tasks required')
        self.observation_space=gym.spaces.Box(-1,1,(303,),np.float32)
        self.gamma=float(gamma);self.alignment_probability=alignment_probability
        if balanced_alignment and alignment_probability!=.5:raise ValueError('Alternating curriculum requires half alignment tasks')
        self.balanced_alignment=bool(balanced_alignment);self.reset_count=0
        self.reward_scale=env.reward_scale
        if alignment_max_steps<50:raise ValueError('At least ten physical seconds for alignment required')
        self.navigation_max_steps=env.unwrapped.max_steps;self.alignment_max_steps=int(alignment_max_steps)

    def _potential(self):
        e=self.env.unwrapped.active;core=self.env.unwrapped.core
        if core.goal_yaw is None:return 0.
        distance=np.linalg.norm(e.truth[:2]-core.goal)
        return -4.*np.exp(-distance/.5)*abs(wrap(core.goal_yaw-e.truth[2]))

    def reset(self,seed=None,options=None):
        rng=np.random.default_rng(seed) if seed is not None else self.env.unwrapped.np_random
        options=dict(options or {})
        if 'pose_alignment_task' not in options:
            options['pose_alignment_task']=(self.reset_count%2==0 if self.balanced_alignment
                else bool(rng.random()<self.alignment_probability))
        self.reset_count+=1
        self.env.unwrapped.max_steps=self.alignment_max_steps if options['pose_alignment_task'] else self.navigation_max_steps
        obs,info=self.env.reset(seed=seed,options=options)
        base=self.env.unwrapped;core=base.core
        required=bool(options.get('goal_yaw_required',True))
        yaw=float(options.get('goal_yaw',base.np_random.uniform(-np.pi,np.pi)))
        core.set_goal([*core.goal,yaw] if required else core.goal.copy());self.previous_potential=self._potential()
        case=json.loads(base.training_case_json);case['goal_yaw']=core.goal_yaw
        case['pose_alignment_task']=options['pose_alignment_task'];base.training_case_json=json.dumps(case)
        return append_goal_features(obs,core),dict(info,goal_yaw=core.goal_yaw)

    def step(self,action):
        obs,reward,done,truncated,info=self.env.step(action)
        potential=0. if done else self._potential()
        heading=self.gamma*potential-self.previous_potential;self.previous_potential=potential
        # Reward shaping uses simulation labels only; actor sees estimated pose.
        # Do not weaken collision/false-arrival penalties with terminal shaping.
        if info['safety_termination'] or (done and not info['verified_mission_success']):heading=min(heading,0.)
        info=dict(info,goal_heading_potential_reward=float(heading),goal_yaw=self.env.unwrapped.core.goal_yaw)
        return append_goal_features(obs,self.env.unwrapped.core),reward+heading*self.reward_scale,done,truncated,info

    def action_masks(self):return self.env.action_masks()


def migrate_goal_policy(checkpoint,env,seed=101401):
    """Append zero-initialized columns, preserving action logits and Adam slots.

    This preserves the initial actor, NOT learned heading capability. It starts
    a new fine-tuning experiment, rather than claiming an exact RNG resume.
    """
    old=MaskablePPO.load(checkpoint,device='cpu')
    if old.observation_space.shape!=(300,) or env.observation_space.shape!=(303,) or old.action_space.n!=12:
        raise ValueError('300-to-303 masked policy migration required')
    if float(getattr(old,'training_reward_scale',1.))!=env.reward_scale:
        raise ValueError('Parent and fine-tuning reward units disagree')
    if old.gamma!=env.gamma:raise ValueError('Potential shaping discount must equal PPO discount')
    new=GoalPoseAnchoredPPO('MlpPolicy',env,device='cpu',seed=seed,n_steps=250,batch_size=125,
        n_epochs=4,learning_rate=3e-5,target_kl=.02,ent_coef=old.ent_coef,gamma=old.gamma,
        gae_lambda=old.gae_lambda,policy_kwargs=copy.deepcopy(old.policy_kwargs),reference_strength=2.,reference_kl_budget=.02)
    source=old.policy.state_dict();target=new.policy.state_dict()
    inputs={'mlp_extractor.policy_net.0.weight','mlp_extractor.value_net.0.weight'}
    for name,value in target.items():
        if name in inputs:value.zero_();value[:,:300]=source[name]
        else:
            if value.shape!=source[name].shape:raise ValueError('Unexpected policy architecture')
            value.copy_(source[name])
    new.policy.load_state_dict(target);previous=dict(old.policy.named_parameters())
    for name,parameter in new.policy.named_parameters():
        state=old.policy.optimizer.state.get(previous[name])
        if not state:continue
        migrated={}
        for key,value in state.items():
            if torch.is_tensor(value):
                if name in inputs and value.shape==previous[name].shape:
                    expanded=torch.zeros_like(parameter);expanded[:,:300]=value;migrated[key]=expanded
                else:migrated[key]=value.clone()
            else:migrated[key]=copy.deepcopy(value)
        new.policy.optimizer.state[parameter]=migrated
    new.training_reward_scale=env.reward_scale;new.parent_training_timesteps=old.num_timesteps
    new.goal_pose_contract=contract();new.snapshot_reference();return new


def export_goal_policy(model,components,output):
    import shutil
    output=Path(output);components=Path(components);output.mkdir(exist_ok=False)
    if model.observation_space.shape!=(303,):raise ValueError('Goal-pose policy required')
    rng=np.random.default_rng(19);inputs=rng.uniform(-1,1,(256,303)).astype(np.float32)
    inputs[:,235:244]=rng.integers(0,2,(256,9));inputs[:,267:269]=rng.integers(0,2,(256,2))
    actor=torch.jit.trace(PortableMaskedActor(model.policy).eval(),torch.from_numpy(inputs))
    expected=ACTIONS[model.predict(inputs,action_masks=masks_from_observation(inputs[:,:282]),deterministic=True)[0]]
    with torch.no_grad():np.testing.assert_array_equal(actor(torch.from_numpy(inputs)).numpy(),expected)
    actor.save(str(output/'actor.pt'))
    for name in ('hardware.json','localizer.pt','router.pt','localization_contract.json','localization_kernel.so','routing_contract.json','uncertainty_contract.json'):
        if (components/name).exists():shutil.copy2(components/name,output/name)
    (output/'quality_observation.json').write_text(json.dumps(quality_contract(),indent=2))
    (output/'goal_pose_observation.json').write_text(json.dumps(contract(),indent=2))
    hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in output.iterdir()}
    manifest=dict(sensor_core='odom-goal-pose',required_runtime='experiments.goal_pose_navigation.GoalPoseRuntime',
        observation_contract=contract(),quality_prefix_contract=quality_contract(),action_contract=json.loads((components/'hardware.json').read_text())['policy_action'],
        requires_quality_feature_generator=True,exact_action_parity=True,comparison_inputs=256,
        deployment_approved=False,files=hashes)
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2))


from neural_navigation.goal_pose_runtime import GoalPoseInput,GoalPoseRuntime
