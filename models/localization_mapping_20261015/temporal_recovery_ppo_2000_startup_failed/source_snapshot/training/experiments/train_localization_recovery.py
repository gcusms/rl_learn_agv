"""Bounded PPO adaptation after a localization frontend change.

Only a learned recovery residual and critic are updated. Ordinary navigation's
base weights and the 303D directional approach/terminal actor stay frozen.
No steering teacher, replacement command, weaker guard or new reward is added.
Recorded training failures are rehearsals, never labelled independent tests.
"""
import argparse
import hashlib
import json
from pathlib import Path
import random
import shutil
import time
import numpy as np
import torch
from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from navrl.geometry import ROOT
from neural_navigation.experiment_budget import ExperimentBudget
from neural_navigation.goal_pose_runtime import check_files
from neural_navigation.masked_actions import ACTIONS,masks_from_observation
from experiments.recovery_sampling import ScheduledRecoveryCurriculum
from experiments.quality_masked_navigation import QualityMaskedWrapper
from experiments.soft_prior_recovery_policy import warm_start_soft_prior
from experiments.focused_recovery_policy import recovery_support
from experiments.masked_navigation import PortableMaskedActor
from experiments.build_preserved_navigation import PreservedNavigationActor
from experiments.experiment_provenance import record_provenance


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def training_case(case):
    if not case['id'].startswith('train_failure_'):
        raise ValueError('Only an explicitly recorded training failure can be rehearsed')
    return dict(scene=str(ROOT/case['scene']),seed=case['seed'],stage='robust',
        options=dict(start=case['start'],goal=case['goal'][:2]),
        interventions={key:value for key,value in case.items() if key not in
            ('id','category','scene','seed','start','goal','pilot_category','max_steps')},
        source_training_case_id=case['id'])


class Progress(BaseCallback):
    def __init__(self,output,budget,steps):
        super().__init__();self.output=output;self.budget=budget;self.steps=steps
        self.counts=dict(nonzero=0,changed_nonzero=0,wait_with_alternative=0,only_stop=0,
            trainable_recovery=0,recorded_frames=0,general_frames=0,episodes=0,success=0,collisions=0)

    def _on_step(self):
        self.budget.tick()
        for info in self.locals['infos']:
            c=self.counts;c['nonzero']+=int(info['masked_choice']!=0)
            c['changed_nonzero']+=int(info['nonzero_choice_changed_by_guard'])
            if c['changed_nonzero']:raise RuntimeError('PPO/action-mask execution mismatch')
            c['wait_with_alternative']+=int(info['masked_chose_wait_with_nonzero_available'])
            c['only_stop']+=int(info['masked_current_nonzero_choices']==0)
            c['recorded_frames' if info['focused_training_replay'] else 'general_frames']+=1
            if 'episode' in info:
                c['episodes']+=1;c['success']+=int(info['verified_mission_success']);c['collisions']+=int(info['collision_events'])
                with (self.output/'episodes.jsonl').open('a') as stream:stream.write(json.dumps(info['episode'])+'\n')
        observations=torch.as_tensor(self.locals['obs_tensor'],device='cpu')
        self.counts['trainable_recovery']+=int(recovery_support(observations).sum())
        if self.num_timesteps%250==0:
            values={key:float(value) for key,value in self.logger.name_to_value.items()
                if key.startswith('train/') and isinstance(value,(int,float,np.number))}
            row=dict(steps=self.num_timesteps,target_steps=self.steps,counts=self.counts,
                resource=self.budget.statistics(),ppo=values)
            (self.output/'progress.json').write_text(json.dumps(row,indent=2)+'\n')
            with (self.output/'updates.jsonl').open('a') as stream:stream.write(json.dumps(row)+'\n')
            print(json.dumps(row),flush=True)
        return True


def export_candidate(model,bundle,output):
    shutil.copytree(bundle,output)
    rng=np.random.default_rng(118615);obs=rng.uniform(-1,1,(512,303)).astype(np.float32)
    obs[:,235:244]=rng.integers(0,2,(512,9));obs[:,267:269]=rng.integers(0,2,(512,2))
    obs[:128,218:220]*=.005;obs[:128,302]=1.
    navigation=PortableMaskedActor(model.policy).eval();parent=torch.jit.load(str(bundle/'actor.pt')).eval()
    with torch.no_grad():
        expected=ACTIONS[model.predict(obs[:,:300],action_masks=masks_from_observation(obs[:,:282]),deterministic=True)[0]]
        np.testing.assert_array_equal(navigation(torch.from_numpy(obs[:,:300])).numpy(),expected)
        composite=PreservedNavigationActor(navigation,parent).eval()
        gate=(np.linalg.norm(obs[:,218:220]*12,axis=1)<=.3)&(obs[:,302]>.5)
        expected[gate]=parent(torch.from_numpy(obs[gate])).numpy()
        actor=torch.jit.trace(composite,torch.from_numpy(obs))
        np.testing.assert_array_equal(actor(torch.from_numpy(obs)).numpy(),expected)
    actor.save(str(output/'actor.pt'))
    preservation=dict(schema='frozen_base_localization_recovery_v1',approach_radius_m=.3,
        original_directional_actor_unchanged=True,base_navigation_weights_frozen=True,
        trainable_component='sensor-gated recovery residual',teacher_at_runtime=False,
        unchanged_actions=True,comparison_inputs=len(obs))
    (output/'navigation_preservation.json').write_text(json.dumps(preservation,indent=2)+'\n')
    manifest=json.loads((output/'manifest.json').read_text());manifest.update(deployment_approved=False,
        navigation_preservation_contract=preservation,parent_manifest_sha256=digest(bundle/'manifest.json'))
    for name in ('actor.pt','navigation_preservation.json'):manifest['files'][name]=digest(output/name)
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    check_files(output,manifest)


def main():
    p=argparse.ArgumentParser();p.add_argument('--bundle',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--cases',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--steps',type=int,default=2000)
    p.add_argument('--seed',type=int,default=118615);p.add_argument('--cpu-cores',type=float,default=.35);a=p.parse_args()
    if a.output.exists() or not 1000<=a.steps<=5000 or a.steps%250:p.error('Fresh output; 1000..5000 steps in multiples of250')
    a.bundle=a.bundle.resolve();a.checkpoint=a.checkpoint.resolve();a.output.mkdir(parents=True)
    torch.set_num_threads(1);budget=ExperimentBudget(a.cpu_cores,directory=a.output)
    manifest=json.loads((a.bundle/'manifest.json').read_text());check_files(a.bundle,manifest)
    cases=[training_case(case) for case in json.loads(a.cases.read_text()) if case['id'].startswith('train_failure_')]
    if not cases:raise ValueError('An explicitly marked training failure is required')
    parent=MaskablePPO.load(a.checkpoint,device='cpu');scale=float(parent.training_reward_scale)
    if parent.observation_space.shape!=(300,):raise ValueError('Archived native300D navigation PPO parent required')
    original_base={key:value.clone() for key,value in parent.policy.state_dict().items()};del parent
    config=dict(steps=a.steps,seed=a.seed,parent_policy_sha256=digest(a.checkpoint),bundle_manifest_sha256=digest(a.bundle/'manifest.json'),
        reward_unchanged=True,reward_scale=scale,guard_unchanged=True,localizer_router_frozen=True,
        terminal_actor_frozen=True,normal_navigation_base_frozen=True,recovery_prior_scale=1/6,
        sampling_and_update_share_masks_and_logits=True,recorded_training_cases=cases,
        additional_training_is_not_independent_evaluation=True,deployment_approved=False)
    (a.output/'config.json').write_text(json.dumps(config,indent=2)+'\n');record_provenance(a.output,a.bundle)
    base=ScheduledRecoveryCurriculum(a.bundle,stage='robust',route_mode='shortcut',motion_profiles=True,
        footprint_clearance_cost=1.,recorded_cases=cases,max_steps=700,replay_seed=a.seed)
    env=Monitor(QualityMaskedWrapper(base,reward_scale=scale))
    status=dict(status='running',started_unix=time.time(),additional_ppo_steps=0,deployment_approved=False)
    (a.output/'status.json').write_text(json.dumps(status,indent=2)+'\n')
    try:
        model=warm_start_soft_prior(a.checkpoint,env,a.seed);model.save(a.output/'policy_initial.zip')
        export_candidate(model,a.bundle,a.output/'initial_portable')
        callback=Progress(a.output,budget,a.steps);model.learn(a.steps,callback=callback,reset_num_timesteps=True)
        model.save(a.output/'policy.zip')
        for key,value in model.policy.mlp_extractor.policy_net.base.state_dict().items():
            torch.testing.assert_close(value,original_base['mlp_extractor.policy_net.'+key],rtol=0,atol=0)
        for key,value in model.policy.action_net.base.state_dict().items():
            torch.testing.assert_close(value,original_base['action_net.'+key],rtol=0,atol=0)
        export_candidate(model,a.bundle,a.output/'portable')
        torch.save(dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),
            curriculum_rng=base.np_random.bit_generator.state,replay_rng=base.replay_rng.bit_generator.state,
            recovery_reset_index=base.recovery_reset_index,exact_mid_episode_resume=False),a.output/'random_state.pt')
        status.update(status='complete',finished_unix=time.time(),additional_ppo_steps=model.num_timesteps,
            counts=callback.counts,resource=budget.statistics(),training_is_not_deployment_acceptance=True)
    except BaseException as error:
        status.update(status='failed',error=repr(error));raise
    finally:
        (a.output/'status.json').write_text(json.dumps(status,indent=2)+'\n');env.close()


if __name__=='__main__':main()
