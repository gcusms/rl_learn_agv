"""Bounded full-actor PPO pilot using shared guard explanations and forecasts."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import shutil
import numpy as np
import torch
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from neural_navigation.decision_feedback import contract, TRAIN_DIM, RUNTIME_DIM, LEGACY_SCHEMA, WAIT_SCHEMA
from neural_navigation.decision_feedback_runtime import RUNTIME_NAME
from neural_navigation.goal_pose_runtime import check_files
from neural_navigation.experiment_budget import ExperimentBudget
from neural_navigation.masked_actions import ACTIONS, masks_from_observation
from experiments.masked_navigation import PortableMaskedActor
from experiments.recovery_sampling import ScheduledRecoveryCurriculum
from experiments.stop_feedback_navigation import StopFeedbackWrapper
from experiments.stop_feedback_ppo import warm_start
from experiments.train_localization_recovery import training_case
from experiments.experiment_provenance import record_provenance


def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class DirectionalFeedbackActor(torch.nn.Module):
    def __init__(self,navigation,directional):
        super().__init__();self.navigation=navigation;self.directional=directional
    def forward(self,x):
        nav=torch.cat([x[:,:300],x[:,303:]],dim=1)
        near=(torch.linalg.vector_norm(x[:,218:220]*12,dim=1)<=.3)&(x[:,302]>.5)
        return torch.where(near[:,None],self.directional(x[:,:303]),self.navigation(nav))


def export_candidate(model,bundle,output):
    bundle=Path(bundle);output=Path(output)
    feedback_schema=getattr(model,'feedback_schema',LEGACY_SCHEMA)
    feedback_contract=contract(feedback_schema)
    if model.feedback_migration.get('feedback_schema',LEGACY_SCHEMA)!=feedback_schema:
        raise ValueError('Export schema differs from the model training contract')
    shutil.copytree(bundle,output,ignore=shutil.ignore_patterns('MODEL_CARD.json'))
    parent=json.loads((bundle/'manifest.json').read_text())
    check_files(bundle,parent)
    navigation=PortableMaskedActor(model.policy).eval()
    directional=torch.jit.load(str(bundle/'actor.pt'),map_location='cpu').eval()
    rng=np.random.default_rng(221015);obs=rng.uniform(-1,1,(512,RUNTIME_DIM)).astype(np.float32)
    obs[:,235:244]=rng.integers(0,2,(512,9));obs[:,267:269]=rng.integers(0,2,(512,2))
    obs[:128,218:220]*=.005;obs[:128,302]=1.
    training=np.concatenate([obs[:,:300],obs[:,303:]],axis=1)
    mask=masks_from_observation(training[:,:282])
    with torch.no_grad():
        expected=ACTIONS[model.predict(training,action_masks=mask,deterministic=True)[0]]
        composite=DirectionalFeedbackActor(navigation,directional).eval()
        gate=(np.linalg.norm(obs[:,218:220]*12,axis=1)<=.3)&(obs[:,302]>.5)
        expected[gate]=directional(torch.from_numpy(obs[gate,:303])).numpy()
        actor=torch.jit.trace(composite,torch.from_numpy(obs))
        np.testing.assert_array_equal(actor(torch.from_numpy(obs)).numpy(),expected)
    actor.save(str(output/'actor.pt'))
    preservation=dict(schema='full_navigation_feedback_ppo_v1',approach_radius_m=.3,
        directional_terminal_actor_preserved=True,navigation_actor_all_parameters_trainable=True,
        legacy_300D_prefix_unchanged=True,training_dimensions=TRAIN_DIM,runtime_dimensions=RUNTIME_DIM,
        unchanged_actions_masks_guard=True,reward_unchanged=True,comparison_inputs=len(obs),
        no_runtime_teacher=True,stop_remains_selectable=True)
    preservation['training_exploration_contract']=getattr(model,'exploration_contract',None)
    (output/'navigation_preservation.json').write_text(json.dumps(preservation,indent=2)+'\n')
    (output/'decision_feedback.json').write_text(json.dumps(feedback_contract,indent=2)+'\n')
    manifest=dict(parent,required_runtime=RUNTIME_NAME,observation_contract=feedback_contract,
        navigation_preservation_contract=preservation,deployment_approved=False,
        parent_manifest_sha256=digest(bundle/'manifest.json'),additional_ppo_steps=model.num_timesteps)
    manifest['files']=dict(parent['files'])
    for name in ('actor.pt','navigation_preservation.json','decision_feedback.json'):
        manifest['files'][name]=digest(output/name)
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    check_files(output,manifest)
    (output/'MODEL_CARD.json').write_text(json.dumps(dict(deployment_approved=False,
        additional_ppo_steps=model.num_timesteps,method='full masked PPO with causal guard feedback',
        frozen_components=['localizer','router','directional_terminal_actor'],
        migration=model.feedback_migration,manifest_sha256=digest(output/'manifest.json'),
        limitations=['Guard admission is not certified future safety.',
                     'An initialization/parity or training check is not navigation acceptance.']),indent=2)+'\n')


class Progress(BaseCallback):
    def __init__(self,output,budget,record_rollouts=False):
        super().__init__();self.output=output;self.budget=budget
        self.record_rollouts=record_rollouts;self.trace_steps=[]
        self.counts=dict(decision_frames=0,only_stop=0,voluntary_wait=0,nonzero=0,
            changed_nonzero=0,episodes=0,success=0,collisions=0)
        self.reasons={}
    def _on_rollout_start(self):
        self.trace_steps=[]
    def _on_step(self):
        self.budget.tick()
        for env_index,info in enumerate(self.locals['infos']):
            c=self.counts;c['decision_frames']+=int(info['decision_feedback_trainable_choice'])
            c['only_stop']+=int(info['masked_current_nonzero_choices']==0)
            c['voluntary_wait']+=int(info['masked_chose_wait_with_nonzero_available'])
            c['nonzero']+=int(info['masked_choice']!=0)
            c['changed_nonzero']+=int(info['nonzero_choice_changed_by_guard'])
            if c['changed_nonzero']:raise RuntimeError('PPO chosen nonzero action differs from execution')
            reason=info['decision_feedback_previous_reason'];self.reasons[reason]=self.reasons.get(reason,0)+1
            if self.record_rollouts:
                self.trace_steps.append(dict(sensor_timestamp=info['decision_feedback_sensor_timestamp'],
                    sampled_choice=info['masked_choice'],executed_command=np.asarray(info['commanded_velocity']).tolist(),
                    executed_reason=reason,nonzero_legal_choices=info['masked_current_nonzero_choices'],
                    native_reward=info['native_reward'],true_geodesic_progress_m=info['fixed_geodesic_progress_m'],
                    case_json=info['training_case_json'],done=bool(self.locals['dones'][env_index])))
            if 'episode' in info:
                c['episodes']+=1;c['success']+=int(info['verified_mission_success'])
                c['collisions']+=int(info['collision_events'])
                with (self.output/'episodes.jsonl').open('a') as stream:
                    stream.write(json.dumps(dict(episode=info['episode'],success=info['verified_mission_success'],
                        collisions=info['collision_events'],case=json.loads(info['training_case_json'])))+'\n')
        if self.num_timesteps%250==0:
            logs={k:float(v) for k,v in self.logger.name_to_value.items() if k.startswith('train/') and isinstance(v,(int,float,np.number))}
            result=dict(steps=self.num_timesteps,counts=self.counts,reasons=self.reasons,
                        resource=self.budget.statistics(),ppo=logs)
            (self.output/'progress.json').write_text(json.dumps(result,indent=2)+'\n')
            print(json.dumps(result),flush=True)
            if self.num_timesteps%1000==0:self.model.save(self.output/f'policy_step_{self.num_timesteps}.zip')
        return True

    def _on_rollout_end(self):
        if not self.record_rollouts:return
        buffer=self.model.rollout_buffer
        from experiments.feedback_rollout_trace import save_rollout
        folder=self.output/'rollouts';folder.mkdir(exist_ok=True)
        save_rollout(folder/f'rollout_{self.num_timesteps:06d}.npz',buffer,self.trace_steps,
            collected_steps=self.num_timesteps,optimizer_epochs_before_update=self.model._n_updates,
            feedback_schema=self.model.feedback_schema)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('bundle','checkpoint','training-cases','output'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--steps',type=int,default=4000);parser.add_argument('--seed',type=int,default=221015)
    parser.add_argument('--cpu-cores',type=float,default=.35)
    parser.add_argument('--feedback-schema',choices=(LEGACY_SCHEMA,WAIT_SCHEMA),default=LEGACY_SCHEMA)
    parser.add_argument('--record-rollouts',action='store_true',help='Archive causal PPO states/masks and actual execution before each update')
    parser.add_argument('--exploration',choices=('none','sensor_temperature_v1'),default='none')
    args=parser.parse_args()
    if args.output.exists() or not 1000<=args.steps<=10000 or args.steps%250:
        parser.error('Fresh output and 1000..10000 steps in multiples of 250 required')
    args.bundle=args.bundle.resolve();args.output.mkdir(parents=True)
    torch.set_num_threads(1);budget=ExperimentBudget(args.cpu_cores,memory_mib=3072,directory=args.output)
    cases=[training_case(c) for c in json.loads(args.training_cases.read_text())]
    env=StopFeedbackWrapper(ScheduledRecoveryCurriculum(args.bundle,stage='robust',route_mode='shortcut',
        motion_profiles=True,recorded_cases=cases,recorded_episodes_per_cycle=1),reward_scale=1/60,
        feedback_schema=args.feedback_schema)
    model=warm_start(args.checkpoint,env,seed=args.seed,exploration_schema=args.exploration)
    progress=Progress(args.output,budget,args.record_rollouts)
    config=dict(steps=args.steps,seed=args.seed,contract=contract(args.feedback_schema),migration=model.feedback_migration,
        training_cases=cases,training_cases_sha256=digest(args.training_cases),
        parent_policy_sha256=digest(args.checkpoint),bundle_sha256=digest(args.bundle/'manifest.json'),
        cpu_cores=args.cpu_cores,reward_changed=False,record_rollouts=args.record_rollouts,
        exploration_contract=model.exploration_contract,deployment_approved=False)
    (args.output/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    kernel=Path(__file__).parents[1]/'neural_navigation/static_tracking_kernel.cpp'
    record_provenance(args.output,args.bundle,[kernel])
    def status(**values):
        (args.output/'status.json').write_text(json.dumps(values,indent=2)+'\n')
    status(status='training',target_steps=args.steps)
    try:
        model.save(args.output/'policy_initial.zip')
        export_candidate(model,args.bundle,args.output/'initial_portable')
        model.set_env(Monitor(env))
        model.learn(total_timesteps=args.steps,callback=progress,reset_num_timesteps=True)
        model.save(args.output/'policy.zip')
        export_candidate(model,args.bundle,args.output/'portable')
        torch.save(dict(torch_rng=torch.get_rng_state(),numpy_rng=np.random.get_state(),python_rng=random.getstate(),
            curriculum_rng=env.unwrapped.np_random.bit_generator.state),args.output/'random_state.pt')
        status(status='complete',additional_ppo_steps=model.num_timesteps,counts=progress.counts,
               reasons=progress.reasons,resource=budget.statistics(),feedback_schema=args.feedback_schema,
               exploration_schema=model.exploration_schema,
               recorded_rollouts=len(list((args.output/'rollouts').glob('*.npz'))),deployment_approved=False)
    except BaseException as error:
        status(status='failed',additional_ppo_steps=model.num_timesteps,error=repr(error),counts=progress.counts)
        raise
    finally:env.close()


if __name__=='__main__':main()
