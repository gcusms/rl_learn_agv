"""Bounded demonstration-assisted terminal PPO experiment, separate outputs."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time
import numpy as np
import torch
from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import BaseCallback
from experiments.docking_navigation import (DockingPolicy,DockingCurriculum,PrecisionDockingCurriculum,
    demonstration_choice,export_docking,docking_contract)
from experiments.masked_navigation import MaskedNavigationCurriculum
from experiments.quality_masked_navigation import QualityMaskedWrapper
from neural_navigation.experiment_budget import ExperimentBudget
from neural_navigation.goal_pose import contract


class Progress(BaseCallback):
    def __init__(self,output,budget,environment):
        super().__init__();self.output=output;self.budget=budget;self.environment=environment
        self.counts={};self.episodes=[]
    def _on_step(self):
        self.environment.curriculum_level=0 if self.num_timesteps<1500 else (1 if self.num_timesteps<3000 else 2)
        for info,done in zip(self.locals['infos'],self.locals['dones']):
            bucket=self.counts.setdefault(info['docking_task'],dict(steps=0,stop_choices=0,only_stop=0,
                overrides=0,success=0,collisions=0,braking_cost=0.,hold_reward=0.))
            bucket['steps']+=1;bucket['stop_choices']+=int(info['masked_choice']==0)
            bucket['only_stop']+=int(info['masked_current_nonzero_choices']==0)
            bucket['overrides']+=int(info['nonzero_choice_changed_by_guard'])
            bucket['braking_cost']+=float(info['docking_braking_cost']);bucket['hold_reward']+=float(info['docking_holding_reward'])
            if done:
                bucket['success']+=int(info['verified_mission_success']);bucket['collisions']+=int(info['collision_events']>0)
                self.episodes.append({k:info.get(k) for k in ('verified_mission_success','collision_events',
                    'true_goal_distance_m','true_goal_yaw_error_rad','physical_goal_settled','training_case_json')})
        self.budget.tick()
        if self.num_timesteps%50==0:self.write('training')
        return True
    def write(self,phase):
        stats={k:float(v) for k,v in self.model.logger.name_to_value.items() if k.startswith('train/') and np.isscalar(v)}
        (self.output/'progress.json').write_text(json.dumps(dict(phase=phase,steps=self.num_timesteps,
            requested_steps=self.model.requested_steps,sampling=self.counts,episodes=self.episodes,
            ppo_statistics=stats,**self.budget.statistics()),indent=2))


def collect_demonstrations(env,budget,output,episodes=64):
    obs_list=[];masks=[];actions=[];episode_ids=[];results=[]
    for episode in range(episodes):
        env.curriculum_level=episode%3;obs,_=env.reset(seed=114930+episode)
        for _ in range(75):
            mask=env.action_masks();choice=demonstration_choice(obs,mask)
            obs_list.append(obs.copy());masks.append(mask.copy());actions.append(choice);episode_ids.append(episode)
            obs,_,done,truncated,info=env.step(choice);budget.tick()
            if done or truncated:break
        results.append(dict(episode=episode,level=episode%3,success=bool(info['verified_mission_success']),
            collision=bool(info['collision_events']),frames=sum(e==episode for e in episode_ids),
            xy=float(info['true_goal_distance_m']),yaw=float(info['true_goal_yaw_error_rad'])))
        (output/'demonstration_progress.json').write_text(json.dumps(dict(completed=episode+1,total=episodes,
            rows=results,**budget.statistics()),indent=2))
    data=dict(observations=np.array(obs_list),masks=np.array(masks),actions=np.array(actions),episode=np.array(episode_ids))
    np.savez_compressed(output/'demonstrations.npz',**data)
    return data,results


def augment_goal_conditions(data,copies=0):
    """Change requested heading on identical sensor states to remove shortcuts.

    These are teacher-labelled task counterfactuals, not physically completed
    demonstrations. Scan, executed history, masks, and goal position stay fixed.
    Their episode grouping is retained for train/validation separation.
    """
    if not copies:return data
    rng=np.random.default_rng(114932)
    eligible=np.flatnonzero(np.linalg.norm(data['observations'][:,218:220]*12,axis=1)<.10)
    extra={k:[] for k in data}
    for _ in range(copies):
        obs=data['observations'][eligible].copy();angles=rng.uniform(-np.pi,np.pi,len(obs))
        obs[:,300]=np.sin(angles);obs[:,301]=np.cos(angles)
        masks=data['masks'][eligible]
        choices=np.array([demonstration_choice(o,m) for o,m in zip(obs,masks)])
        extra['observations'].append(obs);extra['masks'].append(masks.copy())
        extra['actions'].append(choices);extra['episode'].append(data['episode'][eligible].copy())
    return {key:np.concatenate([value,*extra[key]]) for key,value in data.items()}


def augment_boundary_conditions(data,copies=4):
    """Teacher labels for small target changes on held-group sensor states."""
    rng=np.random.default_rng(116110)
    eligible=np.flatnonzero(np.linalg.norm(data['observations'][:,218:220]*12,axis=1)<.10)
    output={key:[value] for key,value in data.items()}
    for _ in range(copies):
        obs=data['observations'][eligible].copy();angles=rng.uniform(-.30,.30,len(obs))
        obs[:,300]=np.sin(angles);obs[:,301]=np.cos(angles)
        masks=data['masks'][eligible].copy()
        output['observations'].append(obs);output['masks'].append(masks)
        output['actions'].append(np.array([demonstration_choice(o,m) for o,m in zip(obs,masks)]))
        output['episode'].append(data['episode'][eligible].copy())
    return {key:np.concatenate(value) for key,value in output.items()}


def pretrain(model,data,budget,updates=1000,select_best=False):
    rng=np.random.default_rng(114931);policy=model.policy
    train=np.flatnonzero(data['episode']%5!=0);validation=np.flatnonzero(data['episode']%5==0)
    classes=np.unique(data['actions'][train]);by_class={int(c):train[data['actions'][train]==c] for c in classes}
    optimizer=torch.optim.Adam(policy.terminal_actor.parameters(),lr=1e-3)
    history=[];best=-1.;best_weights=None;best_update=None
    for update in range(updates):
        selected=np.array([rng.choice(by_class[int(c)]) for c in rng.choice(classes,128)])
        obs=torch.from_numpy(data['observations'][selected]);mask=data['masks'][selected]
        dist=policy.get_distribution(obs,action_masks=mask)
        loss=-dist.log_prob(torch.from_numpy(data['actions'][selected])).mean()
        optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(policy.terminal_actor.parameters(),1.);optimizer.step()
        if update%100==0 or update==updates-1:
            choice=model.predict(data['observations'][validation],action_masks=data['masks'][validation],deterministic=True)[0]
            agreement=float((choice==data['actions'][validation]).mean())
            history.append(dict(update=update+1,loss=float(loss.detach()),held_out_episode_agreement=agreement))
            if agreement>best:
                best=agreement;best_update=update+1
                best_weights={k:v.detach().clone() for k,v in policy.terminal_actor.state_dict().items()}
        budget.tick()
    if select_best:policy.terminal_actor.load_state_dict(best_weights)
    return dict(updates=updates,training_frames=len(train),validation_frames=len(validation),
        selected_update=best_update if select_best else updates,selected_by_held_out_episode_agreement=bool(select_best),
        action_counts={int(c):int((data['actions']==c).sum()) for c in np.unique(data['actions'])},history=history,
        scope='teacher agreement on held-out episodes, not navigation acceptance')


def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--components',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--steps',type=int,default=10000);p.add_argument('--cpu-cores',type=float,default=.5)
    p.add_argument('--demonstrations',type=int,default=64);p.add_argument('--bc-updates',type=int,default=1000)
    p.add_argument('--heading-augmentation',type=int,default=0)
    p.add_argument('--demonstration-data',type=Path,help='Reuse unchanged physically recorded teacher trajectories')
    p.add_argument('--resume-terminal',action='store_true',help='Continue terminal PPO weights/Adam; start fresh physical episodes')
    p.add_argument('--precision-curriculum',action='store_true',help='Tolerance-boundary heading sampling; unchanged physical reward')
    p.add_argument('--terminal-warmup',action='store_true',help='Boundary teacher warmup of retained terminal weights before PPO')
    p.add_argument('--seed',type=int,default=114933);a=p.parse_args()
    if a.output.exists() or a.steps%250 or a.steps<=0 or a.heading_augmentation<0:p.error('Fresh output, positive multiple of 250, nonnegative augmentation required')
    a.output.mkdir(parents=True);torch.set_num_threads(1);torch.set_num_interop_threads(1)
    budget=ExperimentBudget(a.cpu_cores,directory=a.output);components=a.output/'components';components.mkdir()
    for name in ('hardware.json','router.pt','localizer.pt','localization_contract.json'):
        if (a.components/name).exists():shutil.copy2(a.components/name,components/name)
    (components/'goal_pose_observation.json').write_text(json.dumps(contract(),indent=2))
    old=MaskablePPO.load(a.checkpoint,device='cpu')
    if old.observation_space.shape!=(303,) or old.action_space.n!=12:raise ValueError('303D parent required')
    scale=float(getattr(old,'training_reward_scale',1.))
    base=MaskedNavigationCurriculum(components,stage='static',route_mode='shortcut',max_steps=75,footprint_clearance_cost=1.)
    curriculum=PrecisionDockingCurriculum if a.precision_curriculum else DockingCurriculum
    env=curriculum(QualityMaskedWrapper(base,scale),gamma=old.gamma)
    if a.resume_terminal:
        model=MaskablePPO.load(a.checkpoint,env=env,device='cpu',seed=a.seed)
        if not isinstance(model.policy,DockingPolicy):raise ValueError('Terminal checkpoint required')
        if model.num_timesteps>=a.steps:raise ValueError('Requested total must exceed parent steps')
        if a.precision_curriculum:
            model.learning_rate=3e-5;model.lr_schedule=lambda _:3e-5
            model.ent_coef=.004;model.target_kl=.02
    else:
        model=MaskablePPO(DockingPolicy,env,device='cpu',seed=a.seed,n_steps=250,batch_size=125,
            n_epochs=4,learning_rate=1e-4,ent_coef=.008,target_kl=.03,gamma=old.gamma,
            gae_lambda=old.gae_lambda,policy_kwargs=old.policy_kwargs)
        model.policy.load_state_dict(old.policy.state_dict(),strict=False)
    frozen={k:v.detach().clone() for k,v in model.policy.state_dict().items() if k.startswith(('mlp_extractor.policy_net','action_net'))}
    model.training_reward_scale=scale;model.requested_steps=a.steps;del old
    source=Path(__file__).resolve().parents[2]
    config=dict(parent_sha256=hashlib.sha256(a.checkpoint.read_bytes()).hexdigest(),seed=a.seed,
        steps=a.steps,teacher_episodes=a.demonstrations,bc_updates=a.bc_updates,reward_scale=scale,
        heading_task_counterfactual_copies=a.heading_augmentation,
        parent_terminal_steps=int(model.num_timesteps),additional_requested_steps=a.steps-int(model.num_timesteps),
        reward_design='physical_distance_and_near_goal_heading_progress_v2',
        precision_curriculum=bool(a.precision_curriculum),
        learning_rate=float(model.learning_rate),entropy_coefficient=float(model.ent_coef),target_kl=model.target_kl,
        resumed_terminal_optimizer=bool(a.resume_terminal),exact_environment_rng_resume=False,
        docking_contract=docking_contract(model.policy.terminal_radius),goal_contract=contract(),localizer_router_frozen=True,
        deployment_approved=False,soft_average_cpu_cores=a.cpu_cores,memory_budget_mib=4096,
        source_sha256={str(p.relative_to(source)):hashlib.sha256(p.read_bytes()).hexdigest()
            for folder in ('training/neural_navigation','training/experiments') for p in (source/folder).glob('*.py')})
    (a.output/'config.json').write_text(json.dumps(config,indent=2))
    callback=Progress(a.output,budget,env)
    try:
        if a.resume_terminal and not a.terminal_warmup:
            data=None;results=[]
        elif a.demonstration_data:
            with np.load(a.demonstration_data) as archive:data={k:archive[k].copy() for k in archive.files}
            results=json.loads((a.demonstration_data.parent/'demonstration_progress.json').read_text())['rows']
            shutil.copy2(a.demonstration_data,a.output/'demonstrations.npz')
            config['reused_demonstrations_sha256']=hashlib.sha256(a.demonstration_data.read_bytes()).hexdigest()
        else:data,results=collect_demonstrations(env,budget,a.output,a.demonstrations)
        if not a.resume_terminal:
            config['teacher_physical_successes']=sum(r['success'] for r in results)
            augmented=augment_goal_conditions(data,a.heading_augmentation)
            config['behavior_cloning']=pretrain(model,augmented,budget,a.bc_updates,select_best=bool(a.heading_augmentation))
        elif a.terminal_warmup:
            if data is None:raise ValueError('Recorded demonstration data required for warmup')
            augmented=augment_boundary_conditions(data)
            config['boundary_behavior_cloning']=pretrain(model,augmented,budget,a.bc_updates,select_best=True)
            config['warmup_counterfactuals_are_not_physical_successes']=True
            for parameter in model.policy.terminal_actor.parameters():model.policy.optimizer.state.pop(parameter,None)
            config['terminal_optimizer_moments_reset_after_warmup']=True
        (a.output/'config.json').write_text(json.dumps(config,indent=2))
        model.save(a.output/'policy_start.zip');export_docking(model,components,a.output/'start_portable')
        if not a.resume_terminal:
            model.save(a.output/'policy_bc.zip');export_docking(model,components,a.output/'bc_portable')
        env.total_steps=0;env.curriculum_level=0 if model.num_timesteps<1500 else (1 if model.num_timesteps<3000 else 2)
        parent_steps=int(model.num_timesteps)
        milestones=[min(parent_steps+1500,a.steps),min(parent_steps+3000,a.steps),a.steps] if a.precision_curriculum else [min(2500,a.steps),min(5000,a.steps),a.steps]
        for milestone in sorted(set(milestones)):
            if milestone<=model.num_timesteps:continue
            model.learn(milestone-model.num_timesteps,callback=callback,reset_num_timesteps=model.num_timesteps==0)
            model.save(a.output/f'policy_{milestone}.zip')
            export_docking(model,components,a.output/f'portable_{milestone}')
        model.save(a.output/'policy.zip')
        for name,value in frozen.items():torch.testing.assert_close(model.policy.state_dict()[name],value,rtol=0,atol=0)
        config['frozen_navigation_weights_unchanged']=True
        (a.output/'config.json').write_text(json.dumps(config,indent=2));callback.write('complete')
    except Exception as error:
        (a.output/'ERROR.txt').write_text(repr(error));raise
    finally:env.close()


if __name__=='__main__':main()
