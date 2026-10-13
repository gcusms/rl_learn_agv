"""Continue an explicit policy with frozen learned components and parallel physics."""
import argparse,json,time,hashlib,shutil,sys,os
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from train_room_navigation_v3 import TrainingMission
from neural_navigation.curriculum import NavigationCurriculum
from neural_navigation.guarded_curriculum import GuardedNavigationCurriculum
from navrl.geometry import ROOT

def make_env(rank,components,output,safety_training=False,time_cost=0.,curriculum='room',route_mode='raw',execution_mode='raw',projection_cost=.25,motion_profiles=False,footprint_clearance_cost=0.):
    def create():
        torch.set_num_threads(1)
        if curriculum=='room':
            e=TrainingMission(components,safety_training=safety_training,time_cost=time_cost)
            e.active.route_mode=route_mode
        elif execution_mode=='guarded':e=GuardedNavigationCurriculum(components,stage=curriculum,route_mode=route_mode,projection_cost=projection_cost,motion_profiles=motion_profiles,footprint_clearance_cost=footprint_clearance_cost)
        else:e=NavigationCurriculum(components,stage=curriculum,route_mode=route_mode,motion_profiles=motion_profiles)
        fields=('mission_targets_reached','verified_mission_success','collision_events','safety_termination')
        if curriculum!='room':fields+=('training_case_json',)
        if execution_mode=='guarded':fields+=('execution_guard_overrides','execution_guard_steps','execution_guard_override_fraction')
        return Monitor(e,str(output/f'worker_{rank}.csv'),info_keywords=fields)
    return create

class Progress(BaseCallback):
    def __init__(self,output,persistent,x,y,steps,checkpoint_every,anchor_batches):
        super().__init__();self.output=output;self.persistent=persistent;self.x=x;self.y=y;self.steps=steps;self.every=checkpoint_every;self.anchor_batches=anchor_batches;self.rng=np.random.default_rng(33000);self.next_save=checkpoint_every;self.next_report=1000;self.start=time.monotonic();self.episodes=[]
    def _on_rollout_start(self):
        for _ in range(self.anchor_batches):
            ix=self.rng.integers(len(self.x),size=256)
            mean=self.model.policy.get_distribution(self.x[ix]).distribution.mean
            loss=.2*torch.nn.functional.mse_loss(mean,self.y[ix]);self.model.policy.optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(self.model.policy.parameters(),.5);self.model.policy.optimizer.step()
    def _on_step(self):
        for worker,info in enumerate(self.locals['infos']):
            if 'episode' in info:
                record={'step':self.num_timesteps,'worker':worker,**info['episode']};self.episodes.append(record)
                with (self.output/'episodes.jsonl').open('a') as f:f.write(json.dumps(record,default=float)+'\n')
        if self.num_timesteps>=self.next_report:
            elapsed=time.monotonic()-self.start
            status=dict(steps=self.num_timesteps,requested_steps=self.steps,elapsed_seconds=elapsed,steps_per_second=self.num_timesteps/max(elapsed,.001),device=str(self.model.device),episodes=len(self.episodes),training_missions_verified=sum(bool(e.get('verified_mission_success')) for e in self.episodes),gpu_allocated_mb=torch.cuda.memory_allocated()/2**20 if self.model.device.type=='cuda' else 0.)
            (self.output/'progress.json').write_text(json.dumps(status,indent=2));print('PROGRESS',json.dumps(status),flush=True);self.next_report+=1000
        if self.num_timesteps>=self.next_save:
            path=self.output/f'policy_{self.num_timesteps}.zip';self.model.save(path)
            if self.persistent:shutil.copy2(path,self.persistent/path.name)
            self.next_save+=self.every
        return True

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--steps',type=int,default=100000)
    p.add_argument('--envs',type=int,default=4)
    p.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--persistent',type=Path)
    p.add_argument('--checkpoint-every',type=int,default=10000)
    p.add_argument('--anchor-batches',type=int,default=4)
    p.add_argument('--parent-policy',type=Path,default=ROOT/'runs/full_model_navigation_v3/policy.zip')
    p.add_argument('--components',type=Path,help='Defaults to the selected policy directory')
    p.add_argument('--demonstrations',type=Path,default=ROOT/'runs/full_model_navigation_v3/demonstrations.npz')
    p.add_argument('--n-steps',type=int,default=250)
    p.add_argument('--batch-size',type=int,default=250)
    p.add_argument('--epochs',type=int,default=4)
    p.add_argument('--learning-rate',type=float,default=1e-5)
    p.add_argument('--target-kl',type=float,default=.02)
    p.add_argument('--seed',type=int,default=33000)
    p.add_argument('--scale-anchors',action='store_true',help='Preserve demonstration batches per transition relative to 4 x 250 baseline')
    p.add_argument('--safety-training',action='store_true',help='Terminate and penalize contact or simulator penetration rollback')
    p.add_argument('--time-cost',type=float,default=0.,help='Additional cost per 0.2s transition, on top of existing 0.01')
    p.add_argument('--curriculum',choices=['room','static','dynamic','robust'],default='room',help='Independent train-map tasks use fixed geodesic reward and terminal contact penalty')
    p.add_argument('--route-mode',choices=['raw','shortcut'],default='raw')
    p.add_argument('--execution-mode',choices=['raw','guarded'],default='raw')
    p.add_argument('--projection-cost',type=float,default=.25)
    p.add_argument('--motion-profiles',action='store_true',help='Independent crossing tasks can reverse once or pause then resume; no periodic motion')
    p.add_argument('--footprint-clearance-cost',type=float,default=0.,help='Guarded-only measured rectangular-body clearance cost, target 0.16m')
    a=p.parse_args()
    if min(a.steps,a.envs,a.n_steps,a.batch_size,a.epochs,a.checkpoint_every)<=0 or a.anchor_batches<0 or a.learning_rate<=0 or a.target_kl<=0 or not np.isfinite(a.time_cost) or a.time_cost<0:
        p.error('Training sizes, learning rate and target KL must be positive; anchor batches nonnegative')
    a.parent_policy=a.parent_policy.resolve();a.components=(a.components or a.parent_policy.parent).resolve()
    a.output=a.output.resolve();a.demonstrations=a.demonstrations.resolve()
    sources={'policy.zip':a.parent_policy,'localizer.pt':a.components/'localizer.pt','router.pt':a.components/'router.pt','demonstrations.npz':a.demonstrations}
    if (a.components/'hardware.json').exists():sources['hardware.json']=a.components/'hardware.json'
    for source in sources.values():
        if not source.is_file():p.error(f'Missing training input: {source}')
    if a.batch_size>a.envs*a.n_steps:p.error('Batch size exceeds rollout size')
    if a.curriculum!='room' and a.time_cost!=0:p.error('Curriculum has its own explicit reward; do not also set time-cost')
    if a.execution_mode=='guarded' and (a.curriculum=='room' or a.anchor_batches):p.error('Guarded training requires independent curriculum and no auxiliary action fitting')
    if not np.isfinite(a.projection_cost) or a.projection_cost<0:p.error('Projection cost must be nonnegative')
    if a.motion_profiles and a.curriculum in ('room','static'):p.error('Motion profiles require dynamic or robust curriculum')
    if not np.isfinite(a.footprint_clearance_cost) or a.footprint_clearance_cost<0 or (a.footprint_clearance_cost and a.execution_mode!='guarded'):p.error('Nonnegative footprint clearance cost requires guarded mode')
    if a.output.exists() and any(a.output.iterdir()):p.error('Use a fresh empty output directory')
    effective_anchors=round(a.anchor_batches*a.envs*a.n_steps/1000) if a.scale_anchors else a.anchor_batches
    torch.set_num_threads(1);torch.manual_seed(a.seed);np.random.seed(a.seed)
    if a.device=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA requested but unavailable')
    if (a.output/'policy.zip').exists() or (a.output/'progress.json').exists():raise RuntimeError('Use a fresh run directory; existing results must remain intact')
    a.output.mkdir(parents=True,exist_ok=True)
    if a.persistent:a.persistent.mkdir(parents=True,exist_ok=True)
    hashes={n:hashlib.sha256(path.read_bytes()).hexdigest() for n,path in sources.items()}
    # Keep launch-time source evidence; model SHA alone cannot reproduce reward
    # changes or geometric inference changes between experiments.
    project=ROOT.parent;code_sources=set()
    for directory in [ROOT,ROOT/'navrl',ROOT/'neural_navigation',ROOT/'remote']:
        code_sources.update(directory.glob('*.py'))
    code_hashes={str(path.relative_to(project)):hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(code_sources)}
    for path in sorted(code_sources):
        destination=a.output/'code_snapshot'/path.relative_to(project)
        destination.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,destination)
    for n in ['localizer.pt','router.pt']:shutil.copy2(sources[n],a.output/n)
    if 'hardware.json' in sources:shutil.copy2(sources['hardware.json'],a.output/'hardware.json')
    data=np.load(a.demonstrations);x=torch.as_tensor(data['observations'],device=a.device);y=torch.as_tensor(data['actions'],device=a.device)
    if x.ndim!=2 or x.shape[1]!=235 or y.shape!=(len(x),2) or not len(x) or not torch.isfinite(x).all() or not torch.isfinite(y).all():
        raise ValueError('Demonstrations must contain finite 235-dimensional observations and 2-dimensional actions')
    env=SubprocVecEnv([make_env(i,a.output,a.output,a.safety_training,a.time_cost,a.curriculum,a.route_mode,a.execution_mode,a.projection_cost,a.motion_profiles,a.footprint_clearance_cost) for i in range(a.envs)],start_method='spawn');env.seed(a.seed)
    try:
        model=PPO.load(a.parent_policy,env=env,device=a.device,n_steps=a.n_steps,batch_size=a.batch_size,n_epochs=a.epochs,learning_rate=a.learning_rate,target_kl=a.target_kl,seed=a.seed,verbose=0)
        config=dict(parent_weight_sha256=hashes,resume_source=str(a.parent_policy),components_source=str(a.components),demonstrations_source=str(a.demonstrations),additional_ppo_steps_requested=a.steps,num_envs=a.envs,device=a.device,cpu_threads_per_worker=1,seed=a.seed,learning_rate=a.learning_rate,n_steps=a.n_steps,batch_size=a.batch_size,n_epochs=a.epochs,target_kl=a.target_kl,anchor_batches=effective_anchors,scale_anchors=a.scale_anchors,safety_training=a.safety_training,offline_anchor_samples=len(x),runtime_robot_amcl=False,runtime_robot_astar=False,robot_action_overrides=False,training_truth_labels_only=True,training_scope='Continued selected action policy; learned localization/router frozen; randomized starts, headings and goal orders on existing room geometry',started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),status='running')
        config['additional_time_cost_per_step']=a.time_cost
        config['launch_source_sha256']=code_hashes
        config.update(curriculum=a.curriculum,route_mode=a.route_mode)
        config['footprint_clearance_cost']=a.footprint_clearance_cost
        config['motion_profiles']=a.motion_profiles
        if a.motion_profiles:config['dynamic_profile_distribution']='35% dynamic, 65% retained static; crossing half of dynamic, then 50% one-way / 25% one reversal / 25% one pause-resume; speeds 0.08-0.16m/s'
        config.update(execution_mode=a.execution_mode,projection_cost=a.projection_cost if a.execution_mode=='guarded' else 0.,robot_action_overrides=a.execution_mode=='guarded')
        if a.curriculum!='room':
            config.update(safety_training=True,training_scope='Independent randomized tasks on six train layouts and the known room; previous static tasks retained; validation/final acceptance cases excluded',
                          reward='15*fixed static geodesic progress -0.05/step -0.1*travel -jerk -near-scan -long-reverse; verified goal +50, contact/rollback terminal -100; false goal -50',
                          reward_truth_labels_only=True)
        if a.execution_mode=='guarded':
            config['reward']+=' - projection_cost * squared(proposed minus executed action) - footprint_clearance_cost * squared(max(0,(0.16 - measured body gap)/0.16))'
            config['ppo_semantics']='Proposed latent action likelihood; return from shared deterministic deployed guard transform. Executed action history is used. Overrides measured, not called pure learned avoidance.'
        (a.output/'config.json').write_text(json.dumps(config,indent=2));print('START',json.dumps(config),flush=True)
        callback=Progress(a.output,a.persistent,x,y,a.steps,a.checkpoint_every,effective_anchors);callback.rng=np.random.default_rng(a.seed)
        start=time.monotonic();model.learn(total_timesteps=a.steps,callback=callback,reset_num_timesteps=True);model.save(a.output/'policy.zip')
        config.update(status='complete',additional_ppo_steps_completed=model.num_timesteps,wall_seconds=time.monotonic()-start,finished_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
        (a.output/'config.json').write_text(json.dumps(config,indent=2))
        assert hashes=={n:hashlib.sha256(path.read_bytes()).hexdigest() for n,path in sources.items()},'Training source changed'
        if a.persistent:
            for n in ['policy.zip','localizer.pt','router.pt','config.json','progress.json']:shutil.copy2(a.output/n,a.persistent/n)
        print('COMPLETE',json.dumps(config),flush=True)
    finally:env.close()

if __name__=='__main__':main()
