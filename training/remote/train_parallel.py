"""Resume frozen v3 with parallel CPU physics and a selectable CPU/CUDA PPO actor."""
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
from navrl.geometry import ROOT

def make_env(rank,components,output):
    def create():
        torch.set_num_threads(1)
        e=TrainingMission(components)
        return Monitor(e,str(output/f'worker_{rank}.csv'),info_keywords=('mission_targets_reached','verified_mission_success','collision_events'))
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
        for info in self.locals['infos']:
            if 'episode' in info:
                record={'step':self.num_timesteps,**info['episode']};self.episodes.append(record)
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
    p=argparse.ArgumentParser();p.add_argument('--steps',type=int,default=100000);p.add_argument('--envs',type=int,default=4);p.add_argument('--device',choices=['cpu','cuda'],default='cuda');p.add_argument('--output',type=Path,required=True);p.add_argument('--persistent',type=Path);p.add_argument('--checkpoint-every',type=int,default=10000);p.add_argument('--anchor-batches',type=int,default=4);a=p.parse_args()
    torch.set_num_threads(1);torch.manual_seed(33000);np.random.seed(33000)
    if a.device=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA requested but unavailable')
    if (a.output/'policy.zip').exists() or (a.output/'progress.json').exists():raise RuntimeError('Use a fresh run directory; existing results must remain intact')
    a.output.mkdir(parents=True,exist_ok=True)
    if a.persistent:a.persistent.mkdir(parents=True,exist_ok=True)
    parent=ROOT/'runs/full_model_navigation_v3';hashes={n:hashlib.sha256((parent/n).read_bytes()).hexdigest() for n in ['policy.zip','localizer.pt','router.pt']}
    for n in ['localizer.pt','router.pt']:shutil.copy2(parent/n,a.output/n)
    data=np.load(parent/'demonstrations.npz');x=torch.as_tensor(data['observations'],device=a.device);y=torch.as_tensor(data['actions'],device=a.device)
    env=SubprocVecEnv([make_env(i,parent,a.output) for i in range(a.envs)],start_method='spawn');env.seed(33000)
    try:
        model=PPO.load(parent/'policy.zip',env=env,device=a.device,n_steps=250,batch_size=250,n_epochs=4,learning_rate=1e-5,seed=33000,verbose=0)
        config=dict(parent_weight_sha256=hashes,resume_source=str(parent/'policy.zip'),additional_ppo_steps_requested=a.steps,num_envs=a.envs,device=a.device,cpu_threads_per_worker=1,seed=33000,learning_rate=1e-5,n_steps=250,batch_size=250,n_epochs=4,anchor_batches=a.anchor_batches,offline_anchor_samples=len(x),runtime_robot_amcl=False,runtime_robot_astar=False,robot_action_overrides=False,training_truth_labels_only=True,training_scope='Continued v3 action policy; learned localization/router frozen; randomized starts, headings and goal orders on existing room geometry',started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),status='running')
        (a.output/'config.json').write_text(json.dumps(config,indent=2));print('START',json.dumps(config),flush=True)
        callback=Progress(a.output,a.persistent,x,y,a.steps,a.checkpoint_every,a.anchor_batches);start=time.monotonic();model.learn(total_timesteps=a.steps,callback=callback,reset_num_timesteps=True);model.save(a.output/'policy.zip')
        config.update(status='complete',additional_ppo_steps_completed=model.num_timesteps,wall_seconds=time.monotonic()-start,finished_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
        (a.output/'config.json').write_text(json.dumps(config,indent=2))
        assert hashes=={n:hashlib.sha256((parent/n).read_bytes()).hexdigest() for n in hashes},'Parent changed'
        if a.persistent:
            for n in ['policy.zip','localizer.pt','router.pt','config.json','progress.json']:shutil.copy2(a.output/n,a.persistent/n)
        print('COMPLETE',json.dumps(config),flush=True)
    finally:env.close()

if __name__=='__main__':main()
