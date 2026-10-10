"""Learned robot navigation; offline neural-route controller teacher used only while collecting training labels."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback,CheckpointCallback,CallbackList
from navrl.geometry import ROOT,wrap
from neural_navigation.env import LearnedMultiMapEnv


def teacher(env):
    # This is OFFLINE TRAINING supervision, never used by inference/evaluation.
    if env.static_probability<.5 or len(env.path)<2:return np.zeros(2,np.float32)
    nearest=int(np.argmin(np.linalg.norm(env.path-env.estimate[:2],axis=1)))
    target=env.path[min(nearest+2,len(env.path)-1)]
    delta=target-env.estimate[:2];heading=wrap(np.arctan2(delta[1],delta[0])-env.estimate[2])
    speed=.9*max(0.,np.cos(heading)) if abs(heading)<.7 else 0.
    if np.min(np.nan_to_num(env.scan[np.abs(env.lidar.angles)<.55],posinf=12.,nan=12.))<.55:speed=0.
    return np.array([speed,np.clip(heading*1.8/.5,-1,1)],np.float32)


class Anchor(BaseCallback):
    def __init__(self,x,y):
        super().__init__();self.x=x;self.y=y;self.rng=np.random.default_rng(16000)
    def _on_step(self):return True
    def _on_rollout_start(self):
        for _ in range(4):
            index=self.rng.integers(0,len(self.x),128)
            mean=self.model.policy.get_distribution(self.x[index]).distribution.mean
            loss=.5*torch.nn.functional.mse_loss(mean,self.y[index])
            self.model.policy.optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(self.model.policy.parameters(),.5);self.model.policy.optimizer.step()


def main():
    p=argparse.ArgumentParser();p.add_argument('--episodes',type=int,default=24);p.add_argument('--epochs',type=int,default=40);p.add_argument('--steps',type=int,default=8192);p.add_argument('--output',type=Path,default=ROOT/'runs/neural_navigation');p.add_argument('--components',type=Path,default=ROOT/'runs/neural_components');p.add_argument('--parent',type=Path,default=ROOT/'runs/multimap_navigation/policy.zip')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);torch.set_num_threads(4);torch.manual_seed(16000);rng=np.random.default_rng(16000)
    raw=LearnedMultiMapEnv(components=a.components,split='train',dynamic=True,max_steps=700)
    env=Monitor(raw,str(a.output/'monitor.csv'))
    try:
        model=PPO('MlpPolicy',env,device='cpu',n_steps=256,batch_size=64,n_epochs=4,learning_rate=1e-5,gamma=.995,ent_coef=.001,policy_kwargs={'net_arch':dict(pi=[128,128],vf=[128,128])},seed=16000,verbose=1)
        old=PPO.load(a.parent,device='cpu');old_state=old.policy.state_dict();new_state=model.policy.state_dict()
        for name,tensor in new_state.items():
            if name in old_state:
                if tensor.shape==old_state[name].shape:tensor.copy_(old_state[name])
                elif tensor.ndim==2 and tensor.shape[0]==old_state[name].shape[0] and tensor.shape[1]==old_state[name].shape[1]+3:
                    tensor.zero_();tensor[:,:-3].copy_(old_state[name])
        model.policy.load_state_dict(new_state);observations=[];actions=[];outcomes=[]
        for episode in range(a.episodes):
            obs,info=env.reset(seed=16000+episode,options={'map_index':episode%6})
            for step in range(700):
                action=teacher(raw);observations.append(obs.copy());actions.append(action.copy())
                # Include learning-derived route failures as corrective supervision.
                executed=model.predict(obs,deterministic=True)[0] if rng.random()<.10 else action
                obs,_,done,timeout,info=env.step(executed)
                if done or timeout:break
            outcomes.append(dict(seed=16000+episode,map_id=info['map_id'],status=info['status'],steps=step+1,true_goal_distance_m=info['true_goal_distance_m'],localization_error_m=info['localization_error_m']))
            print('learned-system demonstration',outcomes[-1],flush=True)
        x=torch.tensor(np.asarray(observations),dtype=torch.float32);y=torch.tensor(np.asarray(actions),dtype=torch.float32)
        # Teach the actor what the neural heads reporting no route means.
        negative=x[rng.choice(len(x),max(128,len(x)//12),replace=False)].clone();negative[:,-3:-1]=0.;negative[:,-1]=0.;negative[:,216:218]=0.
        x=torch.cat([x,negative]);y=torch.cat([y,torch.zeros((len(negative),2))])
        params=list(model.policy.mlp_extractor.policy_net.parameters())+list(model.policy.action_net.parameters());opt=torch.optim.Adam(params,lr=1e-4)
        for epoch in range(a.epochs):
            losses=[]
            for index in np.array_split(rng.permutation(len(x)),max(1,len(x)//256)):
                mean=model.policy.get_distribution(x[index]).distribution.mean;loss=torch.nn.functional.mse_loss(mean,y[index])
                opt.zero_grad();loss.backward();opt.step();losses.append(float(loss.detach()))
            if epoch%10==0:print('actor cloning',epoch,float(np.mean(losses)),flush=True)
        with torch.no_grad():model.policy.log_std.fill_(-2.3)
        model.save(a.output/'warm_start');np.savez_compressed(a.output/'demonstrations.npz',observations=x.numpy(),actions=y.numpy())
        config=dict(architecture='learned local pose alignment + trained neural goal propagation + PPO control',runtime_amcl=False,runtime_particle_filter=False,runtime_robot_astar=False,runtime_robot_supervisor_override=False,initial_pose='coarse prior provided',training_teacher_only=True,ground_truth_in_policy_inputs=False,lidar=raw.lidar.signature,scene_file='physical_scene.xml',components=str(a.components.resolve()),demonstrations=outcomes,teacher_samples=len(x),requested_ppo_steps=a.steps,observation_dimensions=231)
        (a.output/'config.json').write_text(json.dumps(config,indent=2))
        callback=CallbackList([Anchor(x,y),CheckpointCallback(save_freq=4000,save_path=str(a.output/'checkpoints'),name_prefix='ppo')])
        model.learn(total_timesteps=a.steps,callback=callback);model.save(a.output/'policy')
        config.update(completed_ppo_steps=model.num_timesteps,sampled_tasks=raw.task_history)
        (a.output/'config.json').write_text(json.dumps(config,indent=2));print('Saved learned navigation policy',a.output,flush=True)
    finally:env.close()


if __name__=='__main__':main()
