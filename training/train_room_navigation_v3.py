"""Continue the archived actor using room corrections, physical demonstrations and PPO.
Teachers are used solely in training; evaluation executes the trained actor unchanged.
"""
import argparse,json,hashlib,shutil
from pathlib import Path
from collections import deque
import numpy as np
import torch
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback,CheckpointCallback,CallbackList
from navrl.geometry import ROOT
from navrl.map import continuous_remaining
from navrl.scenarios import configure_column_task
from neural_navigation.room_env import RoomNavigationEnv,extend_observation,corridor_clearance


def training_label(obs):
    """Offline route-following labels from observable heading and corridor returns."""
    target=obs[216:218]*2;norm=float(np.linalg.norm(target))
    if norm<.02 or obs[-7]<.5 or obs[-6]<.15:return np.zeros(2,np.float32)
    heading=float(np.arctan2(obs[231],obs[232]));clearance=float(obs[234]*2)
    if clearance<.31 and abs(heading)>.6:return np.array([-.55,0],np.float32)
    turn=float(np.clip(heading*3.6,-1,1))
    speed=.85*max(0.,np.cos(heading))*min(1.,norm/.35) if abs(heading)<.7 else 0.
    speed*=np.clip((clearance-.27)/.30,0,1)
    return np.array([speed,turn],np.float32)


def replay_observations(trace):
    history=deque(maxlen=3);positions=deque(maxlen=16);x=[]
    angles=np.linspace(trace['lidar']['angle_min'],trace['lidar']['angle_max'],220)
    for s in trace['samples']:
        scan=np.array([12. if v is None else v for v in s['scan']]);bins=np.array([p.min() for p in np.array_split(scan,72)])/12
        if not history:
            for _ in range(3):history.append(bins)
        else:history.append(bins)
        pose=np.array(s['estimate']);path=np.array(s['planned_path']);goal=np.array(s['active_goal']);positions.append(pose[:2])
        i=np.argmin(np.linalg.norm(path-pose[:2],axis=1));target=path[min(i+2,len(path)-1)]
        yaw=pose[2];r=np.array([[np.cos(yaw),np.sin(yaw)],[-np.sin(yaw),np.cos(yaw)]])
        decision=s['decision'];vel=np.array(s['actual_velocity']);last=np.array(s['commanded_velocity'])/[.2,.5]
        features=np.r_[r@(target-pose[:2])/2,r@(goal-pose[:2])/12,vel/[.2,.5],last,np.array(decision['localization_std'])/[.5,.5,.5],
                       np.linalg.norm(path-pose[:2],axis=1).min()/2,decision['static_reachability_probability'],decision['current_reachability_probability'],min(continuous_remaining(path,pose[:2])/20,1)]
        base=np.clip(np.r_[np.concatenate(history),features],-1,1).astype(np.float32)
        progress=np.clip(np.linalg.norm(pose[:2]-positions[0])/.15,0,1) if len(positions)>5 else 1.
        x.append(extend_observation(base,progress,corridor_clearance(scan,angles)))
    return np.array(x)


class TrainingMission(gym.Env):
    def __init__(self,components):
        self.active=RoomNavigationEnv(components=components,scene_dir=ROOT/'assets/room_navigation_scripted',dynamic=True,max_steps=2800)
        self.action_space=self.active.action_space;self.observation_space=self.active.observation_space
        self.corners=np.array([[1.2,1.2],[1.2,6.6],[8.6,6.6],[8.6,1.2]])
    def reset(self,seed=None,options=None):
        super().reset(seed=seed);rng=self.np_random;order=rng.permutation(4);points=self.corners+rng.uniform(-.18,.18,(4,2));self.goals=[points[k] for k in list(order[1:])+[order[0]]]
        self.index=0;self.leg_steps=0;self.verified=[];self.active.dynamic=True
        start=[*points[order[0]],float(rng.uniform(-np.pi,np.pi))]
        obs,info=self.active.reset(seed=seed,options={'task':0,'start':start,'goal':self.goals[0]});obs=configure_column_task(self.active,0)
        return obs,info
    def step(self,action):
        obs,reward,done,timeout,info=self.active.step(action);self.leg_steps+=1
        if done:
            self.verified.append(bool(info['true_goal_distance_m']<=.30));self.index+=1
            if self.index<len(self.goals):
                obs=self.active.set_navigation_goal(self.goals[self.index]);self.leg_steps=0;done=False
        if self.leg_steps>=700 and not done:timeout=True
        info['mission_targets_reached']=sum(self.verified);info['verified_mission_success']=bool(done and len(self.verified)==4 and all(self.verified))
        return obs,reward,done,timeout,info
    def close(self):self.active.close()


class Anchor(BaseCallback):
    def __init__(self,x,y):super().__init__();self.x=x;self.y=y;self.rng=np.random.default_rng(30000)
    def _on_step(self):return True
    def _on_rollout_start(self):
        for _ in range(4):
            indices=self.rng.integers(len(self.x),size=256);mean=self.model.policy.get_distribution(self.x[indices]).distribution.mean
            loss=.5*torch.nn.functional.mse_loss(mean,self.y[indices]);self.model.policy.optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(self.model.policy.parameters(),.5);self.model.policy.optimizer.step()


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=ROOT/'runs/room_navigation_v3');p.add_argument('--episodes',type=int,default=6);p.add_argument('--epochs',type=int,default=100);p.add_argument('--steps',type=int,default=8192);p.add_argument('--collect-only',action='store_true');a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=True);torch.set_num_threads(2);torch.manual_seed(30000);rng=np.random.default_rng(30000)
    parent=ROOT/'runs/full_model_navigation_v2';hashes={n:hashlib.sha256((parent/n).read_bytes()).hexdigest() for n in ['policy.zip','localizer.pt','router.pt']}
    for name in ['localizer.pt','router.pt']:shutil.copy2(parent/name,a.output/name)
    raw=TrainingMission(parent);env=Monitor(raw,str(a.output/'monitor.csv'))
    try:
        model=PPO('MlpPolicy',env,n_steps=256,batch_size=128,n_epochs=4,learning_rate=2e-5,gamma=.995,ent_coef=.001,policy_kwargs={'net_arch':dict(pi=[128,128],vf=[128,128])},device='cpu',seed=30000,verbose=1)
        old=PPO.load(parent/'policy.zip',device='cpu').policy.state_dict();state=model.policy.state_dict()
        for name,tensor in state.items():
            if name not in old:continue
            if tensor.shape==old[name].shape:tensor.copy_(old[name])
            elif tensor.ndim==2 and tensor.shape[0]==old[name].shape[0] and tensor.shape[1]==old[name].shape[1]+4:
                tensor.zero_();tensor[:,:-4].copy_(old[name])
        model.policy.load_state_dict(state)
        xs=[];ys=[];outcomes=[]
        for i in range(3):
            trace=json.loads((ROOT/f'runs/room_navigation_scripted/mission_{i}/replay.json').read_text());x=replay_observations(trace);xs.extend(x);ys.extend(training_label(v) for v in x)
        olddata=np.load(ROOT/'runs/neural_navigation_v2/demonstrations.npz')['observations'];indices=rng.choice(len(olddata),min(2200,len(olddata)),replace=False)
        for obs in olddata[indices]:v=extend_observation(obs);xs.append(v);ys.append(training_label(v))
        # Heading coverage prevents saturated backwards-turn states being
        # underrepresented. These are observable-feature training labels only.
        for _ in range(10000):
            v=np.asarray(xs[rng.integers(len(xs))],np.float32).copy()
            heading=float(rng.uniform(-np.pi,np.pi));distance=float(rng.uniform(.12,.7))
            v[216:218]=distance*np.array([np.cos(heading),np.sin(heading)])/2
            v[231:233]=[np.sin(heading),np.cos(heading)]
            v[228:230]=1.;v[234]=rng.uniform(.1,1.)
            xs.append(v);ys.append(training_label(v))
        print('Corrective replay + older maps:',len(xs),'offline samples',flush=True)
        for episode in range(a.episodes):
            if episode<4:
                obs,info=env.reset(seed=30000+episode)
                max_steps=2200
            else:
                e=raw.active;e.dynamic=False;start=[4.35,5.45,float(rng.uniform(-np.pi,np.pi))] if episode%2==0 else [5.5,4.5,float(rng.uniform(-np.pi,np.pi))]
                goal=[8.4,6.4] if episode%2==0 else [1.4,6.4]
                obs,info=e.reset(seed=30000+episode,options={'task':0,'start':start,'goal':goal});max_steps=650
            for step in range(max_steps):
                label=training_label(obs);xs.append(obs.copy());ys.append(label.copy())
                executed=model.predict(obs,deterministic=True)[0] if rng.random()<.10 else label
                if episode<4:obs,_,done,timeout,info=env.step(executed)
                else:obs,_,done,timeout,info=raw.active.step(executed)
                if done or timeout:break
            outcome=dict(episode=episode,source='offline training teacher; not policy evaluation',steps=step+1,verified_targets=info.get('mission_targets_reached'),verified_mission=info.get('verified_mission_success'),status=info['status'],true_goal_distance_m=info['true_goal_distance_m'],collision_events=info['collision_events'])
            outcomes.append(outcome);print('training demonstration',outcome,flush=True)
            np.savez_compressed(a.output/'demonstrations.npz',observations=np.asarray(xs,np.float32),actions=np.asarray(ys,np.float32))
        x=torch.from_numpy(np.asarray(xs,np.float32));y=torch.from_numpy(np.asarray(ys,np.float32))
        config=dict(revision='v3',policy_environment='RoomNavigationEnv',observation_dimensions=235,action_limits={'forward_m_s':.2,'reverse_m_s':.08,'turn_rad_s':.5},
                    components=str(a.output.resolve()),lidar=raw.active.lidar.signature,scene_file='physical_scene.xml',runtime_amcl=False,runtime_robot_astar=False,runtime_robot_supervisor_override=False,
                    ground_truth_in_policy_inputs=False,teacher_training_only=True,parent_weight_sha256=hashes,training_demonstrations=outcomes,training_samples=len(x),
                    requested_ppo_steps=a.steps,training_maps='Original v2 samples plus observed room failures and jittered room missions; original three tasks are regression cases, not independent held-out tests')
        (a.output/'config.json').write_text(json.dumps(config,indent=2))
        if a.collect_only:return
        params=list(model.policy.mlp_extractor.policy_net.parameters())+list(model.policy.action_net.parameters());optimizer=torch.optim.Adam(params,lr=3e-4)
        for epoch in range(a.epochs):
            losses=[]
            for indices in np.array_split(rng.permutation(len(x)),max(1,len(x)//256)):
                mean=model.policy.get_distribution(x[indices]).distribution.mean
                loss=torch.nn.functional.mse_loss(mean,y[indices]);optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(params,2.);optimizer.step();losses.append(float(loss.detach()))
            if epoch%10==0:print('room actor cloning',epoch,'loss',float(np.mean(losses)),flush=True)
        with torch.no_grad():model.policy.log_std.fill_(-2.8)
        model.save(a.output/'warm_start');print('Warm actor saved; continuing PPO',flush=True)
        raw.active.dynamic=True
        callback=CallbackList([Anchor(x,y),CheckpointCallback(save_freq=4096,save_path=str(a.output/'checkpoints'),name_prefix='ppo')])
        model.learn(total_timesteps=a.steps,callback=callback);model.save(a.output/'policy')
        config.update(completed_ppo_steps=model.num_timesteps);(a.output/'config.json').write_text(json.dumps(config,indent=2))
        if hashes!={n:hashlib.sha256((parent/n).read_bytes()).hexdigest() for n in hashes}:raise RuntimeError('Archived parent weights changed')
        print('Training complete:',a.output,flush=True)
    finally:env.close()

if __name__=='__main__':main()
