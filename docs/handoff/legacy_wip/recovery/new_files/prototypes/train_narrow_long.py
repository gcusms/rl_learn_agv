"""Resource-limited 50k fine-action continuation, checkpoint/eval every 5k."""
import argparse,hashlib,json,time,shutil
from pathlib import Path
from collections import Counter
import numpy as np
import torch
from sb3_contrib import MaskablePPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from neural_navigation.masked_actions import ACTIONS as OLD_ACTIONS
from prototypes.narrow_observation import masks,ACTIONS,contract
from prototypes.narrow_training_env import NarrowTrainingEnv
from prototypes.narrow_long_curriculum import LongNarrowCurriculum
from prototypes.narrow_passage import generate
from prototypes.narrow_ppo import PortableFineActor

CASES=[dict(id='centered',seed=83120,start=[1.4,2.,0.],goal=[5.,2.]),
       dict(id='offset_left',seed=83121,start=[1.5,2.015,.06],goal=[5.,2.]),
       dict(id='offset_right',seed=83122,start=[1.5,1.985,-.06],goal=[5.,2.]),
       dict(id='short_exit',seed=84120,start=[3.7,2.,0.],goal=[4.55,2.]),
       dict(id='open_room',seed=84121,start=[.9,1.95,.05],goal=[1.65,2.])]

class Progress(BaseCallback):
    def __init__(self,status,save,output):super().__init__();self.status=status;self.save=save;self.output=output
    def _on_step(self):
        for info in self.locals['infos']:
            if info['nonzero_choice_changed_by_guard']:raise RuntimeError('Sampled/executed command mismatch')
            self.status['fine_choices']+=int(info['fine_choice'])
            stage=info['training_stage'];counts=self.status['stage_frame_counts'];counts[stage]=counts.get(stage,0)+1
            if 'episode' in info:
                self.status['episodes']+=1
                self.status['training_successes']+=int(info['verified_mission_success'])
                self.status['training_collision_episodes']+=int(info['collision_events'] or info['physical_guards'])
        self.status['steps']=self.num_timesteps
        if self.num_timesteps%100==0:
            self.save()
            if shutil.disk_usage(self.output).free<512*1024**2:
                self.model.save(self.output/'low_disk_checkpoint.zip');raise RuntimeError('Low disk: saved checkpoint and stopped')
        if self.num_timesteps%1000==0:print('TRAIN',self.num_timesteps,'/',self.status['requested_steps'],flush=True)
        return True

def evaluate(model,scene,bundle,output,old=False):
    output.mkdir();results=[]
    for case in CASES:
        env=NarrowTrainingEnv(scene,bundle,seed=case['seed'],max_steps=250);rows=[]
        try:
            obs,_=env.reset(options=dict(start=case['start'],goal=case['goal'],case_seed=case['seed']))
            for _ in range(250):
                mask=env.action_masks()
                action,_=model.predict(obs[:300] if old else obs,action_masks=mask[:12] if old else mask,deterministic=True)
                obs,reward,done,truncated,info=env.step(int(action))
                rows.append(dict(time=float(env.env.elapsed),choice=int(action),command=info['commanded_velocity'],
                    pose=env.runtime.core.estimate.tolist(),route_state=env.runtime.core.narrow_route_state))
                if done or truncated:break
            result=dict(info,case_id=case['id'],frames=len(rows),timeout=truncated,
                        fine_choices=sum(r['choice']>=12 for r in rows))
            (output/f"{case['id']}.json").write_text(json.dumps(dict(case=case,samples=rows,result=result)))
            results.append(result)
        finally:env.close()
    return results

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--steps',type=int,default=50000);p.add_argument('--interval',type=int,default=5000)
    p.add_argument('--checkpoint',type=Path,default=Path('training/runs/narrow_fine_ppo_1000_v2_20261013/policy.zip'))
    p.add_argument('--bundle',type=Path,default=Path('models/recovery_20261012/portable'));args=p.parse_args()
    if args.output.exists() or args.interval%250 or args.interval<=0 or args.steps<=0 or args.steps%args.interval:p.error('Fresh output and positive interval/step multiples required')
    if shutil.disk_usage(args.output.parent).free<1024**3:p.error('Need 1 GiB free')
    torch.set_num_threads(1);args.output.mkdir(parents=True)
    hardware=json.loads((args.bundle/'hardware.json').read_text())
    names=['prototypes/train_narrow_long.py','prototypes/narrow_long_curriculum.py','prototypes/narrow_ppo.py',
        'prototypes/narrow_training_env.py','prototypes/narrow_fine_runtime.py','prototypes/narrow_alignment.py',
        'prototypes/narrow_recovery_guard.py','prototypes/surface_localizer.py','prototypes/entrance_staging.py','prototypes/entrance_guard.py',
        'prototypes/narrow_observation.py','prototypes/narrow_passage.py','prototypes/narrow_runtime.py',
        'prototypes/anchored_narrow_runtime.py','training/neural_navigation/runtime.py','training/neural_navigation/core.py']
    names+= [str(f) for f in Path('training/neural_navigation').glob('*.py')]
    names+= [str(f) for f in Path('training/experiments').glob('*.py')]
    names+= [str(f) for f in Path('training/navrl').glob('*.py')]
    names=sorted(set(names));hashes={f:hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in names}
    for f in names:
        dest=args.output/'source_snapshot'/f;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(Path(f).read_bytes())
    (args.output/'fixed_development_cases.json').write_text(json.dumps(CASES,indent=2))
    status=dict(status='running',phase='preflight',steps=0,requested_steps=args.steps,checkpoint_interval=args.interval,
        started_unix=time.time(),fine_choices=0,episodes=0,training_successes=0,training_collision_episodes=0,
        stage_frame_counts={},checkpoints=[],checks={},source_sha256=hashes,
        parent_checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        published_actor_sha256=hashlib.sha256((args.bundle/'actor.pt').read_bytes()).hexdigest(),
        old_model_replaced=False,deployment_approved=False,
        fine_prior_penalty_before=None,fine_prior_penalty_after=.2,entropy_coefficient=.015,
        observation_action_contract=contract(),runtime_adapter_repair=True,
        optimizer_state_retained=True,critic_retained=True,seed=73123,
        limitation='Static 15cm straight corridor only; five repeated development tasks, not independent deployment acceptance.')
    def save():
        status['updated_unix']=time.time()
        temp=args.output/'status.tmp';temp.write_text(json.dumps(status,indent=2));temp.replace(args.output/'status.json')
        lines=[f'# {args.steps:,}步窄道PPO续训','',f"状态：{status['status']} / {status['phase']}；新增{status['steps']}/{args.steps}步。",'',
            '原发布模型保留。训练到达数是课程内统计，不代表独立成功率。','',
            '| 检查点 | 完整窄道到达 | 出口／简单回放到达 | 碰撞或保护任务 |','|---|---|---|---|']
        for row in status['checkpoints']:
            cases=row['cases'];lines.append(f"| {row['steps']} | {sum(c['verified_mission_success'] for c in cases[:3])}/3 | {sum(c['verified_mission_success'] for c in cases[3:])}/2 | {sum(bool(c['collision_events'] or c['physical_guards']) for c in cases)} |")
        (args.output/'REPORT.md').write_text('\n'.join(lines)+'\n')
    save();env=None;model=None
    try:
        scenes=generate(args.output/'scenes',hardware);scene=Path(scenes[0]['scene'])
        training=LongNarrowCurriculum(scene,args.bundle,seed=73123,max_steps=250)
        env=Monitor(training,str(args.output/'monitor.csv'),info_keywords=('verified_mission_success','training_stage'))
        model=MaskablePPO.load(args.checkpoint,env=env,device='cpu')
        status['fine_prior_penalty_before']=model.policy.mlp_extractor.policy_net.fine_prior_penalty
        before={name:value.detach().clone() for name,value in model.policy.mlp_extractor.policy_net.state_dict().items()}
        model.policy_kwargs['fine_prior_penalty']=.2;model.policy.fine_prior_penalty=.2
        model.policy.mlp_extractor.policy_net.fine_prior_penalty=.2
        model.ent_coef=.015;model.set_random_seed(73123)
        assert all(torch.equal(v,model.policy.mlp_extractor.policy_net.state_dict()[k]) for k,v in before.items())
        status['checks']['parent_parameters_retained']=True
        obs,_=env.reset(seed=73123);mask=training.action_masks()
        actor=torch.jit.trace(PortableFineActor(model.policy).eval(),torch.from_numpy(obs)[None])
        action,_=model.predict(obs,action_masks=mask,deterministic=True)
        np.testing.assert_allclose(actor(torch.from_numpy(obs)[None])[0].detach().numpy(),ACTIONS[int(action)],atol=2e-8,rtol=0)
        model.save(args.output/'policy_initial.zip')
        reloaded=MaskablePPO.load(args.output/'policy_initial.zip',device='cpu')
        other,_=reloaded.predict(obs,action_masks=mask,deterministic=True);assert int(other)==int(action)
        assert reloaded.policy.mlp_extractor.policy_net.fine_prior_penalty==.2
        status['checks']['new_prior_reload_and_export_parity']=True
        # Check easy and full tasks under the same native scan/guard interface.
        for start,goal in [([4.3,2.,0.],[4.9,2.]),([1.4,2.,0.],[5.,2.]),([.9,2.,0.],[1.65,2.])]:
            o,_=training.reset(options=dict(start=start,goal=goal,case_seed=74123))
            choices=np.flatnonzero(training.action_masks()[1:])+1
            if not len(choices):raise RuntimeError('No nonzero action in preflight curriculum task')
            training.step(int(choices[0]))
        status['checks']['curriculum_nonzero_action_physics_parity']=True
        status.update(phase='training',training_started_unix=time.time());save()
        callback=Progress(status,save,args.output)
        for target in range(args.interval,args.steps+1,args.interval):
            status['phase']='training';save()
            model.learn(args.interval,reset_num_timesteps=(target==args.interval),callback=callback)
            model.save(args.output/f'policy_{target:06d}.zip')
            torch.jit.trace(PortableFineActor(model.policy).eval(),torch.from_numpy(obs)[None]).save(str(args.output/'actor_latest.pt'))
            status.update(phase='development_evaluation',evaluation_checkpoint=target);save()
            torch_rng=torch.get_rng_state();np_rng=np.random.get_state()
            try:
                if target==args.interval:
                    initial=MaskablePPO.load(args.output/'policy_initial.zip',device='cpu')
                    status['adapted_initial_cases']=evaluate(initial,scene,args.bundle,args.output/'evaluation_initial20')
                    teacher=MaskablePPO.load('models/recovery_20261012/candidate/policy.zip',device='cpu')
                    status['old12_cases']=evaluate(teacher,scene,args.bundle,args.output/'evaluation_old12',old=True)
                    del initial,teacher
                    save()
                cases=evaluate(model,scene,args.bundle,args.output/f'evaluation_{target:06d}')
            finally:torch.set_rng_state(torch_rng);np.random.set_state(np_rng)
            status['checkpoints'].append(dict(steps=target,cases=cases));save()
            print('EVAL',target,json.dumps(cases),flush=True)
            if not all(hashlib.sha256(Path(f).read_bytes()).hexdigest()==h for f,h in hashes.items()):raise RuntimeError('Runtime source changed during trial; stop attribution')
        model.save(args.output/'policy.zip')
        shutil.copy2(args.output/'actor_latest.pt',args.output/'actor.pt')
        portable=args.output/'portable';shutil.copytree(args.bundle,portable)
        shutil.copy2(args.output/'actor.pt',portable/'actor.pt')
        (portable/'fine_observation_action_contract.json').write_text(json.dumps(contract(),indent=2))
        manifest=json.loads((portable/'manifest.json').read_text());manifest.update(fine_interface=contract(),deployment_approved=False)
        for name in ['actor.pt','fine_observation_action_contract.json']:manifest['files'][name]=hashlib.sha256((portable/name).read_bytes()).hexdigest()
        (portable/'manifest.json').write_text(json.dumps(manifest,indent=2))
        status.update(status='complete',phase='finished',finished_unix=time.time(),source_hashes_unchanged=True,
            published_actor_unchanged=hashlib.sha256((args.bundle/'actor.pt').read_bytes()).hexdigest()==status['published_actor_sha256'])
    except BaseException as error:
        if model:model.save(args.output/'policy_interrupted.zip')
        status.update(status='failed',error=repr(error),finished_unix=time.time());raise
    finally:
        if env:env.close()
        save()

if __name__=='__main__':main()
