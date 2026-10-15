"""Bounded fine-action PPO pilot with parity gates and paired evaluation."""
import argparse,copy,hashlib,json,time,shutil
from pathlib import Path
import numpy as np
import torch
from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import BaseCallback
from prototypes.narrow_passage import generate
from prototypes.narrow_training_env import NarrowTrainingEnv
from prototypes.narrow_ppo import NarrowPolicy,PortableFineActor
from prototypes.narrow_observation import ACTIONS,contract

class Progress(BaseCallback):
    def __init__(self,status,save):super().__init__();self.status=status;self.save=save
    def _on_step(self):
        for i in self.locals['infos']:
            if i['nonzero_choice_changed_by_guard']:raise RuntimeError('Action disagreement')
            self.status['fine_training_choices']+=int(i['fine_choice'])
            if 'episode' in i:self.status['training_episodes']+=1
        if self.num_timesteps%25==0:
            self.status['steps']=self.num_timesteps;self.save()
        if self.num_timesteps%250==0:print('TRAIN',self.num_timesteps,flush=True)
        return True

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--steps',type=int,default=1000)
    p.add_argument('--bundle',type=Path,default=Path('models/recovery_20261012/portable'))
    p.add_argument('--teacher',type=Path,default=Path('models/recovery_20261012/candidate/policy.zip'));args=p.parse_args()
    if args.output.exists() or args.steps<250 or args.steps%250:p.error('Fresh output, positive 250-step multiple required')
    if shutil.disk_usage(args.output.parent).free<512*1024**2:p.error('Need 512 MiB free')
    torch.set_num_threads(1);args.output.mkdir(parents=True)
    hardware=json.loads((args.bundle/'hardware.json').read_text())
    source=[Path(x) for x in ['prototypes/narrow_alignment.py','prototypes/narrow_observation.py','prototypes/narrow_training_env.py','prototypes/narrow_ppo.py','prototypes/narrow_fine_runtime.py','prototypes/train_narrow_ppo.py','prototypes/narrow_passage.py','prototypes/narrow_runtime.py','prototypes/anchored_narrow_runtime.py','training/neural_navigation/runtime.py','training/neural_navigation/core.py']]
    repair_source=Path('prototypes/narrow_recovery_guard.py');source.append(repair_source)
    source.append(Path('prototypes/surface_localizer.py'))
    source.append(Path('prototypes/entrance_staging.py'))
    source.append(Path('prototypes/entrance_guard.py'))
    hashes={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in source}
    for f in source:
        dst=args.output/'source_snapshot'/f;dst.parent.mkdir(parents=True,exist_ok=True);dst.write_bytes(f.read_bytes())
    status=dict(status='running',phase='interface_preflight',steps=0,requested_steps=args.steps,started_unix=time.time(),
        fine_training_choices=0,training_episodes=0,checks={},paired_cases=[],source_sha256=hashes,
        teacher_sha256=hashlib.sha256(args.teacher.read_bytes()).hexdigest(),default_model_replaced=False,deployment_approved=False,
        optimizer_reset=True,critic_reset=True,frozen_teacher=True,
        observation_action_contract=contract(),runtime_adapter_repair=True,
        limitation='1000-step static straight-lane pilot, not broad corridor/junction/dynamic acceptance. Fine priors are interpolation, not pretrained maneuvers.')
    def save():
        temp=args.output/'status.tmp';temp.write_text(json.dumps(status,indent=2));temp.replace(args.output/'status.json')
    save();env=None
    try:
        scenes=generate(args.output/'scenes',hardware);scene=Path(scenes[0]['scene'])
        env=NarrowTrainingEnv(scene,args.bundle,seed=73120,max_steps=180)
        obs,_=env.reset(seed=73120);action_mask=env.action_masks()
        teacher=MaskablePPO.load(args.teacher,device='cpu')
        model=MaskablePPO(NarrowPolicy,env,device='cpu',seed=73120,n_steps=250,batch_size=50,n_epochs=4,
            learning_rate=1e-4,gamma=.99,gae_lambda=.95,ent_coef=.01,target_kl=.03,
            policy_kwargs=dict(teacher_checkpoint=str(args.teacher)))
        observations=torch.from_numpy(np.stack([obs,obs]));observations[1,308]=1
        with torch.no_grad():
            old=teacher.policy.action_net(teacher.policy.mlp_extractor.policy_net(observations[:,:300]))
            new=model.policy.mlp_extractor.forward_actor(observations)
            torch.testing.assert_close(new[:,:12],old,atol=0,rtol=0)
        status['checks']['old_12_raw_logits_exact']=True
        old_actor_hash=hashlib.sha256((args.bundle/'actor.pt').read_bytes()).hexdigest()
        legacy=torch.jit.load(str(args.bundle/'actor.pt'))
        with torch.no_grad():
            teacher_index,_=teacher.predict(obs[:300],action_masks=action_mask[:12],deterministic=True)
            np.testing.assert_allclose(legacy(torch.tensor(obs[:300])[None])[0].numpy(),ACTIONS[int(teacher_index)],atol=2e-8,rtol=0)
        status['checks']['teacher_matches_published_actor_on_reset']=True
        # Save/reload plus self-contained TorchScript export parity.
        model.save(args.output/'policy_initial.zip')
        reloaded=MaskablePPO.load(args.output/'policy_initial.zip',device='cpu')
        a,_=model.predict(obs,action_masks=action_mask,deterministic=True)
        b,_=reloaded.predict(obs,action_masks=action_mask,deterministic=True);assert int(a)==int(b)
        status['checks']['checkpoint_reload']=True
        actor=torch.jit.trace(PortableFineActor(model.policy).eval(),observations)
        np.testing.assert_allclose(actor(torch.tensor(obs)[None])[0].detach().numpy(),ACTIONS[int(a)],atol=2e-8,rtol=0)
        status['checks']['portable_actor_parity']=True
        # One admitted fine command must be sent unchanged through real physics.
        fine=np.flatnonzero(action_mask[12:])+12
        if not len(fine):raise RuntimeError('No supported fine command at preflight start')
        _,_,_,_,info=env.step(int(fine[0]));assert not info['nonzero_choice_changed_by_guard']
        status['checks']['fine_command_physics_parity']=True
        env.reset(seed=73120)
        status['phase']='training';save();started=time.monotonic()
        from stable_baselines3.common.monitor import Monitor
        model.set_env(Monitor(env,str(args.output/'monitor.csv')))
        model.learn(args.steps,callback=Progress(status,save))
        status['training_elapsed_s']=time.monotonic()-started
        model.save(args.output/'policy.zip')
        portable=PortableFineActor(model.policy).eval()
        torch.jit.trace(portable,observations).save(str(args.output/'actor.pt'))
        (args.output/'observation_action_contract.json').write_text(json.dumps(contract(),indent=2))
        (args.output/'hardware.json').write_text(json.dumps(hardware,indent=2))
        portable_dir=args.output/'portable';shutil.copytree(args.bundle,portable_dir)
        shutil.copy2(args.output/'actor.pt',portable_dir/'actor.pt')
        (portable_dir/'fine_observation_action_contract.json').write_text(json.dumps(contract(),indent=2))
        manifest=json.loads((portable_dir/'manifest.json').read_text())
        manifest['fine_interface']=contract();manifest['deployment_approved']=False
        manifest['files']['actor.pt']=hashlib.sha256((portable_dir/'actor.pt').read_bytes()).hexdigest()
        manifest['files']['fine_observation_action_contract.json']=hashlib.sha256((portable_dir/'fine_observation_action_contract.json').read_bytes()).hexdigest()
        (portable_dir/'manifest.json').write_text(json.dumps(manifest,indent=2))
        from prototypes.narrow_fine_runtime import FineSensorRuntime
        exported=FineSensorRuntime(portable_dir,scene/'decision_map.yaml',env.env.truth.tolist(),env.env.goal.tolist(),'shortcut')
        result=exported.step(env.env.scan,env.env.observed_odometry,env.env.measured_velocity,env.env.elapsed,capture_observation=True)
        o=np.asarray(result['policy_observation'],np.float32)
        index,_=model.predict(o,action_masks=__import__('prototypes.narrow_observation',fromlist=['masks']).masks(o),deterministic=True)
        assert int(index)==result['fine_choice']
        status['checks']['exported_sensor_runtime_parity']=True
        env.close();env=None
        status.update(phase='paired_development_evaluation',steps=args.steps);save()
        initial=MaskablePPO.load(args.output/'policy_initial.zip',device='cpu')
        cases=[dict(id='centered',seed=83120,start=[1.4,2.,0.],goal=[5.,2.]),
               dict(id='offset_left',seed=83121,start=[1.5,2.015,.06],goal=[5.,2.]),
               dict(id='offset_right',seed=83122,start=[1.5,1.985,-.06],goal=[5.,2.])]
        for case in cases:
            pair=dict(case=case)
            for name,policy in [('old12',teacher),('initial20',initial),('trained20',model)]:
                trial=NarrowTrainingEnv(scene,args.bundle,seed=case['seed'],max_steps=250)
                rows=[];done=truncated=False
                try:
                    o,_=trial.reset(options=dict(start=case['start'],goal=case['goal'],case_seed=case['seed']))
                    for _ in range(250):
                        mask=trial.action_masks();index,_=policy.predict(o[:300] if name=='old12' else o,
                            action_masks=mask[:12] if name=='old12' else mask,deterministic=True)
                        o,r,done,truncated,info=trial.step(int(index))
                        rows.append(dict(time=float(trial.env.elapsed),choice=int(index),command=info['commanded_velocity'],estimate=trial.runtime.core.estimate.tolist()))
                        if done or truncated:break
                    result=dict(info,frames=len(rows),limit_reached=truncated,fine_choices=sum(x['choice']>=12 for x in rows))
                    pair[name]=result
                    (args.output/f"{case['id']}_{name}.json").write_text(json.dumps(dict(case=case,samples=rows,result=result)))
                finally:trial.close()
            status['paired_cases'].append(pair);save();print('PAIR',json.dumps(pair),flush=True)
        old_success=sum(x['old12']['verified_mission_success'] for x in status['paired_cases'])
        new_success=sum(x['trained20']['verified_mission_success'] for x in status['paired_cases'])
        status.update(status='complete',phase='finished',finished_unix=time.time(),old_successes=old_success,new_successes=new_success,
            pilot_improved=new_success>old_success,
            source_hashes_unchanged=all(hashlib.sha256(Path(f).read_bytes()).hexdigest()==h for f,h in hashes.items()),
            published_actor_unchanged=hashlib.sha256((args.bundle/'actor.pt').read_bytes()).hexdigest()==old_actor_hash)
    except BaseException as error:
        status.update(status='failed',error=repr(error),finished_unix=time.time());raise
    finally:
        if env:env.close()
        save()

if __name__=='__main__':main()
