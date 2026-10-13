"""Isolated conservative PPO experiment: safety feedback + renewed exploration."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from remote.train_parallel import Progress
from neural_navigation.context_curriculum import ContextNavigationCurriculum
from neural_navigation.context_policy import warm_start
from neural_navigation.safety_context import contract,OBSERVATION_DIM,validate_contract


def make_context_env(rank, components, failures):
    def create():
        torch.set_num_threads(1)
        env=ContextNavigationCurriculum(components,stage='robust',route_mode='shortcut',
            motion_profiles=True,footprint_clearance_cost=1.,failure_templates=failures)
        return Monitor(env,str(components/f'worker_{rank}.csv'),info_keywords=(
            'mission_targets_reached','verified_mission_success','collision_events','safety_termination',
            'training_case_json','execution_guard_overrides','execution_guard_steps','execution_guard_override_fraction'))
    return create


class ContextProgress(Progress):
    def _on_rollout_start(self):
        # Before collecting the next rollout, never between sampling and PPO's
        # likelihood update. Keep a consistent policy for each rollout.
        with torch.no_grad():
            self.model.policy.log_std.clamp_(float(np.log(.20)), float(np.log(.50)))
        super()._on_rollout_start()

    def _on_step(self):
        result = super()._on_step()
        if self.num_timesteps % 1000 < self.training_env.num_envs:
            p = self.output/'progress.json'
            data = json.loads(p.read_text())
            data['exploration_std'] = self.model.policy.log_std.detach().exp().tolist()
            data['context_schema'] = contract()['schema']
            p.write_text(json.dumps(data, indent=2))
        return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--parent', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--steps', type=int, default=100000)
    p.add_argument('--envs', type=int, default=4)
    p.add_argument('--seed', type=int, default=92001)
    p.add_argument('--failure-pool',type=Path)
    p.add_argument('--resume-policy',type=Path)
    a = p.parse_args()
    if not 1 <= a.envs <= 4 or a.steps <= 0:
        p.error('Use 1-4 workers and positive steps')
    if a.output.exists():
        p.error('Fresh output required; preserve earlier results')
    a.output = a.output.resolve()
    a.parent = a.parent.resolve()
    a.output.mkdir(parents=True)
    for name in ['localizer.pt', 'router.pt', 'hardware.json']:
        shutil.copy2(a.parent/name, a.output/name)
    hardware = json.loads((a.output/'hardware.json').read_text())
    hardware['policy_observation'] = contract()
    (a.output/'hardware.json').write_text(json.dumps(hardware, indent=2))
    torch.set_num_threads(1)
    failures=[]
    if a.failure_pool:
        pool=json.loads(a.failure_pool.read_text())
        if pool.get('kind')!='training' or not pool.get('templates'):p.error('Nonempty training-only failure pool required')
        failures=pool['templates']
        shutil.copy2(a.failure_pool,a.output/'failure_pool.json')
    env = SubprocVecEnv([make_context_env(i,a.output,failures) for i in range(a.envs)], start_method='spawn')
    env.seed(a.seed)
    config = dict(status='initializing', parent_policy_sha256=hashlib.sha256((a.parent/'policy.zip').read_bytes()).hexdigest(),
                  requested_steps=a.steps, num_envs=a.envs, device='cpu', seed=a.seed,
                  policy_observation=contract(), deployment_approved=False,
                  reward_unchanged_from_guarded_clearance=True,
                  training_curriculum='robust, retains static and slow one-way/reversal/stop-go dynamic tasks',
                  localizer_router_frozen=True, guard_chooses_navigation_actions=False,robot_action_overrides=True,
                  learning_rate=3e-5,ent_coef=.003,n_steps=250,batch_size=250,n_epochs=4,target_kl=.02,
                  failure_template_fraction=.4 if failures else 0.,failure_template_count=len(failures),
                  checkpoint_every=2000,
                  exploration_std_floor=.20, exploration_std_ceiling=.50,
                  optimizer_reset_reason='input dimensions changed; new PPO optimizer, existing actor/critic weights retained')
    project = Path(__file__).resolve().parents[1]
    sources = sorted((project/'training').rglob('*.py'))
    sources = [s for s in sources if 'runs' not in s.relative_to(project).parts]
    config['launch_source_sha256'] = {}
    for source in sources:
        name = source.relative_to(project)
        config['launch_source_sha256'][str(name)] = hashlib.sha256(source.read_bytes()).hexdigest()
        target = a.output/'code_snapshot'/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    try:
        if a.resume_policy:
            old_hardware=json.loads((a.resume_policy.parent/'hardware.json').read_text())
            old_contract=old_hardware.pop('policy_observation',None)
            validate_contract(old_contract)
            if old_hardware!={k:v for k,v in hardware.items() if k!='policy_observation'}:raise ValueError('Resume policy hardware contract mismatch')
            for name in ['localizer.pt','router.pt']:
                if hashlib.sha256((a.resume_policy.parent/name).read_bytes()).digest()!=hashlib.sha256((a.output/name).read_bytes()).digest():
                    raise ValueError('Resume learned components differ')
            model=PPO.load(a.resume_policy,env=env,device='cpu',seed=a.seed,
                n_steps=250,batch_size=250,n_epochs=4,learning_rate=3e-5,target_kl=.02,ent_coef=.003)
            if model.observation_space.shape!=(OBSERVATION_DIM,):raise ValueError('Safety-context resume policy required')
            config['warm_start']=dict(resume_policy=str(a.resume_policy.resolve()),
                resume_policy_sha256=hashlib.sha256(a.resume_policy.read_bytes()).hexdigest(),
                prior_checkpoint_steps=model.num_timesteps,optimizer_retained=True,
                observation_schema_migration=[old_contract['schema'],hardware['policy_observation']['schema']],
                exact_rollout_or_rng_resume=False)
            config['optimizer_reset_reason']=None
        else:
            model, config['warm_start'] = warm_start(a.parent/'policy.zip', env, a.seed)
        model.save(a.output/'policy_initial.zip')
        config['status'] = 'running'
        (a.output/'config.json').write_text(json.dumps(config, indent=2))
        callback = ContextProgress(a.output, None, None, None, a.steps, 2000, 0)
        start = time.monotonic()
        model.learn(total_timesteps=a.steps, callback=callback, reset_num_timesteps=True)
        model.save(a.output/'policy.zip')
        config.update(status='complete', additional_ppo_steps_completed=model.num_timesteps,
                      wall_seconds=time.monotonic()-start)
        (a.output/'config.json').write_text(json.dumps(config, indent=2))
        print('COMPLETE', json.dumps(config), flush=True)
    finally:
        env.close()


if __name__ == '__main__':
    main()
