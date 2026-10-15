"""Bounded PPO recovery pilot with automatic paired development evaluation.

The default model is never replaced. Progress and terminal results survive the
interactive session. Evaluation trajectories are never inserted into training.
"""
import argparse
from collections import Counter
import gc
import hashlib
import json
from pathlib import Path
import shutil
import time

import numpy as np
import torch
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from experiments.quality_masked_navigation import QualityMaskedWrapper
from experiments.recovery_sampling import RecoveryCoverageProgress
from experiments.time_consistent_tracking import TimeConsistentCore, TimeConsistentFreshRuntime
from experiments.train_quality_navigation import export, digest
import experiments.evaluate_quality_regression as evaluation
from prototypes.balanced_recovery import BalancedRecoveryCurriculum, WaitReward
from prototypes.continued_recovery_policy import warm_start_continued


class BalancedProgress(RecoveryCoverageProgress):
    def __init__(self, output, steps, status, save):
        super().__init__(output, steps)
        self.status, self.save_status = status, save
        self.templates = Counter()
        self.wait_cost = 0.

    def _on_step(self):
        for info in self.locals['infos']:
            self.wait_cost += info['supported_wait_cost']
            if info['balanced_template_index'] is not None:
                self.templates[str(info['balanced_template_index'])] += 1
        if self.num_timesteps % 100 == 0:
            self.status.update(steps=self.num_timesteps, template_frame_counts=dict(self.templates),
                native_wait_cost_sum=self.wait_cost,
                training_elapsed_s=time.monotonic()-self.started)
            self.save_status()
            if shutil.disk_usage(self.output).free < 200*1024**2:
                self.model.save(self.output/'policy_low_disk.zip')
                raise RuntimeError('Low disk; checkpoint saved and experiment stopped')
        if self.num_timesteps % 500 == 0:
            print('TRAIN',self.num_timesteps,'/',self.steps,'templates',len(self.templates),flush=True)
        return super()._on_step()


def report_markdown(output, status):
    lines = ['# 均衡恢复 PPO 实验', '', f"状态：`{status['status']}`；新增训练步数：{status.get('steps',0)}。",
        '', '本实验不自动替换部署模型。训练只使用历史训练模板及随机普通课程，专项开发场景只用于评估。',
        '', '| 专项场景 | 当前发布模型 | 新训练模型 | 新模型用时 | 新模型距离 |',
        '|---|---|---|---:|---:|']
    for pair in status.get('paired_cases',[]):
        a,b=pair['initial'],pair['final']
        label=lambda x:'到达' if x['safe_success'] else '碰撞／保护' if x['collisions'] or x['physical_guards'] else '超时／未完成'
        lines.append(f"| {b['case_id']} | {label(a)} | {label(b)} | {b['seconds']:.1f} 秒 | {b['travel_m']:.3f} 米 |")
    lines += ['', '验收判定与限制见 `status.json`。专项场景是复用开发集，不是实车或独立部署验收。']
    (output/'REPORT.md').write_text('\n'.join(lines)+'\n')


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',type=Path,default=Path('models/recovery_20261012/candidate/policy.zip'))
    p.add_argument('--baseline-bundle',type=Path,default=Path('models/recovery_20261012/portable'))
    p.add_argument('--failure-pool',type=Path,default=Path('models/recovery_20261012/training_inputs/failure_pool.json'))
    p.add_argument('--reference',type=Path,default=Path('training/runs/safety_context_20261011/continued_recovery_10000_20261012_five'))
    p.add_argument('--ordinary-reference',type=Path,default=Path('training/runs/safety_context_20261011/continued_recovery_10000_20261012'))
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--steps',type=int,default=2500)
    p.add_argument('--seed',type=int,default=92613)
    args=p.parse_args()
    if args.output.exists() or args.steps<=0 or args.steps%250:
        p.error('Fresh output and positive multiple of 250 steps required')
    if shutil.disk_usage(args.output.parent).free<512*1024**2:
        p.error('At least 512 MiB free disk required')
    pool=json.loads(args.failure_pool.read_text())
    if pool.get('kind')!='training' or not pool.get('templates'):
        p.error('Training-only template pool required')
    torch.set_num_threads(1)
    args.output.mkdir()
    components=args.output/'candidate';components.mkdir()
    for name in ['hardware.json','localizer.pt','router.pt','quality_observation.json']:
        shutil.copy2(args.checkpoint.parent/name,components/name)
    source_names=['prototypes/balanced_recovery.py','prototypes/train_balanced_recovery.py',
        'prototypes/continued_recovery_policy.py','training/neural_navigation/core.py',
        'training/neural_navigation/route_efficiency.py']
    source={}
    for name in source_names:
        path=Path(name);source[name]=digest(path)
        destination=args.output/'source_snapshot'/name
        destination.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,destination)
    status=dict(status='running',phase='initializing',started_unix=time.time(),steps=0,
        requested_steps=args.steps,source_sha256=source,default_model_replaced=False,
        deployment_approved=False,paired_cases=[])
    def save():
        temp=args.output/'status.tmp';temp.write_text(json.dumps(status,indent=2));temp.replace(args.output/'status.json')
        report_markdown(args.output,status)
    save()
    config=dict(seed=args.seed,steps=args.steps,parent_policy_sha256=digest(args.checkpoint),
        failure_pool_sha256=digest(args.failure_pool),training_template_count=len(pool['templates']),
        training_schedule='alternate randomized historical-training templates and general tasks; shuffled full template coverage before reuse',
        gae_lambda=.98,ent_coef=.006,training_reward_scale=1/60,
        supported_wait_maximum_native=.02,supported_wait_budget_native_per_episode=2.,
        supported_wait_grace_s=2.,native_base_reward_unchanged=True,guard_unchanged=True,
        localizer_router_frozen=True,ordinary_actor_base_frozen=True,efficient_routes=False,
        evaluation_in_training=False,source_sha256=source,deployment_approved=False)
    (components/'config.json').write_text(json.dumps(config,indent=2))
    environment=None
    try:
        def create():
            base=BalancedRecoveryCurriculum(components,training_templates=pool['templates'],
                schedule_seed=args.seed,stage='robust',route_mode='shortcut',motion_profiles=True,
                footprint_clearance_cost=1.,core_factory=TimeConsistentCore)
            wrapped=WaitReward(QualityMaskedWrapper(base,reward_scale=1/60))
            return Monitor(wrapped,str(components/'worker_0.csv'),info_keywords=(
                'verified_mission_success','collision_events','safety_termination','training_case_json'))
        environment=DummyVecEnv([create]);environment.seed(args.seed)
        model=warm_start_continued(args.checkpoint,environment,seed=args.seed)
        model.gae_lambda=.98;model.rollout_buffer.gae_lambda=.98;model.ent_coef=.006
        model.save(components/'policy_initial.zip')
        export(components/'policy_initial.zip',components,args.output/'initial_portable','time-consistent')
        model.snapshot_reference()
        callback=BalancedProgress(components,args.steps,status,save)
        status['phase']='training';save()
        model.learn(args.steps,callback=callback,reset_num_timesteps=True)
        if callback.changed:
            raise ValueError('Nonzero sampled/executed action disagreement')
        model.save(components/'policy.zip')
        environment.close();environment=None;del model;gc.collect()
        export(components/'policy.zip',components,args.output/'portable','time-consistent')
        evaluation.QualitySensorNavigationRuntime=TimeConsistentFreshRuntime
        status.update(phase='paired_specialist_evaluation',steps=args.steps);save()
        names=['independent_dynamic_006','independent_dynamic_021','independent_dynamic_023',
               'motion_profile_005','train_failure_w11_e7']
        for name in names:
            case=json.loads((args.reference/f'{name}.json').read_text())['case']
            pair={}
            for label,bundle in [('initial',args.output/'initial_portable'),('final',args.output/'portable')]:
                trace=evaluation.evaluate(case,bundle)
                path=args.output/f'{label}_{name}.json';path.write_text(json.dumps(trace,allow_nan=False))
                pair[label]=dict(trace['result'],trace_sha256=digest(path))
                print('EVAL',label,json.dumps(trace['result']),flush=True);gc.collect()
            status['paired_cases'].append(pair);save()
        status.update(phase='ordinary_regression',ordinary_cases=[]);save()
        for index in range(1,9):
            record=json.loads((args.ordinary_reference/f'final_{index}.json').read_text())['case']
            from navrl.geometry import ROOT
            case=dict(record['interventions'],id=f'ordinary_pair_{index}',category='ordinary_development',
                scene=str(Path(record['scene']).relative_to(ROOT)),seed=record['seed'],**record['options'])
            pair={}
            for label,bundle in [('initial',args.output/'initial_portable'),('final',args.output/'portable')]:
                trace=evaluation.evaluate(case,bundle)
                path=args.output/f'{label}_ordinary_{index}.json';path.write_text(json.dumps(trace,allow_nan=False))
                pair[label]=dict(trace['result'],trace_sha256=digest(path))
                print('ORDINARY',label,index,json.dumps(trace['result']),flush=True);gc.collect()
            status['ordinary_cases'].append(pair);save()
        pairs=status['paired_cases'];ordinary=status['ordinary_cases']
        successes=lambda label:sum(x[label]['safe_success'] for x in pairs)
        failures=lambda label:sum(x[label]['collisions']+x[label]['physical_guards'] for x in pairs+ordinary)
        safe_preserved=all(not x['initial']['safe_success'] or x['final']['safe_success'] for x in pairs+ordinary)
        status.update(status='complete',phase='finished',finished_unix=time.time(),
            pilot_gate_passed=bool(successes('final')>successes('initial') and safe_preserved and failures('final')<=failures('initial')),
            initial_specialist_successes=successes('initial'),final_specialist_successes=successes('final'),
            ordinary_successes_preserved=safe_preserved,
            source_hashes_unchanged=all(digest(Path(name))==sha for name,sha in source.items()),
            recommendation='Continue only if pilot gate passes; otherwise diagnose recorded failures instead of automatically adding steps.')
        if not status['source_hashes_unchanged']:
            status['pilot_gate_passed']=False
            raise RuntimeError('Source changed during experiment; reject attribution')
    except BaseException as error:
        status.update(status='failed',error=repr(error),finished_unix=time.time())
        raise
    finally:
        if environment is not None:environment.close()
        save()


if __name__=='__main__':main()
