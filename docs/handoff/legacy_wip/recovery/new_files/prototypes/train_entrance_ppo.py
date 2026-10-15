"""Bounded, single-thread PPO continuation and paired 25-case final evaluation."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil
import time
import numpy as np
import torch
from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from prototypes.entrance_ppo_curriculum import EntrancePPOCurriculum, fixed_cases
from prototypes.narrow_training_env import NarrowTrainingEnv
from prototypes.narrow_ppo import PortableFineActor
from prototypes.narrow_observation import ACTIONS, contract
from prototypes.narrow_passage import generate


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def parameter_sha(net):
    h = hashlib.sha256()
    for key, value in net.state_dict().items():
        h.update(key.encode()); h.update(value.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def summarize(cases):
    result = {}
    for group in ('all', 'full_passage', 'short_exit', 'open_room', 'heldout'):
        selected = [c for c in cases if group == 'all' or c['category'] == group or
                    (group == 'heldout' and c['split'] == group)]
        result[group] = dict(total=len(selected), reached=sum(c['verified_mission_success'] for c in selected),
                             collision_or_guard=sum(bool(c['collision_events'] or c['physical_guards']) for c in selected),
                             timeouts=sum(c['timeout'] for c in selected))
    return result


def evaluate(model, env, cases, output, status, save, check_sources):
    output.mkdir()
    results = []
    for case in cases:
        check_sources()
        obs, _ = env.reset(options=dict(start=case['start'], goal=case['goal'], case_seed=case['seed']))
        exported = torch.jit.trace(PortableFineActor(model.policy).eval(), torch.from_numpy(obs)[None])
        rows = []; counts = Counter(); started = time.time()
        for frame in range(env.max_steps):
            if frame % 25 == 0:
                status.update(current_case=case['id'], evaluation_frame=frame); save()
            mask = env.action_masks()
            choice, _ = model.predict(obs, action_masks=mask, deterministic=True)
            choice = int(choice)
            with torch.no_grad(): command = exported(torch.from_numpy(obs)[None])[0].numpy()
            np.testing.assert_allclose(command, ACTIONS[choice], atol=2e-8, rtol=0)
            counts['only_stop'] += int(not mask[1:].any())
            counts['stop_with_alternatives'] += int(choice == 0 and mask[1:].any())
            obs, _, done, truncated, info = env.step(choice)
            core = env.runtime.core
            rows.append(dict(frame=frame, choice=choice, command=info['commanded_velocity'],
                             pose=core.estimate.tolist(), truth_for_scoring_only=env.env.truth.tolist(),
                             pose_error_m=float(np.linalg.norm(core.estimate[:2]-env.env.truth[:2])),
                             route_state=core.narrow_route_state, legal_choices=np.flatnonzero(mask).tolist()))
            if done or truncated: break
        result = dict(info, case_id=case['id'], category=case['category'], split=case['split'],
                      frames=len(rows), timeout=bool(truncated), counts=dict(counts),
                      maximum_pose_error_m=max(r['pose_error_m'] for r in rows), wall_seconds=time.time()-started)
        (output/f"{case['id']}.json").write_text(json.dumps(dict(case=case, result=result, samples=rows), indent=2))
        results.append(result)
        status['evaluation_completed_cases'] = len(results); save()
        print('EVAL', output.name, case['id'], result['verified_mission_success'], flush=True)
    (output/'summary.json').write_text(json.dumps(summarize(results), indent=2))
    return results


class Progress(BaseCallback):
    def __init__(self, status, save, output):
        super().__init__(); self.status=status; self.save=save; self.output=output

    def _on_rollout_start(self):
        # Called after the preceding rollout's optimizer update, so progress
        # can establish real learning without waiting for a 5k checkpoint.
        if self.model._n_updates > self.status['parent_ppo_epoch_updates']:
            net = self.model.policy.mlp_extractor.policy_net
            changed = parameter_sha(net.residual) != self.status['initial_residual_sha256']
            frozen = parameter_sha(net.teacher) == self.status['initial_teacher_sha256']
            if not changed or not frozen:
                raise RuntimeError('Optimizer did not update residual or changed frozen teacher')
            self.status['checks']['first_optimizer_update_verified'] = True
            self.status['additional_ppo_epoch_updates'] = self.model._n_updates-self.status['parent_ppo_epoch_updates']
            self.save()

    def _on_step(self):
        for info in self.locals['infos']:
            if info['nonzero_choice_changed_by_guard']:
                raise RuntimeError('Selected/executed action mismatch')
            stage = info['training_stage']
            row = self.status['stage_counts'].setdefault(stage, dict(frames=0, only_stop_frames=0,
                episodes=0, milestones=0, mission_successes=0, deadlocks=0, collision_episodes=0))
            row['frames'] += 1; row['only_stop_frames'] += int(info['only_stop_before_step'])
            if 'episode' in info:
                row['episodes'] += 1
                row['milestones'] += int(info['verified_training_milestone'])
                row['mission_successes'] += int(info['verified_mission_success'])
                row['deadlocks'] += int(info['training_deadlock_truncation'])
                row['collision_episodes'] += int(bool(info['collision_events'] or info['physical_guards']))
        self.status.update(steps=self.num_timesteps, ppo_epoch_updates=self.model._n_updates,
                           train_wall_seconds=time.time()-self.status['training_started_unix'])
        if self.num_timesteps % 100 == 0:
            self.save()
            if shutil.disk_usage(self.output).free < 1024**3:
                raise RuntimeError('Less than 1 GiB disk free; preserving partial checkpoint')
        if self.num_timesteps % 1000 == 0:
            print('TRAIN', self.num_timesteps, 'updates', self.model._n_updates, flush=True)
        return True


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=20000)
    parser.add_argument('--interval', type=int, default=5000)
    parser.add_argument('--checkpoint', type=Path, default=Path('training/runs/narrow_long_50000_v2_20261013/policy.zip'))
    parser.add_argument('--bundle', type=Path, default=Path('models/recovery_20261012/portable'))
    args=parser.parse_args()
    if args.output.exists() or args.steps <= 0 or args.interval <= 0 or args.interval % 250 or args.steps % args.interval:
        parser.error('Fresh output and positive step/interval multiples of 250 required')
    if shutil.disk_usage(args.output.parent).free < 2*1024**3:
        parser.error('Need at least 2 GiB free')
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    args.output.mkdir(parents=True)
    paths = sorted(set(list(Path('prototypes').glob('*.py')) + list(Path('training/neural_navigation').glob('*.py')) +
                       list(Path('training/experiments').glob('*.py')) + list(Path('training/navrl').glob('*.py'))))
    hashes = {str(path):sha(path) for path in paths}
    for path in paths:
        dest=args.output/'source_snapshot'/path; dest.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(path, dest)
    cases=fixed_cases(); (args.output/'fixed_cases.json').write_text(json.dumps(cases, indent=2))
    status=dict(status='running', phase='preflight', steps=0, requested_steps=args.steps,
        checkpoint_interval=args.interval, started_unix=time.time(), stage_counts={}, checkpoints=[], checks={},
        source_sha256=hashes, parent_checkpoint_sha256=sha(args.checkpoint),
        published_actor_sha256=sha(args.bundle/'actor.pt'), deployment_approved=False, old_model_replaced=False,
        evaluation_cases=25, heldout_cases=20, development_cases=5, seed=74140,
        reward_contract='native straight-lane reward + bounded pose alignment potential + .5 estimated milestone; milestone is not mission success',
        resources=dict(cpu_cores=.5, torch_threads=1, memory_max_gib=4),
        limitation='Static 15cm straight passage only. Repeated development cases and 20 final holdouts; no junction/dynamic/10cm/real-robot acceptance.',
        observation_action_contract=contract(), optimizer_state_retained=True, critic_retained=True)

    def save():
        status['updated_unix']=time.time()
        temp=args.output/'status.tmp'; temp.write_text(json.dumps(status, indent=2)); temp.replace(args.output/'status.json')
        lines=['# 入口专项 PPO 增量训练', '', f"状态：{status['status']} / {status['phase']}；新增 {status['steps']:,}/{args.steps:,} 步。", '',
               '原部署模型保留。入口课程里程碑单独统计，不等同于完整窄道成功。', '',
               '约 0.5 CPU 核，单线程，内存上限 4 GiB；每 5,000 步检查，最终旧新候选分别运行同一组 25 场景。', '',
               '| 检查点 | 完整窄道 | 出口/空旷 | 碰撞或物理保护 |', '|---|---|---|---|']
        for item in status['checkpoints']:
            summary=item['summary']; lines.append(f"| {item['steps']} | {summary['full_passage']['reached']}/3 | {summary['short_exit']['reached']+summary['open_room']['reached']}/2 | {summary['all']['collision_or_guard']} |")
        if 'final_comparison' in status:
            lines += ['', '## 固定 25 场景对比', '', '| 模型 | 总到达 | 完整窄道 | 碰撞或保护 |', '|---|---|---|---|']
            for label, summary in status['final_comparison'].items():
                lines.append(f"| {label} | {summary['all']['reached']}/25 | {summary['full_passage']['reached']}/13 | {summary['all']['collision_or_guard']} |")
        (args.output/'REPORT.md').write_text('\n'.join(lines)+'\n')

    def check_sources():
        if any(sha(path) != expected for path, expected in hashes.items()):
            raise RuntimeError('Source changed during training/evaluation; stop attribution')

    env=None; evaluation_env=None; model=None
    save()
    try:
        scene=Path(generate(args.output/'scenes', json.loads((args.bundle/'hardware.json').read_text()))[0]['scene'])
        training=EntrancePPOCurriculum(scene, args.bundle, seed=74140, max_steps=250)
        env=Monitor(training, str(args.output/'monitor.csv'), info_keywords=('verified_mission_success', 'training_stage', 'verified_training_milestone'))
        model=MaskablePPO.load(args.checkpoint, env=env, device='cpu')
        if model.n_steps != 250 or model.n_epochs != 4:
            raise RuntimeError('Unexpected parent rollout/optimizer contract')
        model.ent_coef=.015; model.set_random_seed(74140)
        status['parent_ppo_epoch_updates']=model._n_updates
        status['initial_residual_sha256']=parameter_sha(model.policy.mlp_extractor.policy_net.residual)
        status['initial_teacher_sha256']=parameter_sha(model.policy.mlp_extractor.policy_net.teacher)
        # Retain original checkpoint and optimization settings, not a new actor.
        model.save(args.output/'policy_initial.zip')
        for start, goal in [([1.35, 2., 0.], [5., 2.]), ([1.65, 2., 0.], [5., 2.]), ([.9, 2., 0.], [1.65, 2.])]:
            obs, _=training.reset(options=dict(start=start, goal=goal, case_seed=74141))
            choices=np.flatnonzero(training.action_masks()[1:])+1
            if not len(choices): raise RuntimeError('No legal movement in required entrance preflight')
            training.step(int(choices[0]))
        status['checks']['native_preflight_action_parity']=True
        status.update(phase='training', training_started_unix=time.time()); save()
        callback=Progress(status, save, args.output)
        # Reuse expensive native physics/map loading throughout evaluations.
        for target in range(args.interval, args.steps+1, args.interval):
            check_sources(); status['phase']='training'; save()
            model.learn(args.interval, reset_num_timesteps=(target==args.interval), callback=callback)
            status['steps']=model.num_timesteps; status['ppo_epoch_updates']=model._n_updates
            status['residual_sha256']=parameter_sha(model.policy.mlp_extractor.policy_net.residual)
            status['checks']['residual_weights_updated']=status['residual_sha256']!=status['initial_residual_sha256']
            status['checks']['frozen_teacher_unchanged']=parameter_sha(model.policy.mlp_extractor.policy_net.teacher)==status['initial_teacher_sha256']
            if not all(status['checks'].values()): raise RuntimeError('Trainable/frozen parameter contract failed')
            model.save(args.output/f'policy_{target:06d}.zip')
            model.save(args.output/'policy.zip')
            if evaluation_env is None: evaluation_env=NarrowTrainingEnv(scene, args.bundle, max_steps=250)
            status.update(phase='development_evaluation', evaluation_completed_cases=0); save()
            torch_rng=torch.get_rng_state(); numpy_rng=np.random.get_state()
            try:
                results=evaluate(model, evaluation_env, cases[:5], args.output/f'evaluation_{target:06d}', status, save, check_sources)
            finally:
                torch.set_rng_state(torch_rng); np.random.set_state(numpy_rng)
            status['checkpoints'].append(dict(steps=target, summary=summarize(results)))
            save()
            if any(c['collision_events'] or c['physical_guards'] for c in results):
                status['stop_reason']='Development collision/physical guard; stop added training and evaluate saved candidate'
                break
        status.update(phase='final_candidate_25', evaluation_completed_cases=0); save()
        candidate=evaluate(model, evaluation_env, cases, args.output/'evaluation_final', status, save, check_sources)
        status.update(phase='final_parent_25', evaluation_completed_cases=0); save()
        parent=MaskablePPO.load(args.output/'policy_initial.zip', device='cpu')
        baseline=evaluate(parent, evaluation_env, cases, args.output/'evaluation_parent', status, save, check_sources)
        status['final_comparison']=dict(parent=summarize(baseline), candidate=summarize(candidate))
        status['checks'].update(export_parity_every_evaluation_frame=True, selected_executed_action_parity=True)
        obs, _=evaluation_env.reset(options=dict(start=cases[0]['start'], goal=cases[0]['goal'], case_seed=cases[0]['seed']))
        torch.jit.trace(PortableFineActor(model.policy).eval(), torch.from_numpy(obs)[None]).save(str(args.output/'actor.pt'))
        (args.output/'observation_action_contract.json').write_text(json.dumps(contract(), indent=2))
        check_sources()
        status.update(status='complete', phase='finished', finished_unix=time.time(), source_hashes_unchanged=True,
                      published_actor_unchanged=sha(args.bundle/'actor.pt')==status['published_actor_sha256'])
    except BaseException as error:
        if model is not None: model.save(args.output/'policy_interrupted.zip')
        status.update(status='failed', phase='failed', error=repr(error), finished_unix=time.time())
        raise
    finally:
        if env is not None: env.close()
        if evaluation_env is not None: evaluation_env.close()
        save()


if __name__ == '__main__':
    main()
