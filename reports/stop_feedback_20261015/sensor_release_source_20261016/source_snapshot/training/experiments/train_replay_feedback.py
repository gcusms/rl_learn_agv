"""Isolated on-policy PPO with validated sensor-feasibility auxiliary replay."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import torch
from stable_baselines3.common.vec_env import SubprocVecEnv
from experiments.replay_feedback_ppo import ReplayFeedbackPPO
from experiments.train_recovery_reward import environment, RecoveryProgress
from neural_navigation.safety_context import RECOVERY_SCHEMA, validate_contract


class ReplayProgress(RecoveryProgress):
    def _on_step(self):
        result = super()._on_step()
        if self.num_timesteps % 1000 < self.training_env.num_envs:
            path = self.output/'progress.json'
            data = json.loads(path.read_text())
            data['auxiliary_replay'] = self.model.last_replay_statistics
            path.write_text(json.dumps(data, indent=2))
        return result


def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--feedback', type=Path, required=True)
    p.add_argument('--failure-pool', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--steps', type=int, default=12000)
    p.add_argument('--seed', type=int, default=92102)
    p.add_argument('--manifest', type=Path, action='append', required=True)
    args = p.parse_args()
    if args.output.exists() or args.steps <= 0: p.error('Fresh output and positive training steps required')
    args.checkpoint = args.checkpoint.resolve(); args.output = args.output.resolve()
    args.feedback = args.feedback.resolve()
    components = args.checkpoint.parent
    hardware = json.loads((components/'hardware.json').read_text())
    validate_contract(hardware['policy_observation'])
    if hardware['policy_observation']['schema'] != RECOVERY_SCHEMA: p.error('V3 warm start required')
    pool = json.loads(args.failure_pool.read_text())
    if pool.get('kind') != 'training' or not pool.get('templates'): p.error('Training-only templates required')
    # Refuse empty/incomplete data before starting a simulation subprocess.
    meta = json.loads((args.feedback/'status.json').read_text())
    if meta.get('status') != 'complete' or meta.get('sample_count', 0) <= 0:
        p.error('Completed, nonempty feedback collection required')
    args.output.mkdir(parents=True); candidate = args.output/'candidate'; candidate.mkdir()
    for name in ('hardware.json', 'localizer.pt', 'router.pt'):
        shutil.copy2(components/name, candidate/name)
    shutil.copy2(args.failure_pool, candidate/'failure_pool.json')
    shutil.copytree(args.feedback, candidate/'feedback_collection')
    project = Path(__file__).resolve().parents[2]
    config = dict(status='initializing', requested_steps=args.steps, seed=args.seed, num_envs=1,
        parent_checkpoint=str(args.checkpoint), parent_checkpoint_sha256=digest(args.checkpoint),
        policy_observation=hardware['policy_observation'], parent_probe_schema=RECOVERY_SCHEMA,
        optimizer_retained=True, exact_rollout_or_rng_resume=False, guard_unchanged=True,
        physical_constraints_unchanged=True, localizer_router_frozen=True,
        footprint_clearance_cost=1.,
        repeated_rejection_cost=dict(grace_s=1., maximum_per_step=.2, ramp_s=2.,
            only_rejected_translation=True, require_accepted_probe=True, exempt_voluntary_wait=True),
        failure_template_fraction=.4, reference_mean_anchor=dict(enabled=False),
        static_deadlock_terminal_cost=None, deployment_approved=False, final_acceptance_used=False,
        replay_feedback=dict(strength=.1, replay_loss_multiplier=.5, maximum_replay_batch=64,
            thinning_frames=5, sample_episode_uniformly=True, source=str(args.feedback),
            data_meta_sha256=digest(args.feedback/'status.json'),
            replay_ppo_returns=False, replay_ppo_log_probabilities=False,
            policy_inputs_sensor_only=True, future_safety_guaranteed=False), launch_source_sha256={})
    for source in sorted((project/'training').rglob('*.py')):
        relative = source.relative_to(project)
        if 'runs' in relative.parts: continue
        config['launch_source_sha256'][str(relative)] = digest(source)
        target = candidate/'code_snapshot'/relative
        target.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(source, target)
    status = dict(status='running', phase='initializing', deployment_approved=False, started_unix=time.time())
    def save():
        (candidate/'config.json').write_text(json.dumps(config, indent=2))
        (args.output/'status.json').write_text(json.dumps(status, indent=2))
    save(); torch.set_num_threads(1); env = None
    try:
        model = ReplayFeedbackPPO.load(args.checkpoint, device='cpu', seed=args.seed,
            n_steps=250, batch_size=250, n_epochs=4, learning_rate=3e-5,
            target_kl=.02, ent_coef=.003, anchor_strength=.1, anchor_scale=.3)
        config['replay_feedback']['validated_collection'] = model.load_feedback(
            candidate/'feedback_collection', seed=args.seed+118)
        env = SubprocVecEnv([environment(candidate, pool['templates'])], start_method='spawn')
        env.seed(args.seed); model.set_env(env)
        config['prior_checkpoint_steps'] = model.num_timesteps
        model.save(candidate/'policy_initial.zip')
        config['status'] = 'running'; status['phase'] = 'training'; save()
        model.learn(total_timesteps=args.steps, callback=ReplayProgress(candidate, None, None, None,
            args.steps, 2000, 0, schema=RECOVERY_SCHEMA), reset_num_timesteps=True)
        model.save(candidate/'policy.zip'); env.close(); env = None
        config.update(status='complete', additional_ppo_steps_completed=model.num_timesteps)
        status['phase'] = 'export'; save()
        portable = args.output/'portable'
        with (args.output/'export.log').open('w') as log:
            subprocess.run([sys.executable, 'training/export_navigation_actor.py', '--policy',
                str(candidate/'policy.zip'), '--components', str(candidate), '--output', str(portable)],
                cwd=project, stdout=log, stderr=subprocess.STDOUT, check=True)
        summaries = []
        for index, manifest in enumerate(args.manifest):
            status['phase'] = f'runtime_{index}'; save()
            with (args.output/f'runtime_{index}.log').open('w') as log:
                subprocess.run([sys.executable, 'training/evaluate_sensor_navigation.py', '--bundle',
                    str(portable), '--manifest', str(manifest.resolve()), '--workers', '1',
                    '--output', str(args.output/f'runtime_{index}.json'), '--trace-dir',
                    str(args.output/f'traces_{index}')], cwd=project, stdout=log, stderr=subprocess.STDOUT, check=True)
            report = json.loads((args.output/f'runtime_{index}.json').read_text())
            summaries.append({key: report[key] for key in ('completed', 'safe_success', 'tasks', 'collisions')})
        status.update(status='complete', summaries=summaries,
            development_gate_passed=all(x['completed'] and x['safe_success'] == x['tasks'] for x in summaries),
            finished_unix=time.time()); save()
    except Exception as error:
        status.update(status='failed', error=str(error)); save(); raise
    finally:
        if env is not None: env.close()


if __name__ == '__main__': main()
