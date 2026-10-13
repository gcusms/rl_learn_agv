"""Checkpoint recovery for one authorized pilot, under a process-wide lock.

New attempts keep interrupted logs intact. This restores policy/optimizer
weights, not an in-flight rollout, environment state or exact RNG sequence.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile


def save(path,data):
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data,indent=2));os.replace(temporary,path)


def checkpoint(candidate):
    choices=sorted((p for p in candidate.glob('policy_*.zip') if p.stem.split('_')[-1].isdigit()),key=lambda p:int(p.stem.split('_')[-1]),reverse=True)
    for path in choices:
        try:
            with zipfile.ZipFile(path) as archive:
                if archive.testzip() is not None:continue
                steps=int(json.loads(archive.read('data'))['num_timesteps'])
                if steps!=int(path.stem.split('_')[-1]):continue
            return path,steps
        except (OSError,ValueError,KeyError,zipfile.BadZipFile):continue
    return None,0


def verify_source(candidate,project):
    config=json.loads((candidate/'config.json').read_text())
    for name,expected in config['launch_source_sha256'].items():
        # Orchestration can gain restart support, but inference, physics and
        # rewards must remain the same throughout this training lineage.
        if name=='training/run_optimization_pilot.py':continue
        # Explicitly authorize this resource-only supervisor revision. Keep
        # all simulation, reward and policy source hashes pinned.
        if (name=='training/recover_optimization_pilot.py' and
                expected==os.environ.get('AGV_PREVIOUS_RECOVERY_SHA256')):continue
        source=project/name
        if not source.is_file() or hashlib.sha256(source.read_bytes()).hexdigest()!=expected:
            raise RuntimeError(f'Training source changed: {name}; refusing automatic continuation')


def pilot_arguments(root,parent,output,steps,seed):
    baseline=root.parent/'development_forecast_directional.json'
    status=json.loads((root/'status.json').read_text())
    envs=int(os.environ.get('AGV_TRAINING_ENVS','12'))
    if not 1<=envs<=12:raise ValueError('AGV_TRAINING_ENVS must be between 1 and 12')
    result=['--parent',str(parent),'--baseline',str(baseline),'--output',str(output),'--steps',str(steps),
            '--envs',str(envs),'--seed',str(seed),'--stage',status['stage'],'--execution-mode',status['execution_mode']]
    if status.get('motion_profiles'):result.append('--motion-profiles')
    result+=['--footprint-clearance-cost',str(status.get('footprint_clearance_cost',0.))]
    for command in status['commands']:
        if '--manifest' not in command:continue
        manifest=command[command.index('--manifest')+1]
        if command[1].endswith('evaluate_navigation_suite.py'):
            if Path(command[command.index('--output')+1]).name!='evaluation.json':result+=['--safety-manifest',manifest]
        else:
            flag='--runtime-manifest' if '--runtime-manifest' not in result else '--extra-runtime-manifest'
            result+=[flag,manifest]
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    args=parser.parse_args();root=args.root.resolve();project=Path(__file__).resolve().parents[1]
    state_path=root/'recovery.json'
    with (root/'recovery.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        state=json.loads(state_path.read_text()) if state_path.exists() else dict(status='recovering',target_steps=300000,attempts=[],current=str(root),base_steps=0)
        while True:
            current=Path(state['current']);candidate=current/'candidate'
            if not (candidate/'config.json').exists() and state.get('current_arguments'):
                if current.exists():
                    index=1
                    while current.with_name(current.name+f'.startup_interrupted_{index}').exists():index+=1
                    current.rename(current.with_name(current.name+f'.startup_interrupted_{index}'))
                command=[sys.executable,str(project/'training/run_optimization_pilot.py'),*state['current_arguments']]
                with (root/'recovery.log').open('a') as log:
                    subprocess.run(command,cwd=project,stdout=log,stderr=subprocess.STDOUT,check=True)
                continue
            verify_source(candidate,project)
            config=json.loads((candidate/'config.json').read_text())
            final=candidate/'policy.zip'
            if config['status']=='complete' and final.is_file():
                current_status=json.loads((current/'status.json').read_text())
                if current_status.get('status')=='complete':
                    state.update(status='complete',completed_sampling_steps=state['base_steps']+config['additional_ppo_steps_completed'],
                                 development_gate_passed=current_status['development_gate_passed'],deployment_approved=False)
                    save(state_path,state);print('COMPLETE',json.dumps(state),flush=True);return
                command=[sys.executable,str(project/'training/run_optimization_pilot.py'),*state['current_arguments'],'--resume-evaluation']
            else:
                weights,steps=checkpoint(candidate)
                if weights is None:
                    # No update was checkpointed in this attempt; restart from
                    # its intact parent, never count its discarded samples.
                    weights=Path(config['resume_source']);steps=0
                    with zipfile.ZipFile(weights) as archive:
                        if archive.testzip() is not None:raise RuntimeError('Parent checkpoint corrupted')
                base=state['base_steps']+steps
                remaining=state['target_steps']-base
                if remaining<=0:raise RuntimeError('Final checkpoint exists but final policy/config is incomplete; explicit finalization needed')
                attempt=len(state['attempts'])+1
                parent=root.parent/f'clearance_recovery_parent_{attempt}'
                output=root.parent/f'pilot_clearance_recovery_{attempt}'
                if parent.exists() or output.exists():raise RuntimeError('Recovery output already exists without matching state')
                parent.mkdir()
                for name in ['localizer.pt','router.pt','hardware.json']:shutil.copy2(candidate/name,parent/name)
                shutil.copy2(weights,parent/'policy.zip')
                seed=91000+attempt
                arguments=pilot_arguments(root,parent,output,remaining,seed)
                state['attempts'].append(dict(interrupted=str(current),checkpoint=str(weights),checkpoint_sha256=hashlib.sha256(weights.read_bytes()).hexdigest(),
                                             retained_steps=steps,remaining_steps=remaining,seed=seed,
                                             exact_rollout_or_rng_resume=False,
                                             parallel_envs=int(arguments[arguments.index('--envs')+1])))
                state.update(status='running',current=str(output),current_arguments=arguments,base_steps=base)
                save(state_path,state)
                command=[sys.executable,str(project/'training/run_optimization_pilot.py'),*arguments]
            with (root/'recovery.log').open('a') as log:
                print('START',json.dumps(command),flush=True)
                subprocess.run(command,cwd=project,stdout=log,stderr=subprocess.STDOUT,check=True)


if __name__=='__main__':main()
