"""Continue PPO on independent training layouts, evaluate, then record a gate.

Never promotes a candidate to the robot launcher or uses final acceptance cases.
"""
import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
import numpy as np


def completed_artifact(command,candidate):
    """Only reuse completed outputs tied to this policy and frozen input."""
    output=Path(command[command.index('--output')+1])
    policy_hash=hashlib.sha256((candidate/'policy.zip').read_bytes()).hexdigest()
    if output.is_dir():
        manifest=output/'manifest.json'
        if not manifest.exists():return False
        try:x=json.loads(manifest.read_text())
        except (ValueError,OSError):return False
        return x.get('policy_sha256')==policy_hash and all((output/n).is_file() and hashlib.sha256((output/n).read_bytes()).hexdigest()==h for n,h in x['files'].items())
    if not output.exists():return False
    try:x=json.loads(output.read_text())
    except (ValueError,OSError):return False
    source=Path(command[command.index('--manifest')+1])
    return bool(x.get('completed') and x.get('manifest_sha256')==hashlib.sha256(source.read_bytes()).hexdigest()
                and (x.get('policy_sha256') or x.get('bundle_manifest',{}).get('policy_sha256'))==policy_hash)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--parent',type=Path,required=True)
    parser.add_argument('--baseline',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--steps',type=int,default=30000)
    parser.add_argument('--envs',type=int,default=12)
    parser.add_argument('--seed',type=int,default=56000)
    parser.add_argument('--stage',choices=['static','dynamic','robust'],default='static')
    parser.add_argument('--safety-manifest',type=Path,action='append',default=[],help='Extra frozen development regressions; all must be safe')
    parser.add_argument('--runtime-manifest',type=Path,help='Closed-loop exported runtime/guard development suite')
    parser.add_argument('--execution-mode',choices=['raw','guarded'],default='raw')
    parser.add_argument('--motion-profiles',action='store_true')
    parser.add_argument('--footprint-clearance-cost',type=float,default=0.)
    parser.add_argument('--extra-runtime-manifest',type=Path,action='append',default=[])
    parser.add_argument('--resume-evaluation',action='store_true',help='Reuse completed training and verified evaluation artifacts after interruption')
    args=parser.parse_args();project=Path(__file__).resolve().parents[1]
    if args.output.exists() and not args.resume_evaluation:parser.error('Use a fresh pilot directory')
    if args.resume_evaluation:
        config=args.output/'candidate/config.json'
        if not config.exists() or json.loads(config.read_text()).get('status')!='complete' or not (args.output/'candidate/policy.zip').is_file():
            parser.error('Evaluation resume requires completed training and its final policy')
    if min(args.steps,args.envs)<=0:parser.error('Training sizes must be positive')
    baseline=json.loads(args.baseline.read_text())
    if not baseline['completed'] or len(baseline['cases'])!=40:parser.error('A complete development baseline is required')
    if baseline['route_mode']!='shortcut':parser.error('Compare identical route modes')
    args.output.mkdir(parents=True,exist_ok=args.resume_evaluation);candidate=args.output/'candidate';evaluation=args.output/'evaluation.json'
    commands=[
        [sys.executable,str(project/'training/remote/train_parallel.py'),'--parent-policy',str(args.parent/'policy.zip'),
         '--components',str(args.parent),'--envs',str(args.envs),'--device','cpu','--steps',str(args.steps),
         '--anchor-batches','0','--curriculum',args.stage,'--route-mode','shortcut','--seed',str(args.seed),'--execution-mode',args.execution_mode,
         '--checkpoint-every','10000','--output',str(candidate)],
        [sys.executable,str(project/'training/evaluate_navigation_suite.py'),'--policy',str(candidate/'policy.zip'),
         '--components',str(candidate),'--manifest',str(project/'training/config/navigation_development_v3.json'),
         '--route-mode','shortcut','--workers',str(args.envs),'--output',str(evaluation)]]
    names=['training','evaluation'];safety_outputs=[]
    if args.motion_profiles:commands[0].append('--motion-profiles')
    if args.footprint_clearance_cost:commands[0].extend(['--footprint-clearance-cost',str(args.footprint_clearance_cost)])
    for index,manifest in enumerate(args.safety_manifest):
        extra=args.output/f'safety_{index}.json';safety_outputs.append(extra);names.append(f'safety_{index}')
        commands.append([sys.executable,str(project/'training/evaluate_navigation_suite.py'),'--policy',str(candidate/'policy.zip'),
                         '--components',str(candidate),'--manifest',str(manifest.resolve()),'--route-mode','shortcut',
                         '--workers',str(args.envs),'--output',str(extra)])
    runtime_output=None;extra_runtime_outputs=[]
    if args.runtime_manifest:
        portable=args.output/'portable';runtime_output=args.output/'sensor_runtime.json'
        names.extend(['export','sensor_runtime'])
        commands.extend([
            [sys.executable,str(project/'training/export_navigation_actor.py'),'--policy',str(candidate/'policy.zip'),'--components',str(candidate),'--output',str(portable)],
            [sys.executable,str(project/'training/evaluate_sensor_navigation.py'),'--bundle',str(portable),'--manifest',str(args.runtime_manifest.resolve()),
             '--workers',str(args.envs),'--output',str(runtime_output)]])
        for index,manifest in enumerate(args.extra_runtime_manifest):
            path=args.output/f'sensor_runtime_extra_{index}.json';extra_runtime_outputs.append(path);names.append(f'sensor_runtime_extra_{index}')
            commands.append([sys.executable,str(project/'training/evaluate_sensor_navigation.py'),'--bundle',str(portable),'--manifest',str(manifest.resolve()),
                             '--workers',str(args.envs),'--output',str(path)])
    elif args.extra_runtime_manifest:parser.error('Extra runtime suites require --runtime-manifest')
    status=dict(status='running',stage=args.stage,commands=commands,requested_steps=args.steps,seed=args.seed,
                execution_mode=args.execution_mode,motion_profiles=args.motion_profiles,footprint_clearance_cost=args.footprint_clearance_cost,
                deployment_approved=False,final_acceptance_used=False)
    (args.output/'status.json').write_text(json.dumps(status,indent=2))
    try:
        for name,command in zip(names,commands):
            if args.resume_evaluation:
                if name=='training':continue
                if completed_artifact(command,candidate):
                    print('REUSE',name,flush=True);continue
                output=Path(command[command.index('--output')+1])
                if output.exists():
                    # Preserve partial evidence; tools require fresh outputs.
                    index=1
                    while output.with_name(output.name+f'.interrupted_{index}').exists():index+=1
                    output.rename(output.with_name(output.name+f'.interrupted_{index}'))
            print('START',name,flush=True)
            status['current_phase']=name
            (args.output/'status.json').write_text(json.dumps(status,indent=2))
            with (args.output/f'{name}.log').open('w') as log:
                subprocess.run(command,cwd=project,stdout=log,stderr=subprocess.STDOUT,check=True)
        report=json.loads(evaluation.read_text());old={c['case_id']:c for c in baseline['cases']};new={c['case_id']:c for c in report['cases']}
        if not report['completed'] or old.keys()!=new.keys():raise RuntimeError('Evaluation case mismatch')
        if baseline['manifest_sha256']!=report['manifest_sha256']:raise RuntimeError('Frozen evaluation manifest mismatch')
        means=lambda d,key:float(np.mean([c[key] for c in d.values()]))
        timing=means(new,'seconds')/means(old,'seconds');length=means(new,'travel_m')/means(old,'travel_m')
        checks=dict(all_safe=all(c['safe_success'] for c in new.values()),
                    actual_goal_within_20cm=all(c['final_true_goal_distance_m']<=.2 for c in new.values()),
                    time_improved_at_least_1pct=timing<=.99,path_increase_at_most_1pct=length<=1.01,
                    localization_no_material_regression=all(new[k]['localization_error_p95_m']<=old[k]['localization_error_p95_m']+.02 for k in old))
        extras=[]
        for path in safety_outputs:
            extra=json.loads(path.read_text());passed=extra['completed'] and all(c['safe_success'] and c['final_true_goal_distance_m']<=.2 for c in extra['cases'])
            extras.append(dict(report=str(path),all_safe=bool(passed),tasks=len(extra['cases'])))
        if extras:checks['additional_suites_all_safe']=all(r['all_safe'] for r in extras)
        if runtime_output:
            runtime=json.loads(runtime_output.read_text())
            checks['sensor_runtime_all_safe']=runtime['completed'] and runtime['safe_success']==runtime['tasks']
            status['sensor_runtime_summary']=dict(tasks=runtime['tasks'],safe_success=runtime['safe_success'],collisions=runtime['collisions'],
                                                   override_steps=sum(c['emergency_override_steps'] for c in runtime['cases']))
            additional=[]
            for path in extra_runtime_outputs:
                extra=json.loads(path.read_text());additional.append(dict(report=str(path),tasks=extra['tasks'],safe_success=extra['safe_success'],collisions=extra['collisions'],
                                                                        all_safe=extra['completed'] and extra['safe_success']==extra['tasks']))
            status['additional_runtime_suites']=additional
            if additional:checks['additional_runtime_suites_all_safe']=all(x['all_safe'] for x in additional)
        status.update(status='complete',development_gate_passed=all(checks.values()),checks=checks,
                      additional_safety_suites=extras,
                      time_ratio=timing,path_ratio=length,baseline=str(args.baseline.resolve()),
                      localization_10cm_goal_met=all(c['localization_error_p95_m']<=.1 for c in new.values()),
                      limitation='40 development cases only; no claim of deployment readiness, dynamic optimality or Orin timing. Candidate and original retained independently.')
    except Exception as error:
        status.update(status='failed',error=str(error));raise
    finally:(args.output/'status.json').write_text(json.dumps(status,indent=2))
    print(json.dumps(status,indent=2),flush=True)


if __name__=='__main__':main()
