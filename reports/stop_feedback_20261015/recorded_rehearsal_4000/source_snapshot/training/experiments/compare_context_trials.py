"""Compare complete sensor-runtime trials without double-counting shared cases."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from experiments.analyze_context_traces import analyze
from navrl.geometry import ROOT
from neural_navigation.safety_context import validate_contract


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def manifests(paths):
    by_hash={};unique={}
    for path in paths:
        data=json.loads(path.read_text())
        if data.get('kind') not in ('development','training_failure_replay'):
            raise ValueError('Only development or training-failure manifests are allowed')
        cases={}
        for case in data['cases']:
            if case['id'] in cases:raise ValueError('Duplicate ID within manifest')
            normalized=dict(case)
            normalized['scene']=str((ROOT/case.get('scene','assets/room_navigation_scripted')).resolve())
            if case['id'] in unique and unique[case['id']]!=normalized:
                raise ValueError('Shared case ID has different world definitions')
            cases[case['id']]=normalized;unique[case['id']]=normalized
        by_hash[digest(path)]=cases
    return by_hash,unique


def read_trial(run,known,expected):
    status=json.loads((run/'status.json').read_text())
    if status['status']!='complete':raise ValueError(f'{run.name} has not completed training/evaluation')
    bundle=run/'portable';manifest=json.loads((bundle/'manifest.json').read_text())
    if digest(run/'candidate/policy.zip')!=manifest['policy_sha256']:
        raise ValueError('Report actor is not the final candidate checkpoint')
    for name,checksum in manifest['files'].items():
        if digest(bundle/name)!=checksum:raise ValueError('Bundle file hash mismatch')
    hardware=json.loads((bundle/'hardware.json').read_text());validate_contract(hardware['policy_observation'])
    unique={};sources={};overlap_checks=0
    for path in sorted(run.glob('runtime_[0-9]*.json')):
        report=json.loads(path.read_text())
        if not report['completed'] or report['bundle_manifest']!=manifest:
            raise ValueError('Incomplete report or different evaluated actor')
        cases=report['cases'];ids=[x['case_id'] for x in cases]
        if report['manifest_sha256'] not in known or set(ids)!=set(known[report['manifest_sha256']]):
            raise ValueError('Report case set/manifest differs from frozen input')
        if len(ids)!=len(set(ids)) or report['tasks']!=len(ids):raise ValueError('Duplicate or missing report cases')
        if report['safe_success']!=sum(x['safe_success'] for x in cases):raise ValueError('Inconsistent success aggregate')
        for case in cases:
            verified=bool(case['arrived'] and case['protocol_valid'] and not case['false_arrival'] and
                not case['collisions'] and not case['physical_guards'])
            if case['safe_success']!=verified:raise ValueError('Inconsistent safe-arrival label')
            timeout=bool(not case['arrived'] and not case['false_arrival'] and
                not case['collisions'] and not case['physical_guards'])
            if case['timeout']!=timeout:raise ValueError('Inconsistent timeout label')
            key=case['case_id']
            if key in unique:
                for field in ('safe_success','collisions','physical_guards','protocol_valid','timeout'):
                    if case[field]!=unique[key][field]:raise ValueError('Repeated case outcome changed within trial')
                overlap_checks+=1
                continue
            unique[key]=case
            sources[key]=run/f'traces_{path.stem.split("_")[1]}'/(key+'.json')
    if set(unique)!=set(expected):raise ValueError('Full required unique case set is not evaluated')
    trace_results={}
    for key,path in sources.items():
        trace=json.loads(path.read_text())
        if trace['case']['id']!=key or trace['result']!=unique[key]:raise ValueError('Trace/report result mismatch')
        world=dict(trace['case']);world['scene']=str((ROOT/world.get('scene','assets/room_navigation_scripted')).resolve())
        if world!=expected[key]:raise ValueError('Trace world differs from frozen case')
        trace_results[key]=analyze(trace)
    rows=list(unique.values())
    summary=dict(unique_tasks=len(rows),overlap_reports_checked=overlap_checks,
        safe_success=sum(x['safe_success'] for x in rows),collisions=sum(x['collisions'] for x in rows),
        timeouts=sum(x['timeout'] for x in rows),physical_guards=sum(x['physical_guards'] for x in rows),
        schema=hardware['policy_observation']['schema'],policy_sha256=manifest['policy_sha256'],
        development_gate_passed=all(x['safe_success'] for x in rows),
        failures=[x['case_id'] for x in rows if not x['safe_success']])
    for metric in ('idle_with_forward_probe','idle_with_only_turn_probe','rejected_reverse_with_forward_probe'):
        classifications=[x['classifications'][metric] for x in trace_results.values() if x['classifications'] is not None]
        summary[metric]=dict(steps=sum(x['steps'] for x in classifications),
            longest_seconds=max((x['longest_seconds'] for x in classifications),default=0.))
    return dict(summary=summary,cases=unique,trace_analysis=trace_results,
        physical_hardware={k:v for k,v in hardware.items() if k!='policy_observation'},
        learned_component_hashes={k:manifest['files'][k] for k in ('localizer.pt','router.pt')})


def compare(trials):
    first=next(iter(trials.values()))
    for trial in trials.values():
        if trial['physical_hardware']!=first['physical_hardware'] or trial['learned_component_hashes']!=first['learned_component_hashes']:
            raise ValueError('Physical limits/sensors/guard memory or frozen learned components differ')
    common=set.intersection(*(set(k for k,v in trial['cases'].items() if v['safe_success']) for trial in trials.values()))
    paired={}
    for name,trial in trials.items():
        rows=[trial['cases'][k] for k in sorted(common)]
        paired[name]=dict(common_safe_cases=len(rows),
            mean_seconds=float(np.mean([x['seconds'] for x in rows])) if rows else None,
            mean_travel_m=float(np.mean([x['travel_m'] for x in rows])) if rows else None,
            mean_overrides=float(np.mean([x['emergency_override_steps'] for x in rows])) if rows else None)
    return dict(trials=trials,paired_common_safe_efficiency=paired,deployment_approved=False,
        goal_complete=False,note='Safety failures take precedence over efficiency. Nine present-frame probes do not prove future dynamic safety. '
        'Shared cases are counted once. Training-seed/environment-count differences limit causal interpretation; no real-board acceptance claim.')


def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,action='append',required=True)
    p.add_argument('--run',type=Path,action='append');p.add_argument('--output',type=Path)
    p.add_argument('--check-manifests-only',action='store_true');a=p.parse_args()
    known,expected=manifests(a.manifest)
    if a.check_manifests_only:
        print(json.dumps(dict(manifests=len(known),unique_cases=len(expected))));return
    if not a.run or len(a.run)<2 or a.output is None or a.output.exists():
        p.error('At least two completed runs and a fresh output are required')
    if len({x.resolve() for x in a.run})!=len(a.run):p.error('Run paths must be distinct')
    result=compare({str(run.resolve()):read_trial(run.resolve(),known,expected) for run in a.run})
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2))
    print(json.dumps({name:x['summary'] for name,x in result['trials'].items()},indent=2))


if __name__=='__main__':main()
