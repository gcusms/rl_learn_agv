"""Verify every completed fixed group before interpreting the retention gain."""
from datetime import datetime
import hashlib
import json
from pathlib import Path
from zoneinfo import ZoneInfo

from experiments.paired_navigation_gate import compare


ROOT=Path(__file__).resolve().parents[3]
OUT=Path(__file__).resolve().parent
BASE=OUT.parent


def main():
    if (OUT/'FULL35_PAIR.json').exists():
        raise FileExistsError('Completed paired evidence is immutable')
    inputs={str(Path(__file__).relative_to(ROOT)):hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    groups={}
    for suffix,count in [('navigation',18),('prefixes',2),('w11',3),('fresh',12)]:
        rows={}
        for variant,stem in [('reference','w11_recovery_bc800'),
                             ('control','w11_control2000'),('retained','w11_retained2000')]:
            folder=BASE/(stem+'_'+suffix+'_20261016')
            source=folder/'RESULTS.json';raw=source.read_bytes();report=json.loads(raw)
            if not report['completed']:
                raise ValueError('Every fixed evaluation group must finish')
            selected=report['rows']
            if variant=='reference' and suffix=='fresh':
                selected=[r for r in selected if r['variant']=='w11_augmented_bc800']
            if len(selected)!=count or len({r['case_id'] for r in selected})!=count:
                raise ValueError('Exact fixed case identities required')
            inputs[str(source.relative_to(ROOT))]=hashlib.sha256(raw).hexdigest()
            for row in selected:
                tail='' if row.get('recovery_test') else '_'+row['variant']
                path=folder/(row['case_id']+tail+'.npz')
                actual=hashlib.sha256(path.read_bytes()).hexdigest()
                if actual!=row['trace_sha256']:
                    raise ValueError('Completed trace changed')
                inputs[str(path.relative_to(ROOT))]=actual
            rows[variant]=selected
        groups[suffix]=dict(summaries={name:dict(rows=len(items),safe=sum(r['safe_success'] for r in items),
            collisions=sum(r['collisions'] for r in items),timeouts=sum(r['timeout'] for r in items),
            failed_case_ids=[r['case_id'] for r in items if not r['safe_success']]) for name,items in rows.items()},
            reference_to_control=compare(rows['reference'],rows['control']),
            reference_to_retained=compare(rows['reference'],rows['retained']),
            control_to_retained=compare(rows['control'],rows['retained']))
    previous=OUT/'RESULTS.json'
    inputs[str(previous.relative_to(ROOT))]=hashlib.sha256(previous.read_bytes()).hexdigest()
    original=json.loads(previous.read_text())
    result=dict(completed=True,at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
        separate_fixed_groups=groups,original14_and_directional_gates=original['gates'],
        completed_rows_per_candidate=35,
        full35_retained_comparison_complete=True,aggregated_independent_success_rate=None,
        inputs=inputs,new_physical_steps=0,new_ppo_steps=0,
        both_candidates_fail_original_success_preservation=True,
        no_automatic_promotion=True,goal_achieved=False,deployment_approved=False)
    for candidate in ['control','retained']:
        if not original['gates'][candidate]['original14']['lost_successes']:
            raise ValueError('Recheck the explicitly reported original-success failure')
    (OUT/'FULL35_PAIR.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(completed=True,groups={name:value['summaries'] for name,value in groups.items()},
        source_files_checked=len(inputs)),ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
