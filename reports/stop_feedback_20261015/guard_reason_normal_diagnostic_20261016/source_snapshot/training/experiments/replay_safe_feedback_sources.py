"""Rebuild 625D causal observations on existing complete-safe 439D traces.

Physical success is reused evidence, not a new episode or candidate success.
Every sensor prefix, action mask, execution and released reference decision
must reproduce. Initial scripted histories are excluded from later labels.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np
import torch

from experiments.long_risk_actor import digest
from neural_navigation.experiment_budget import ExperimentBudget
from neural_navigation.masked_actions import ACTIONS, masks_from_observation
from neural_navigation.release_runtime import ReleaseRuntime


ROOT = Path(__file__).resolve().parents[2]


def write(path, value):
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,indent=2)+'\n')
    temporary.replace(path)


def main(args):
    output, model, specifications_path = [getattr(args,name).resolve()
        for name in ['output','model','specifications']]
    output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1)
    budget=ExperimentBudget(.35,memory_mib=2500,directory=output)
    specifications=json.loads(specifications_path.read_text())
    assert 1<=len(specifications)<=3
    protocol_path=specifications_path.parent/'PLAN.json'
    protocol=json.loads(protocol_path.read_text())
    for name,expected in protocol['inputs'].items():
        assert digest(ROOT/name)==expected,name
    bundle=model/'portable'
    manifest=json.loads((bundle/'manifest.json').read_text())
    files=[Path(__file__),specifications_path,protocol_path,bundle/'manifest.json']
    files += [ROOT/name for name in protocol['inputs']]
    files += [bundle/name for name in manifest['files']]
    for folder in ['neural_navigation','navrl','experiments']:
        files += list((ROOT/'training'/folder).rglob('*.py'))
    for spec in specifications:
        files += [ROOT/spec[name] for name in ['trace','reference_actor','physical_source_report']]
        files += [p for p in (ROOT/'training'/spec['case']['scene']).iterdir() if p.is_file()]
    hashes={str(p.relative_to(ROOT)):digest(p) for p in set(files)}
    write(output/'PLAN.json',dict(inputs=hashes,maximum_recorded_sources=len(specifications),
        sensor_only_replay=True,simulator_truth_not_input=True,original_439_prefix_atol=2e-6,
        actions_masks_and_execution_must_reproduce=True,new_physical_episodes=0,
        original_physical_success_reused_explicitly=True,new_optimizer_updates=0,new_ppo_steps=0))
    (output/'driver_snapshot.py').write_text(Path(__file__).read_text())
    report=dict(completed=False,rows=[],new_physical_episodes=0,new_optimizer_updates=0,new_ppo_steps=0,
        not_candidate_navigation_success=True,goal_achieved=False,deployment_approved=False)

    def save():
        budget.tick()
        report.update(at=datetime.now(timezone.utc).isoformat(),resource=budget.statistics())
        write(output/'RESULTS.json',report)

    def finish(reason):
        for name,expected in hashes.items():
            assert digest(ROOT/name)==expected,name
        report.update(completed=True,reason=reason,simulator_loaded='mujoco' in sys.modules)
        save()
        write(output/'STATUS.json',dict(status='complete',reason=reason,converted_sources=len(report['rows'])))

    save()
    for spec in specifications:
        write(output/'STATUS.json',dict(status='causal_sensor_replay',case_id=spec['id'],finished_sources=len(report['rows'])))
        row=dict(spec['physical_source_row'])
        assert row['safe_success'] and row['physical_arrival'] and row['protocol_valid']
        assert not any(row[k] for k in ['collisions','physical_guards','timeout','false_arrival'])
        physical=json.loads((ROOT/spec['physical_source_report']).read_text())
        assert physical['completed']
        matches=[r for r in physical['rows'] if r['case_id']==row['case_id'] and r.get('variant')==row.get('variant')]
        assert len(matches)==1 and matches[0]==row
        path=ROOT/spec['trace']
        assert digest(path)==row['trace_sha256']
        reference_path=ROOT/spec['reference_actor']
        assert digest(reference_path.parent/'hardware.json')==digest(bundle/'hardware.json')
        reference=torch.jit.load(str(reference_path),map_location='cpu').eval()
        with np.load(path,allow_pickle=False) as archive:
            trace={key:archive[key].copy() for key in archive.files}
        controlled=trace['model_controlled'].astype(bool)
        count=len(trace['choices']);n=int(np.flatnonzero(controlled)[0])
        np.testing.assert_array_equal(controlled,np.arange(count)>=n)
        assert abs(trace['labels'][n,0]-spec['release_s'])<1e-8
        cursor=[0]

        class ReplayPolicy:
            def predict(self,observation):
                index=cursor[0]
                expected=ACTIONS[trace['choices'][index]]
                if not controlled[index]:
                    return expected.copy()
                with torch.no_grad():
                    action=reference(torch.from_numpy(np.asarray(observation[:439],np.float32)[None])).numpy()[0]
                np.testing.assert_array_equal(action,expected)
                return action.copy()

        case=spec['case'];prior=np.asarray(case['start'])+np.asarray(case.get('initial_prior_offset',[0,0,0]))
        runtime=ReleaseRuntime(bundle,ROOT/'training'/case['scene']/'decision_map.yaml',prior,case['goal'],'shortcut')
        runtime.policy.policy=ReplayPolicy()
        observations=[];max_prefix_difference=0.
        try:
            for i in range(count):
                cursor[0]=i
                result=runtime.step(trace['scans'][i],trace['odometry'][i],trace['measured_velocity'][i],
                    float(trace['labels'][i,0]),capture_observation=True)
                observation=np.asarray(result['policy_observation'],np.float32)
                np.testing.assert_allclose(observation[:439],trace['observations'][i],rtol=0,atol=2e-6)
                max_prefix_difference=max(max_prefix_difference,float(np.max(abs(observation[:439]-trace['observations'][i]))))
                np.testing.assert_array_equal(masks_from_observation(observation[:282]),trace['masks'][i])
                np.testing.assert_array_equal(np.asarray(result['proposed_action']),ACTIONS[trace['choices'][i]])
                np.testing.assert_allclose(result['command'],trace['commands'][i],rtol=0,atol=2e-8)
                observations.append(observation)
                budget.tick()
                if i%100==0:
                    write(output/'STATUS.json',dict(status='causal_sensor_replay',case_id=spec['id'],
                        matched_frames=i+1,total_frames=count,finished_sources=len(report['rows'])))
            assert result['stop_reason']=='estimated_goal_reached'
        except Exception as error:
            write(output/'FAILURE.json',dict(case_id=spec['id'],matched_frames=len(observations),
                error=repr(error),partial_trace_not_admitted=True))
            finish('Recorded source did not reproduce; no partial-source admission or expansion')
            return
        prior_observations=trace['observations'].copy()
        trace['observations']=np.asarray(observations)
        trace['reference_replay_controlled']=controlled.copy()
        trace['original_439_observations']=prior_observations
        destination=output/(spec['id']+'.npz')
        np.savez_compressed(destination,**trace)
        row.update(case_id=spec['id'],physical_source_case_id=spec['physical_source_row']['case_id'],
            physical_source_report=spec['physical_source_report'],physical_source_trace=spec['trace'],
            physical_source_trace_sha256=spec['physical_source_row']['trace_sha256'],
            reference_actor=spec['reference_actor'],reference_actor_sha256=digest(reference_path),
            trace_path=str(destination.relative_to(ROOT)),trace_sha256=digest(destination),
            frozen_prefix_frames=n,reference_replayed_frames=int(controlled.sum()),
            maximum_439_prefix_absolute_difference=max_prefix_difference,
            original_masks_actions_and_execution_reproduced=True,
            independent_original_physical_success_reused=True,no_new_physical_success_claim=True,
            reference_success_is_not_candidate_success=True,complete_safe_source_admitted=True)
        report['rows'].append(row);save()
        print(json.dumps(dict(case_id=spec['id'],frames=count,reference_frames=int(controlled.sum()),
            max_prefix_difference=max_prefix_difference,source_admitted=True,new_physical_episodes=0)),flush=True)
        del runtime,reference
    finish('All complete-safe recorded sources causally reproduced; label admission still required')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['output','model','specifications']:
        parser.add_argument('--'+name,type=Path,required=True)
    main(parser.parse_args())
