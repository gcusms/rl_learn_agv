"""Verify controlled frozen-policy memory trials and compare their outcomes."""
import argparse
import json
from pathlib import Path
from experiments.train_quality_navigation import digest


def main():
    p=argparse.ArgumentParser()
    for name in ('candidate','baseline','baseline-bundle','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():p.error('Fresh comparison output required')
    baseline=json.loads((a.baseline/'RESULTS.json').read_text())
    candidate=json.loads((a.candidate/'evaluation/RESULTS.json').read_text())
    change=json.loads((a.candidate/'CHANGE.json').read_text())
    assert baseline['completed'] and candidate['completed']
    assert baseline['policy_sha256']==candidate['policy_sha256']==change['policy_sha256']
    assert baseline['bundle_manifest_sha256']==change['source_bundle_manifest_sha256']==digest(a.baseline_bundle/'manifest.json')
    bundle=a.candidate/'portable'
    manifest=json.loads((bundle/'manifest.json').read_text())
    assert candidate['bundle_manifest_sha256']==digest(bundle/'manifest.json')
    for name,expected in manifest['files'].items():assert digest(bundle/name)==expected
    for name in ('actor.pt','localizer.pt','router.pt','quality_observation.json'):
        assert digest(bundle/name)==digest(a.baseline_bundle/name)
    hardware=json.loads((bundle/'hardware.json').read_text())
    previous=json.loads((a.baseline_bundle/'hardware.json').read_text())
    for item in (hardware,previous):
        for key in ('free_evidence_margin_m','free_evidence_frames','free_evidence_max_gap_s'):
            item['near_field_memory'].pop(key,None)
    assert hardware==previous
    project=Path(__file__).resolve().parents[2]
    assert candidate['source_hashes_unchanged']
    for relative,expected in candidate['source_sha256'].items():assert digest(project/relative)==expected
    rows=[]
    for old,new in zip(baseline['cases'],candidate['cases']):
        cid=old['case_id'];assert cid==new['case_id']
        old_path=a.baseline/(cid+'.json');new_path=a.candidate/'evaluation'/(cid+'.json')
        assert digest(old_path)==old['trace_sha256'];assert digest(new_path)==new['trace_sha256']
        assert json.loads(old_path.read_text())['case']==json.loads(new_path.read_text())['case']
        rows.append(dict(case_id=cid,initial_safe=old['safe_success'],candidate_safe=new['safe_success'],
            initial_collisions=old['collisions'],candidate_collisions=new['collisions'],
            initial_voluntary_wait=old['voluntary_wait_frames'],candidate_voluntary_wait=new['voluntary_wait_frames'],
            initial_only_stop=old['only_stop_frames'],candidate_only_stop=new['only_stop_frames'],
            initial_seconds=old['seconds'],candidate_seconds=new['seconds'],
            initial_travel_m=old['travel_m'],candidate_travel_m=new['travel_m']))
    assert len(rows)==5
    result=dict(controlled_frozen_actor_and_components_verified=True,matched_cases_verified=True,
        candidate_sources_unchanged=True,initial_safe=sum(x['safe_success'] for x in baseline['cases']),
        candidate_safe=sum(x['safe_success'] for x in candidate['cases']),
        initial_collisions=sum(x['collisions'] for x in baseline['cases']),
        candidate_collisions=sum(x['collisions'] for x in candidate['cases']),rows=rows,
        deployment_approved=False,
        limitation='Five reused development cases; execution parity checks come from evaluator, contact labels from simulator. Not broad acceptance, real lidar calibration or proof of semantic understanding.')
    a.output.write_text(json.dumps(result,indent=2));print(json.dumps(result))


if __name__=='__main__':main()
