"""Frozen policy experiment: clear remembered points using three finite scans."""
import argparse
import json
from pathlib import Path
import shutil
import sys
from experiments.train_quality_navigation import digest
import experiments.evaluate_recovery_temperature as evaluator


def main():
    p=argparse.ArgumentParser()
    for name in ('bundle','policy','reference-run','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():p.error('Fresh output required')
    a.output.mkdir();bundle=a.output/'portable';shutil.copytree(a.bundle,bundle)
    hardware=json.loads((bundle/'hardware.json').read_text())
    old=json.loads(json.dumps(hardware))
    memory=hardware.setdefault('near_field_memory',{})
    memory.update(free_evidence_margin_m=.03,free_evidence_frames=3,free_evidence_max_gap_s=.4)
    (bundle/'hardware.json').write_text(json.dumps(hardware,indent=2))
    manifest=json.loads((bundle/'manifest.json').read_text())
    assert manifest['policy_sha256']==digest(a.policy)
    manifest.update(files={name:digest(bundle/name) for name in manifest['files']},
        deployment_approved=False,
        diagnostic_change='Remembered points cleared only after three distinct contiguous finite free-scan confirmations; margin .03m, gap .4s')
    (bundle/'manifest.json').write_text(json.dumps(manifest,indent=2))
    for name in ('actor.pt','localizer.pt','router.pt','quality_observation.json'):
        assert digest(bundle/name)==digest(a.bundle/name)
    (a.output/'CHANGE.json').write_text(json.dumps(dict(actor_frozen=True,
        source_bundle_manifest_sha256=digest(a.bundle/'manifest.json'),policy_sha256=digest(a.policy),
        original_memory=old.get('near_field_memory'),candidate_memory=memory,
        deployment_approved=False,
        limitation='Experimental free-evidence thresholds, not calibrated against real lidar. No action substitution or blind-sector expiration.'),indent=2))
    sys.argv=[sys.argv[0],'--bundle',str(bundle),'--policy',str(a.policy),
        '--reference-run',str(a.reference_run),'--output',str(a.output/'evaluation')]
    evaluator.main()


if __name__=='__main__':main()
