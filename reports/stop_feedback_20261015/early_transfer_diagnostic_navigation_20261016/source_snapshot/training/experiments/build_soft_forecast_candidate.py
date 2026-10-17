"""Isolate forecast route semantics; no learned weights or guards are changed."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
from neural_navigation.goal_pose_runtime import check_files
from neural_navigation.routing_contract import soft_forecast_contract


def build(source,output):
    source=Path(source);output=Path(output)
    manifest=json.loads((source/'manifest.json').read_text());check_files(source,manifest)
    if output.exists():raise FileExistsError(output)
    output.mkdir(parents=True)
    for name in manifest['files']:shutil.copy2(source/name,output/name)
    path=output/'routing_contract.json';path.write_text(json.dumps(soft_forecast_contract(),indent=2)+'\n')
    manifest['files'][path.name]=hashlib.sha256(path.read_bytes()).hexdigest()
    manifest.update(deployment_approved=False,parent_manifest_sha256=hashlib.sha256((source/'manifest.json').read_bytes()).hexdigest(),
        routing_acceptance=soft_forecast_contract()['schema'])
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');check_files(output,manifest)
    for name in ('actor.pt','localizer.pt','router.pt','hardware.json'):
        assert (source/name).read_bytes()==(output/name).read_bytes()
    (output/'MODEL_CARD.json').write_text(json.dumps(dict(deployment_approved=False,actor_trained=False,
        actor_router_localizer_guard_byte_identical=True,parent=str(source.resolve()),
        method='separate future-only route costs from measured hard occupancy'),indent=2)+'\n')
    return output


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();print(build(a.source,a.output))
