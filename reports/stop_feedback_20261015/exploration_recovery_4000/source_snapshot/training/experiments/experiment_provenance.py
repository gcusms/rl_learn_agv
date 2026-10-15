"""Freeze the source and model identity before each new validation run."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess


def record_provenance(output,bundle,extra_sources=()):
    root=Path(__file__).resolve().parents[2];output=Path(output);snapshot=output/'source_snapshot'
    snapshot.mkdir(exist_ok=False)
    sources={}
    for folder in ('training/navrl','training/neural_navigation','training/experiments'):
        for source in (root/folder).glob('*.py'):
            if source.name.startswith('test_'):continue
            relative=source.relative_to(root);target=snapshot/relative;target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(source,target);sources[str(relative)]=hashlib.sha256(source.read_bytes()).hexdigest()
    for source in map(Path,extra_sources):
        relative=source.relative_to(root);target=snapshot/relative;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(source,target);sources[str(relative)]=hashlib.sha256(source.read_bytes()).hexdigest()
    provenance=dict(parent_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
        bundle_manifest_sha256=hashlib.sha256((Path(bundle)/'manifest.json').read_bytes()).hexdigest(),source_sha256=sources)
    (output/'PROVENANCE.json').write_text(json.dumps(provenance,indent=2))
    return provenance
