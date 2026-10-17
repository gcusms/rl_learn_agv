"""Verify the published snapshot with the standard library; never run the robot."""
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RELEASE = ROOT / 'docs/releases/20261017'


def checked_path(relative):
    path = (ROOT / relative).resolve()
    path.relative_to(ROOT)
    return path


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def main():
    index = json.loads((RELEASE / 'FILES.json').read_text())
    for name, expected in index['files'].items():
        if digest(checked_path(name)) != expected:
            raise ValueError('Snapshot file mismatch: ' + name)
    snapshot = json.loads((RELEASE / 'SNAPSHOT.json').read_text())
    candidate = checked_path(snapshot['candidate']['path'])
    if digest(candidate / 'checkpoint.pt') != snapshot['candidate']['checkpoint_sha256']:
        raise ValueError('Candidate checkpoint mismatch')
    comparison = json.loads(checked_path(snapshot['comparison']).read_text())
    latest = comparison['recent_versions']['latest2501']
    if latest['checkpoint_sha256'] != snapshot['candidate']['checkpoint_sha256']:
        raise ValueError('Comparison belongs to another candidate')
    if latest['saved_check_counts'] != snapshot['saved_checks']:
        raise ValueError('Saved evaluation counts differ from snapshot')
    if snapshot['deployment_approved'] or comparison['deployment_approved']:
        raise ValueError('This snapshot has not passed deployment acceptance')
    print(json.dumps(dict(completed=True, branch=snapshot['branch'],
        checked_files=len(index['files']), candidate_sha256=latest['checkpoint_sha256'],
        saved_checks=snapshot['saved_checks'], training_status=snapshot['training_status'],
        new_training_steps=0, new_physical_episodes=0, deployment_approved=False,
        scope='File integrity and saved-result consistency; no navigation or hardware acceptance.'),
        ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
