"""Verify the fixed cross-PC snapshot using only the Python standard library."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, help='Optional new output directory')
    args = parser.parse_args()
    if args.output is not None and args.output.exists():
        parser.error('The output directory must not exist')
    index = json.loads((ROOT / 'docs/handoff/FILES.json').read_text())
    branch = index.get('branch')
    if not isinstance(branch, str) or not branch:
        raise ValueError('Missing handoff branch metadata')
    for name, expected in index['files'].items():
        if digest(checked_path(name)) != expected:
            raise ValueError('File checksum mismatch: ' + name)
    snapshot = json.loads(checked_path(index['snapshot']).read_text())
    if snapshot['branch'] != branch or snapshot['deployment_approved']:
        raise ValueError('Snapshot metadata mismatch')
    trace_count = 0
    for name, item in snapshot['validation'].items():
        if not item['started']:
            continue
        path = checked_path(item['report'])
        if digest(path) != item['report_sha256']:
            raise ValueError('Report changed: ' + name)
        report = json.loads(path.read_text())
        rows = report['rows']
        if len(rows) != item['finished_rows'] or report['completed'] != item['completed']:
            raise ValueError('Evaluation completion mismatch: ' + name)
        if report['completed'] and len(rows) != item['expected_rows']:
            raise ValueError('Incomplete evaluation marked complete: ' + name)
        for filename, expected in item['trace_files'].items():
            if Path(filename).name != filename or digest(path.parent / filename) != expected:
                raise ValueError('Trace checksum mismatch: ' + name)
            trace_count += 1
        hashes = set(item['trace_files'].values())
        if any(row['trace_sha256'] not in hashes for row in rows):
            raise ValueError('Missing finished trace: ' + name)
    for name, item in snapshot['models'].items():
        if digest(checked_path(item['folder'] + '/' + item['checkpoint'])) != item['sha256']:
            raise ValueError('Model checksum mismatch: ' + name)
    report = dict(completed=True, at=datetime.now(timezone.utc).isoformat(), branch=branch,
        snapshot=index['snapshot'], pinned_files_checked=len(index['files']),
        finished_traces_checked=trace_count, model_hashes_checked=len(snapshot['models']),
        new_physical_steps=0, new_training_steps=0, deployment_approved=False,
        scope='Byte integrity and frozen completion metadata; no navigation/real-world acceptance.')
    if args.output is not None:
        args.output.mkdir(parents=True, exist_ok=False)
        (args.output / 'RESULTS.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
