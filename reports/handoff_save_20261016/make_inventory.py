"""Pin the files staged for the save branch; verification outputs are separate."""
from datetime import datetime
import hashlib
import json
from pathlib import Path
import subprocess
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parents[2]
EXCLUDED = {
    'docs/handoff/FILES.json',
    'docs/handoff/VERIFICATION.json',
    'docs/handoff/verification_save_tests.txt',
}


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def main():
    names = subprocess.check_output(['git', 'ls-files', '-z'], cwd=ROOT).decode().rstrip('\0').split('\0')
    files = {}
    total = 0
    for name in names:
        if name in EXCLUDED:
            continue
        path = ROOT / name
        size = path.stat().st_size
        if size >= 100 * 1024**2:
            raise ValueError('File exceeds the normal GitHub per-file limit: ' + name)
        files[name] = digest(path)
        total += size
    snapshot = 'reports/handoff_save_20261016/SNAPSHOT.json'
    manifest = dict(schema='agv_handoff_files_v14', branch='save',
        date=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
        snapshot=snapshot, snapshot_captured_at=json.loads((ROOT/snapshot).read_text())['captured_at'],
        parent_handoff_branch='dev',
        parent_handoff_commit='8fe7c0b9d6ad2bcbc36058acf41ece3715ff6e5d',
        original_deployment_replaced=False, deployment_approved=False,
        independent_completed_ppo_steps_per_model={'control':2000,'retained':2000},
        training_running_at_snapshot=False, paired_evaluation_running_on_source=True,
        training_snapshot='models/stop_feedback_20261015/proactive_ppo_ship_20261016',
        training_snapshot_scope='Historical unfinished 7k snapshot only; both current 2k models are complete.',
        exported_pinned_file_bytes=total,
        excluded_self_referential_verification_records=sorted(EXCLUDED),
        files=files)
    (ROOT/'docs/handoff/FILES.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(pinned_files=len(files),bytes=total,branch='save')))


if __name__ == '__main__':
    main()
