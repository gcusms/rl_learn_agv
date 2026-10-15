"""Export the lossless inputs needed by the localization replay, without ROS.

This is an analysis fixture, not a replacement for the original rosbag. Output
keeps original float values, ordering, timestamps and manual reset boundaries.
It deliberately contains no new ground-truth pose labels.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def export(record, analysis, output):
    record, analysis, output = map(Path, (record, analysis, output))
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    (output / 'analysis').mkdir()
    (output / 'record').mkdir()
    shutil.copy2(analysis / 'sensors.npz', output / 'analysis/sensors.npz')
    feedback = json.loads((analysis / 'feedback.json').read_text())
    feedback_keys = ('fresh', 'measurement_valid', 'measured_rps',
                     'body_velocity', 'geometry_m')
    reduced = [dict(t=row['t'], data={key: row['data'][key] for key in feedback_keys})
               for row in feedback]
    (output / 'analysis/feedback.json').write_text(json.dumps(reduced) + '\n')
    count = 0
    with (record / 'decisions.jsonl').open() as src, (output / 'record/decisions.jsonl').open('w') as dst:
        for line in src:
            original = json.loads(line)['data']
            row = {key: original[key] for key in
                   ('mission_id', 'inference_id', 'scan_stamp_ns',
                    'decision_stamp_ns', 'initial', 'goal')}
            row['inputs'] = {key: original['inputs'][key] for key in
                             ('scan', 'odom', 'velocity', 'scan_valid', 'received_timestamp')}
            row['result'] = dict(timestamp=original['result']['timestamp'])
            dst.write(json.dumps(dict(data=row), separators=(',', ':')) + '\n')
            count += 1
    session = json.loads((record / 'session.json').read_text())
    (output / 'record/session.json').write_text(json.dumps(dict(started_unix=session['started_unix'])) + '\n')
    run, = (record / 'snapshots/runs').iterdir()
    target = output / 'record/snapshots/runs' / run.name / 'prepared_map'
    target.mkdir(parents=True)
    for name in ('map.yaml', 'map.pgm'):
        shutil.copy2(run / 'prepared_map' / name, target / name)
    inputs = [analysis / 'sensors.npz', analysis / 'feedback.json',
              record / 'decisions.jsonl', record / 'session.json',
              run / 'prepared_map/map.yaml', run / 'prepared_map/map.pgm']
    manifest = dict(schema='localization_recording_inputs_v1', decisions=count,
        feedback_frames=len(feedback), independent_pose_truth_available=False,
        source_sha256={str(p.relative_to(analysis if p.is_relative_to(analysis) else record)): sha256(p)
                       for p in inputs},
        projection='Only unused JSON fields removed; sensors.npz and prepared map copied byte-for-byte.',
        clocks='Sensor rows retain receive/run-relative and acquisition timestamps; never mix them with ROS epoch.',
        files={str(p.relative_to(output)): sha256(p)
               for p in sorted(output.rglob('*')) if p.is_file()})
    (output / 'MANIFEST.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--record', type=Path, required=True)
    parser.add_argument('--analysis', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(export(args.record, args.analysis, args.output), indent=2))
