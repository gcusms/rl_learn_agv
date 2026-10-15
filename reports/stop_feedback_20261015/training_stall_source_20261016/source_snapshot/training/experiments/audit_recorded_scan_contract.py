"""Read-only bag audit of native scan geometry and acquisition-time metadata.

This does not infer the driver's true acquisition clock or absolute pose error.
Run with the ROS 2 Python environment; no ROS nodes or publishers are created.
"""
import argparse
import json
import sqlite3
from pathlib import Path

import numpy as np


def percentiles(values):
    return np.percentile(values, [5, 50, 95]).tolist() if len(values) else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recording', type=Path, required=True)
    parser.add_argument('--hardware', type=Path, required=True)
    parser.add_argument('--limit', type=int, default=500)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.limit < 2:
        parser.error('--limit must be at least 2')
    from rclpy.serialization import deserialize_message
    from sensor_msgs.msg import LaserScan

    lidar = json.loads(args.hardware.read_text())['lidar']
    database = (args.recording / 'rosbag/rosbag_0.db3').resolve()
    result = dict(scope='First messages per LaserScan topic; metadata and angular coverage audit only',
                  limit_per_topic=args.limit, recording=str(args.recording.resolve()),
                  model_lidar=lidar, topics={},
                  limitations=['Receipt time is not independent acquisition-time truth.',
                               'Header semantics must be verified against the deployed driver.',
                               'Message angular extent is not evidence of usable rear coverage.'])
    with sqlite3.connect(database.as_uri() + '?mode=ro', uri=True) as connection:
        topics = list(connection.execute("SELECT id,name FROM topics WHERE type='sensor_msgs/msg/LaserScan'"))
        for topic_id, topic_name in topics:
            total_messages = connection.execute('SELECT COUNT(*) FROM messages WHERE topic_id=?', (topic_id,)).fetchone()[0]
            metadata = []
            counts = {name: dict(samples=0, finite_hits=0, positive_inf=0, nan=0, other=0)
                      for name in ['model_angular_sector', 'outside_model_angular_sector']}
            query = 'SELECT timestamp,data FROM messages WHERE topic_id=? ORDER BY timestamp LIMIT ?'
            for receipt_ns, data in connection.execute(query, (topic_id, args.limit)):
                message = deserialize_message(data, LaserScan)
                stamp = message.header.stamp.sec + message.header.stamp.nanosec * 1e-9
                receipt = receipt_ns * 1e-9
                ranges = np.asarray(message.ranges, float)
                angle = message.angle_min + np.arange(len(ranges)) * message.angle_increment
                angle = (angle + np.pi) % (2 * np.pi) - np.pi
                inside = (angle >= lidar['angle_min']) & (angle <= lidar['angle_max'])
                for name, mask in [('model_angular_sector', inside), ('outside_model_angular_sector', ~inside)]:
                    values = ranges[mask]
                    counts[name]['samples'] += len(values)
                    counts[name]['finite_hits'] += int((np.isfinite(values) & (values >= message.range_min)
                                                        & (values < message.range_max)).sum())
                    counts[name]['positive_inf'] += int(np.isposinf(values).sum())
                    counts[name]['nan'] += int(np.isnan(values).sum())
                span = max(len(ranges) - 1, 0) * message.time_increment
                metadata.append([stamp, receipt - stamp, message.scan_time, message.time_increment,
                                 len(ranges), span, stamp + span - receipt,
                                 message.range_min, message.range_max, message.angle_min, message.angle_max])
            if not metadata:
                continue
            rows = np.asarray(metadata)
            delta = np.diff(rows[:, 0])
            for count in counts.values():
                count['other'] = count['samples'] - count['finite_hits'] - count['positive_inf'] - count['nan']
                for key in ['finite_hits', 'positive_inf', 'nan', 'other']:
                    count[key + '_fraction'] = count[key] / max(count['samples'], 1)
            result['topics'][topic_name] = dict(
                frames=len(rows), recorded_messages=total_messages,
                all_recorded_messages_audited=len(rows) == total_messages,
                header_interval_p05_p50_p95_s=percentiles(delta[delta > 0]),
                receive_age_p05_p50_p95_s=percentiles(rows[:, 1]),
                scan_time_min_max_s=[float(rows[:, 2].min()), float(rows[:, 2].max())],
                time_increment_min_max_s=[float(rows[:, 3].min()), float(rows[:, 3].max())],
                beam_count_min_max=[int(rows[:, 4].min()), int(rows[:, 4].max())],
                beam_span_min_max_s=[float(rows[:, 5].min()), float(rows[:, 5].max())],
                end_minus_receipt_assuming_first_beam_p05_p50_p95_s=percentiles(rows[:, 6]),
                end_after_receipt_assuming_first_beam_fraction=float((rows[:, 6] > 0).mean()),
                first_range_limits_m=rows[0, 7:9].tolist(),
                first_angular_limits_rad=rows[0, 9:11].tolist(),
                backward_headers=int((delta < 0).sum()), angular_regions=counts)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
