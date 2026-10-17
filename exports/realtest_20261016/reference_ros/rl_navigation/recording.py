"""ROS-free recording files, bounded asynchronous logging and snapshots."""
import hashlib
import json
import math
import queue
import re
import shutil
import subprocess
import threading
import time
import uuid
from collections import Counter, deque
from datetime import datetime
from pathlib import Path

import yaml


TOPICS = {
    '/scan': 'sensor_msgs/msg/LaserScan',
    '/scan_raw': 'sensor_msgs/msg/LaserScan',
    '/ros_robot_controller/imu_raw': 'sensor_msgs/msg/Imu',
    '/imu_corrected': 'sensor_msgs/msg/Imu', '/imu': 'sensor_msgs/msg/Imu',
    '/command/odom': 'nav_msgs/msg/Odometry', '/wheel/odom': 'nav_msgs/msg/Odometry',
    '/rl_navigation/odom_raw': 'nav_msgs/msg/Odometry', '/odom': 'nav_msgs/msg/Odometry',
    '/wheel/feedback': 'std_msgs/msg/String',
    '/rl_navigation/chassis_status': 'std_msgs/msg/String',
    '/tf': 'tf2_msgs/msg/TFMessage', '/tf_static': 'tf2_msgs/msg/TFMessage',
    '/map': 'nav_msgs/msg/OccupancyGrid',
    '/initialpose': 'geometry_msgs/msg/PoseWithCovarianceStamped',
    '/goal_pose': 'geometry_msgs/msg/PoseStamped',
    '/rl_navigation/model_pose': 'geometry_msgs/msg/PoseStamped',
    '/rl_navigation/path': 'nav_msgs/msg/Path',
    '/rl_navigation/footprint': 'geometry_msgs/msg/PolygonStamped',
    '/rl_navigation/cmd_vel_candidate': 'geometry_msgs/msg/Twist',
    '/rl_navigation/cmd_vel': 'geometry_msgs/msg/Twist',
    '/rl_navigation/model_status': 'std_msgs/msg/String',
    '/rl_navigation/decision': 'std_msgs/msg/String',
    '/rl_navigation/session_metadata': 'std_msgs/msg/String',
    '/parameter_events': 'rcl_interfaces/msg/ParameterEvent',
    '/robot_description': 'std_msgs/msg/String', '/clock': 'rosgraph_msgs/msg/Clock',
    '/rosout': 'rcl_interfaces/msg/Log',
}
LATCHED = {'/map', '/tf_static', '/rl_navigation/session_metadata', '/robot_description'}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''): h.update(chunk)
    return h.hexdigest()


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    temp.replace(path)


def snapshot_map(source, target):
    source, target = Path(source).resolve(), Path(target)
    config = yaml.safe_load(source.read_text())
    image = (source.parent / config['image']).resolve()
    target.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target/'original.yaml')
    image_name = 'map' + image.suffix
    shutil.copy2(image, target/image_name)
    config['image'] = image_name
    (target/'map.yaml').write_text(yaml.safe_dump(config))
    write_json(target/'source.json', {'yaml': str(source), 'image': str(image),
               'yaml_sha256': digest(source), 'image_sha256': digest(image)})


class RecordingStore:
    def __init__(self, output_root, name='', queue_size=10000):
        if name and not re.fullmatch(r'[A-Za-z0-9_-]{1,80}', name):
            raise ValueError('session_name must use 1-80 letters, digits, underscores or hyphens')
        root = Path(output_root).expanduser().resolve()
        root.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.path = root / f'{stamp}_{name + "_" if name else ""}{uuid.uuid4().hex[:8]}'
        self.path.mkdir()
        self.queue = queue.Queue(maxsize=queue_size)
        self.dropped = 0; self.error = None; self.closed = False
        self.started = time.time(); self.monotonic_started = time.monotonic()
        self.session = dict(schema='rl_navigation_recording_v1', state='starting',
            started_unix=self.started, started_local=datetime.now().astimezone().isoformat(),
            expected_topics=list(TOPICS), snapshots=[], warnings=[],
            clock_semantics={'bag': 'recorder receive ROS time', 'header': 'sensor publisher timestamp',
                'jsonl': 'receive_ros_ns + receive_unix_ns + receive_monotonic_ns',
                'twist': 'no header; receive time is not physical execution time'})
        self.states = Counter(); self.state_seconds = Counter(); self.stop_reasons = Counter()
        self.last_state = None; self.last_status_time = None; self.last_status_event = 0.
        self.timings = deque(maxlen=100000); self.ages = deque(maxlen=100000)
        self.decision_count = 0; self.full_observation_count = 0; self.duplicates = 0
        self.route_complete = Counter(); self.eligibility = Counter(); self.goal_count = 0
        self.last_inference = {}; self.snapshotted_runs = set(); self.parameter_hashes = {}
        self.command_count = 0; self.nonzero_command_count = 0
        self.invalid_command_count = 0
        self.zero_command_seconds = 0.; self.last_command = None
        self.streams = {name: (self.path/f'{name}.jsonl').open('w', buffering=1)
                        for name in ('decisions', 'events', 'commands')}
        write_json(self.path/'session.json', self.session)
        self.thread = threading.Thread(target=self._run, name='recording_writer', daemon=True)
        self.thread.start()

    def submit(self, kind, data, ros_ns=0):
        if self.closed: return
        item = (kind, dict(receive_ros_ns=int(ros_ns), receive_unix_ns=time.time_ns(),
                           receive_monotonic_ns=time.monotonic_ns(), data=data))
        try: self.queue.put_nowait(item)
        except queue.Full: self.dropped += 1

    def _line(self, name, payload):
        self.streams[name].write(json.dumps(payload, ensure_ascii=False, allow_nan=False) + '\n')

    def _run(self):
        while True:
            item = self.queue.get()
            try:
                if item is None: return
                kind, envelope = item
                try: self._process(kind, envelope)
                except Exception as exc:
                    # Preserve the exception; the node stops recording on writer errors.
                    self.error = f'{kind}: {type(exc).__name__}: {exc}'
            finally: self.queue.task_done()

    def _process(self, kind, envelope):
        data = envelope['data']; now = envelope['receive_monotonic_ns']/1e9
        if kind == 'decision':
            key = (data['run_id'], data['mission_id'])
            sequence = int(data['inference_id'])
            if sequence <= self.last_inference.get(key, -1):
                self.duplicates += 1; return
            self.last_inference[key] = sequence
            self.decision_count += 1
            if data.get('observation') is not None: self.full_observation_count += 1
            result = data.get('result', {})
            self.stop_reasons[str(result.get('stop_reason'))] += 1
            self.route_complete[str(result.get('diagnostics', {}).get('route_complete'))] += 1
            self.eligibility[str(data.get('eligibility'))] += 1
            latency = result.get('adapter_inference_ms')
            if latency is not None and math.isfinite(latency): self.timings.append(latency)
            self._line('decisions', envelope)
        elif kind == 'status':
            state = data.get('state', 'UNKNOWN')
            self.states[state] += 1
            if self.last_status_time is not None:
                gap = now-self.last_status_time
                # A missing publisher must not be counted as continuous RUNNING.
                if 0 <= gap <= 1.: self.state_seconds[self.last_state] += gap
                else: self.state_seconds['UNOBSERVED_GAP'] += max(0., gap)
            if state != self.last_state or now-self.last_status_event >= 1.:
                self._line('events', dict(envelope, event='model_status'))
                self.last_status_event = now
            age = data.get('result_age_s')
            if age is not None and math.isfinite(age): self.ages.append(age)
            self.last_state = state; self.last_status_time = now
        elif kind == 'command':
            self.command_count += 1
            invalid = any(not isinstance(x,(int,float)) or not math.isfinite(x) for x in data['velocity'])
            self.invalid_command_count += int(invalid)
            nonzero = invalid or any(abs(x) > 1e-6 for x in data['velocity'])
            self.nonzero_command_count += int(nonzero and not invalid)
            if self.last_command:
                old_time, old_nonzero = self.last_command
                if not old_nonzero and 0 <= now-old_time <= .3:
                    self.zero_command_seconds += now-old_time
            self.last_command = (now, nonzero)
            self._line('commands', envelope)
        elif kind == 'metadata':
            self._line('events', dict(envelope, event='session_metadata'))
            run = data['run_id']
            if not re.fullmatch(r'[a-fA-F0-9]{32}', run): raise ValueError('Invalid model run ID')
            if run not in self.snapshotted_runs:
                self._snapshot_model(data)
                self.snapshotted_runs.add(run)
        elif kind == 'parameters':
            key = data['node']
            content = json.dumps(data['values'], sort_keys=True)
            if self.parameter_hashes.get(key) == content: return
            self.parameter_hashes[key] = content
            folder = self.path/'snapshots/config/runtime_parameters'
            write_json(folder/(key.strip('/').replace('/', '__') + '.json'), envelope)
            self._line('events', dict(envelope, event='parameters_snapshot'))
            for name in ('calib_file', 'yaml_filename'):
                value = data['values'].get(name)
                if value and Path(value).is_file():
                    target = self.path/'snapshots/config/referenced'/key.strip('/').replace('/', '__')
                    target.mkdir(parents=True, exist_ok=True)
                    if name == 'yaml_filename': snapshot_map(value, target/'map')
                    else: shutil.copy2(value, target/Path(value).name)
        else:
            if kind == 'goal_pose': self.goal_count += 1
            self._line('events', dict(envelope, event=kind))

    def _snapshot_model(self, data):
        target = self.path/'snapshots/runs'/data['run_id']
        target.mkdir(parents=True)
        write_json(target/'metadata.json', data)
        options = data['options']
        from .model_adapter import resolve_map
        errors = []
        for label, source in [('map', resolve_map(options['map'])), ('prepared_map', data['prepared_map'])]:
            try: snapshot_map(source, target/label)
            except Exception as exc: errors.append(f'{label}: {exc}')
        try:
            bundle = Path(data['model']['bundle'])
            manifest = json.loads((bundle/'manifest.json').read_text())
            model_dir = target/'model'; model_dir.mkdir()
            for name in ['manifest.json', *manifest['files']]:
                if Path(name).name != name: raise ValueError('Unsafe model filename')
                shutil.copy2(bundle/name, model_dir/name)
                expected = manifest['files'].get(name)
                if expected and digest(model_dir/name) != expected: errors.append('Model hash mismatch: '+name)
            shutil.copy2(options['model_config'], target/'model_config.json')
            for source, expected in data.get('loaded_files', {}).items():
                if not Path(source).is_file() or digest(source) != expected:
                    errors.append('Startup file changed or missing: '+source)
            actor_hash = data['model'].get('actor_sha256')
            if actor_hash and digest(model_dir/'actor.pt') != actor_hash:
                errors.append('Actor differs from loaded model identity')
            source = options['model_source']
            def git(*args):
                p = subprocess.run(['git', '-C', source, *args], text=True, capture_output=True, timeout=5)
                return {'returncode': p.returncode, 'output': p.stdout.strip(), 'error': p.stderr.strip()}
            write_json(target/'source_version.json', {'path':source,'head':git('rev-parse','HEAD'),
                'branch':git('branch','--show-current'),'status':git('status','--porcelain')})
        except Exception as exc: errors.append(f'model snapshot: {exc}')
        write_json(target/'snapshot_status.json', {'complete': not errors, 'errors':errors})
        self.session['snapshots'].append({'run_id':data['run_id'],'model':data['model'],
            'map':options['map'],'complete':not errors,'errors':errors})
        self.session['warnings'].extend(errors)

    def close(self, final):
        if self.closed: return
        self.closed = True
        self.queue.put(None); self.thread.join()
        for stream in self.streams.values(): stream.close()
        self.session.update(final)
        self.session.update(ended_unix=time.time(), duration_s=time.monotonic()-self.monotonic_started,
            log_queue_dropped=self.dropped, writer_error=self.error,
            unique_decisions=self.decision_count, full_observation_decisions=self.full_observation_count,
            duplicate_decisions_discarded=self.duplicates, model_status_counts=dict(self.states),
            observed_goal_messages=self.goal_count, decision_route_complete_counts=dict(self.route_complete),
            decision_eligibility_counts=dict(self.eligibility),
            observed_state_seconds=dict(self.state_seconds), decision_stop_reasons=dict(self.stop_reasons),
            published_command_count=self.command_count, nonzero_published_command_count=self.nonzero_command_count,
            invalid_published_command_count=self.invalid_command_count,
            observed_zero_command_seconds=self.zero_command_seconds)
        def stats(values):
            if not values: return None
            ordered = sorted(values)
            return dict(samples=len(ordered), mean=sum(ordered)/len(ordered),
                p95=ordered[round((len(ordered)-1)*.95)], p99=ordered[round((len(ordered)-1)*.99)],
                maximum=ordered[-1], window='latest 100000 samples at most')
        self.session['inference_ms'] = stats(self.timings)
        self.session['published_result_age_s'] = stats(self.ages)
        metadata = self.path/'rosbag/metadata.yaml'
        if metadata.exists():
            bag = yaml.safe_load(metadata.read_text())['rosbag2_bagfile_information']
            counts = {item['topic_metadata']['name']:item['message_count'] for item in bag['topics_with_message_count']}
            self.session['bag_message_counts'] = counts
            duration = bag['duration']['nanoseconds']/1e9
            self.session['bag_duration_s'] = duration
            self.session['bag_average_received_hz'] = {topic:count/duration for topic,count in counts.items()} if duration>0 else {}
            self.session['missing_topics'] = [t for t in TOPICS if not counts.get(t)]
        else:
            self.session['warnings'].append('rosbag metadata missing: bag may require reindexing')
        self.session['snapshot_sha256'] = {str(p.relative_to(self.path)):digest(p)
            for p in (self.path/'snapshots').rglob('*') if p.is_file()}
        if not self.session['snapshots']:
            self.session['warnings'].append('No model metadata: actual model/map snapshot unavailable')
        if self.full_observation_count < self.decision_count:
            self.session['warnings'].append('Some in-flight decisions have no captured observation; inspect observation_available')
        write_json(self.path/'session.json', self.session)
        lines = ['# 导航录制摘要', '', f"- 目录：`{self.path}`",
            f"- 结束原因：{self.session.get('stop_reason')}",
            f'- 录制时长：{self.session["duration_s"]:.1f} 秒',
            f'- 唯一推理：{self.decision_count}；含完整观测：{self.full_observation_count}',
            f'- 录制期间收到目标消息：{self.goal_count}（附加录制前的当前目标见 metadata）',
            f'- 发布到底盘话题的速度消息：{self.command_count}；非零：{self.nonzero_command_count}',
            f'- 日志队列丢弃：{self.dropped}；写入错误：{self.error}',
            '', '## 推理与状态', '',
            '```json', json.dumps({k:self.session[k] for k in ('inference_ms','observed_state_seconds','decision_stop_reasons','decision_route_complete_counts','decision_eligibility_counts')},ensure_ascii=False,indent=2), '```',
            '', '## 数据完整性', '',
            '- 未录到的话题（可包含未启用的 scan_raw、clock 等）：'+', '.join(self.session.get('missing_topics',[])),
            '- 提醒：'+'；'.join(self.session['warnings']),
            '- JSON日志可能比rosbag更早开始接收；bag_ready事件标记原始录制就绪时刻。',
            '', '## 阅读顺序与限制', '',
            '先读 session.json 和本摘要，再查看 decisions.jsonl、events.jsonl、commands.jsonl 与 snapshots；原始消息在 rosbag。',
            'decisions 的 eligibility 表示结果取回时是否具备使用条件，不代表已下发或电机已执行。真实发布见 commands/rosbag，运动反馈见 wheel/feedback。',
            '初始位置、目标、雷达和里程计需要按消息时间对齐；状态时长按接收间隔统计，超过1秒的间隙记为未观测。',
            '本录制不提供独立定位真值，不自动判定物理到达、碰撞或导航成功；请补充人工观察。',
            '回放包含速度话题，应在独立ROS域中分析，避免连接正在运行的底盘。', '']
        (self.path/'summary.md').write_text('\n'.join(lines))
