"""Record actual sensor-driven rollouts and render a synchronized MuJoCo comparison.

Run from the repository root with MUJOCO_GL=egl .venv/bin/python
training/record_recovery_comparison.py --record --render. Truth is recorded only
for scoring/display; it is never supplied to a policy or portable runtime.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import time

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from stable_baselines3 import PPO
import torch

from experiments.time_consistent_tracking import TimeConsistentFreshRuntime
from navrl.geometry import ROOT
from navrl.visualization import NavigationOverlay
from neural_navigation.pick_place_env import PickPlaceEnv
from preview_sensor_navigation import draw_checker_floor, sensor_control_step

REPO = ROOT.parent
RELEASE = REPO / 'models/recovery_20261012'
SCENE = ROOT / 'assets/room_navigation_manual_two'
START, GOAL, COLUMN = [1.2, 1.2, 0.], [3.8, 1.2], [2.2, 1.2]
METHODS = [
    ('clone_100k', '首次克隆 · PPO 100k', ROOT/'runs/remote_room_100k', False),
    ('parent_8k', '恢复训练 · 8k 父模型', RELEASE/'baseline_portable', False),
    ('latest_10k', '本轮追加 10k · 原路线', RELEASE/'portable', False),
    ('latest_efficient', '同一最新模型 · 路线优化', RELEASE/'portable', True),
]


def world(bundle):
    env = PickPlaceEnv(components=bundle, scene_dir=SCENE, route_mode='shortcut', max_steps=700)
    env.reset(seed=88000, options={'start': START, 'goal': GOAL})
    assert env.pick_up(0) and env.place(COLUMN), 'Obstacle placement failed'
    obs = env._observation(env.scan, initialize=True)
    return env, obs


def record(directory):
    directory.mkdir(parents=True, exist_ok=True)
    results = {}
    for name, title, bundle, efficient in METHODS:
        env, obs = world(bundle)
        legacy = name == 'clone_100k'
        policy_file = bundle/'policy_100000.zip' if legacy else bundle/'actor.pt'
        policy = PPO.load(policy_file, device='cpu') if legacy else None
        runtime = None if legacy else TimeConsistentFreshRuntime(
            bundle, SCENE/'decision_map.yaml', START, GOAL, 'shortcut')
        if runtime:
            runtime.core.efficient_route_selection = efficient
        samples, control_ms = [], []
        state = 'timeout'
        command = [0., 0.]

        def sample():
            path = env.path if legacy else runtime.core.path
            samples.append(dict(time_s=float(env.elapsed), qpos=env.data.qpos.tolist(),
                qvel=env.data.qvel.tolist(), truth=env.truth.tolist(),
                scan=[float(x) if np.isfinite(x) else None for x in env.scan],
                planned_path=np.asarray(path).tolist(), travel_m=float(env.travel),
                actual_velocity=env.measured_velocity.tolist(), command=list(map(float, command))))

        sample()
        try:
            for step in range(700):
                tick = time.perf_counter()
                if legacy:
                    action, _ = policy.predict(obs, deterministic=True)
                    command = env._velocity_command(action)
                else:
                    result = sensor_control_step(runtime, env)
                    command = result['command']
                    if result['stop_reason'] == 'estimated_goal_reached':
                        state = 'arrived' if np.linalg.norm(env.truth[:2]-GOAL) <= .2 else 'false_arrival'
                        break
                    action = np.array([command[0]/(.2 if command[0]>=0 else .08), command[1]/.5])
                control_ms.append((time.perf_counter()-tick)*1000)
                obs, _, done, truncated, _ = env.step(action)
                sample()
                if env.collision_events or env.hard_guard_activations:
                    state = 'collision_or_physical_guard'
                    break
                if legacy and done:
                    state = 'arrived' if np.linalg.norm(env.truth[:2]-GOAL) <= .2 else 'false_arrival'
                    break
                if step % 100 == 0:
                    print(name, 'step', step, 'sim_seconds', round(env.elapsed, 1), flush=True)
            outcome = dict(state=state, seconds=float(env.elapsed), travel_m=float(env.travel),
                collisions=int(env.collision_events), physical_guards=int(env.hard_guard_activations),
                goal_distance_m=float(np.linalg.norm(env.truth[:2]-GOAL)),
                control_ms_p95=float(np.percentile(control_ms, 95)),
                efficient_routes=efficient, policy_sha256=hashlib.sha256(policy_file.read_bytes()).hexdigest())
            results[name] = outcome
            (directory/f'{name}.json').write_text(json.dumps(dict(samples=samples, outcome=outcome)))
            print('RESULT', name, json.dumps(outcome), flush=True)
        finally:
            env.close()
    metadata = dict(seed=88000, start=START, goal=GOAL, obstacle=COLUMN,
        scene='training/assets/room_navigation_manual_two', decision_period_s=.2,
        speed_limit_m_s=.2, yaw_limit_rad_s=.5, laser_beams=220, laser_fov_degrees=219.44,
        methods=results, deployment_approved=False,
        scope='One static-obstacle demonstration, not aggregate validation or optimality proof.',
        fairness='Same physical scene/seed/start/goal. Clone uses its original 235D continuous actor with current environment sensor features; new models use 300D masked portable runtimes. This is a pipeline comparison, not an isolated training-step comparison. The two latest panes differ only by route-efficiency flag. No legacy Nav2/TEB run is included.')
    (REPO/'docs/results/recovery_video_20261013.json').write_text(json.dumps(metadata, indent=2))


def render(directory, output, speed=4., fps=20):
    traces = [json.loads((directory/f'{x[0]}.json').read_text()) for x in METHODS]
    fontfile = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
    small, normal, large = [ImageFont.truetype(fontfile, n) for n in (16, 22, 30)]
    envs, renderers, overlays = [], [], []
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = [2.65, 1.8, .15]
    camera.distance = 6.3
    camera.azimuth = 90
    camera.elevation = -75
    for _, _, bundle, _ in METHODS:
        env, _ = world(bundle)
        env.model.vis.global_.offwidth = 768
        env.model.vis.global_.offheight = 432
        envs.append(env)
        renderers.append(mujoco.Renderer(env.model, height=432, width=768))
        overlays.append(NavigationOverlay(env.model, env.lidar, [], GOAL))
    end = max(t['outcome']['seconds'] for t in traces)
    count = math.ceil(end/speed*fps)
    output.parent.mkdir(parents=True, exist_ok=True)
    times = [np.array([s['time_s'] for s in t['samples']]) for t in traces]
    colors = ['#acbdce', '#e8b275', '#79c7ed', '#7cdda6']
    labels = dict(arrived='安全到达', timeout='超时', false_arrival='误判到达',
                  collision_or_physical_guard='碰撞／物理保护')
    try:
        with imageio.get_writer(output, fps=fps, codec='libx264', macro_block_size=16,
                ffmpeg_params=['-crf', '25', '-preset', 'fast', '-threads', '2', '-movflags', '+faststart']) as video:
            for frame_index in range(count+1):
                clock = min(end, frame_index*speed/fps)
                frame = Image.new('RGB', (1536, 1088), '#101820')
                d = ImageDraw.Draw(frame)
                d.text((18, 8), f'真实 MuJoCo 导航对比  |  同步 {speed:g} 倍速  |  模拟时间 {clock:.1f} 秒', font=large, fill='white')
                d.text((18, 48), '同起点／目标／静态柱子 · 青色规划、橙色实走、绿色激光 · 仅一次演示，未通过部署验收', font=small, fill='#bbcbdc')
                for k, trace in enumerate(traces):
                    idx = max(0, int(np.searchsorted(times[k], clock, side='right'))-1)
                    s, outcome = trace['samples'][idx], trace['outcome']
                    env, renderer, overlay = envs[k], renderers[k], overlays[k]
                    env.data.qpos[:] = s['qpos']; env.data.qvel[:] = s['qvel']
                    mujoco.mj_forward(env.model, env.data)
                    renderer.update_scene(env.data, camera=camera)
                    draw_checker_floor(renderer.scene)
                    overlay.path = np.asarray(s['planned_path'])
                    overlay.draw(renderer.scene, env.data, s['truth'], np.array([np.inf if x is None else x for x in s['scan']]))
                    x, y = (k % 2)*768, 80+(k//2)*496
                    frame.paste(Image.fromarray(renderer.render()), (x, y+64))
                    finished = clock >= outcome['seconds']
                    status = labels[outcome['state']] if finished else '导航中'
                    d.text((x+12, y+2), METHODS[k][1], font=normal, fill=colors[k])
                    d.text((x+12, y+31), f"{status}  |  {s['time_s']:.1f} 秒  |  {s['travel_m']:.2f} 米  |  碰撞 {outcome['collisions'] if finished else 0}", font=small, fill='white')
                video.append_data(np.asarray(frame))
                if frame_index in (0, count//2, count):
                    frame.save(directory/f'frame_{frame_index}.jpg')
                if frame_index == count//2:
                    frame.save(output.with_suffix('.jpg'))
                if frame_index % 100 == 0:
                    print('FRAME', frame_index, '/', count, flush=True)
            card = Image.new('RGB', (1536, 1088), '#101820')
            d = ImageDraw.Draw(card)
            d.text((70, 90), '本次实录结果（未对失败结果进行筛选）', font=large, fill='white')
            for k, trace in enumerate(traces):
                o = trace['outcome']; y = 210+k*140
                d.text((70, y), METHODS[k][1], font=large, fill=colors[k])
                d.text((70, y+48), f"{labels[o['state']]}  |  {o['seconds']:.1f} 秒  |  {o['travel_m']:.2f} 米  |  碰撞 {o['collisions']}  |  物理保护 {o['physical_guards']}", font=normal, fill='white')
            d.text((70, 830), '旧模型与新模型接口不同；下方两组仅改变路线优化开关。', font=normal, fill='#bbccdd')
            d.text((70, 880), '不是 TEB 对比，不证明全局最短路、动态避障成功率或实车安全。', font=normal, fill='#bbccdd')
            d.text((70, 930), '最新专项评估仍为 2/5 到达、1 碰撞、2 超时；仍需继续修复。', font=normal, fill='#bbccdd')
            card.save(directory/'results.jpg')
            for _ in range(6*fps):
                video.append_data(np.asarray(card))
    finally:
        for renderer in renderers: renderer.close()
        for env in envs: env.close()
    print('SAVED', output, flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--record', action='store_true')
    p.add_argument('--render', action='store_true')
    p.add_argument('--directory', type=Path, default=ROOT/'runs/recovery_video_20261013')
    p.add_argument('--output', type=Path, default=REPO/'docs/media/recovery-comparison-20261013.mp4')
    a = p.parse_args()
    torch.set_num_threads(1)
    if a.record: record(a.directory)
    if a.render: render(a.directory, a.output)


if __name__ == '__main__': main()
