"""Real-time MuJoCo preview of the evaluated portable sensor runtime.

Truth is used only for display/scoring. B resets the same seeded world and
switches bundles; simulated laser, odometry and velocity feed the controller.
"""
import argparse
import json
from pathlib import Path
from queue import SimpleQueue
import time

import mujoco
import numpy as np
import torch

from navrl.geometry import ROOT
from navrl.visualization import NavigationOverlay
from neural_navigation.runtime import SensorNavigationRuntime
from neural_navigation.stress_env import StressNavigationEnv


def sensor_control_step(runtime, env):
    """Use sensor odometry in both scripted-fault and manual world previews."""
    odometry = env.observed_odometry if hasattr(env, 'observed_odometry') else env.odom
    return runtime.step(env.scan, odometry, env.measured_velocity, env.elapsed)


def draw_checker_floor(scene):
    """Display-only blue/gray tiles inside the existing 10 x 8 m walls."""
    colors = ([.20, .36, .52, 1.], [.55, .60, .65, 1.])
    for x in range(10):
        for y in range(8):
            if scene.ngeom >= scene.maxgeom:
                return
            geom = scene.geoms[scene.ngeom]
            scene.ngeom += 1
            mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_BOX,
                               np.array([.5, .5, .001]),
                               np.array([x+.5, y+.5, .001]),
                               np.eye(3).reshape(-1),
                               np.asarray(colors[(x+y) % 2], dtype=np.float32))
            geom.category = mujoco.mjtCatBit.mjCAT_DECOR


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--bundle', type=Path, required=True)
    p.add_argument('--baseline', type=Path, required=True)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--case', default='independent_dynamic_000')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--interactive-goals', action='store_true',
                   help='Open coordinate controls for sending goals; start paused')
    p.add_argument('--pick-place-obstacles', action='store_true',
                   help='Static columns: double-click to hide/pick up, right-click to place')
    p.add_argument('--efficient-routes', action='store_true',
                   help='Experimental bounded learned-route comparison by full length and initial direction')
    a = p.parse_args()
    torch.set_num_threads(1)
    case = next(c for c in json.loads(a.manifest.read_text())['cases'] if c['id'] == a.case)
    bundles = [a.bundle.resolve(), a.baseline.resolve()]
    scene = ROOT / case.get('scene', 'assets/room_navigation_scripted')
    if a.pick_place_obstacles:
        from neural_navigation.pick_place_env import PickPlaceEnv
        scene = ROOT/'assets/room_navigation_manual_two'
        env = PickPlaceEnv(components=bundles[0], scene_dir=scene,
                           route_mode='shortcut', max_steps=700)
    else:
        env = StressNavigationEnv(case=case, components=bundles[0], scene_dir=scene,
                                  route_mode='shortcut', max_steps=700)
    a.output.mkdir(parents=True, exist_ok=True)
    events = SimpleQueue()
    panel = None
    panel_message = None
    def make_runtime(bundle, prior, goal):
        manifest = json.loads((bundle/'manifest.json').read_text())
        if manifest.get('requires_quality_feature_generator'):
            if manifest.get('sensor_core') == 'time-consistent':
                from experiments.time_consistent_tracking import TimeConsistentFreshRuntime
                factory = TimeConsistentFreshRuntime
            else:
                from experiments.quality_sensor_runtime import QualitySensorNavigationRuntime
                factory = QualitySensorNavigationRuntime
        else:
            factory = SensorNavigationRuntime
        runtime = factory(bundle, scene/'decision_map.yaml', prior, goal, 'shortcut')
        runtime.core.efficient_route_selection = a.efficient_routes
        return runtime
    def key(code):
        events.put(code)
    def reset(index):
        env.reset(seed=case['seed'], options={k: case[k] for k in
                  ['start', 'goal', 'initial_prior_offset'] if k in case})
        prior = np.asarray(case['start']) + np.asarray(case.get('initial_prior_offset', [0, 0, 0]))
        return make_runtime(bundles[index], prior, case['goal'])
    index = 0
    try:
        runtime = reset(index)
        if a.interactive_goals:
            import tkinter as tk
            from tkinter import ttk
            panel = tk.Tk()
            panel.title('MuJoCo 仿真目标点 — 最新模型')
            panel.geometry('440x260')
            ttk.Label(panel, text='输入地图坐标（米），点击发送目标开始导航。').pack(pady=10)
            row = ttk.Frame(panel); row.pack()
            x_value = tk.StringVar(value=str(case['goal'][0]))
            y_value = tk.StringVar(value=str(case['goal'][1]))
            for label, value in [('X', x_value), ('Y', y_value)]:
                ttk.Label(row, text=label).pack(side='left', padx=4)
                ttk.Entry(row, textvariable=value, width=12).pack(side='left')
            panel_message = tk.StringVar(value='等待目标。墙体或障碍物内的目标会被拒绝。')
            def send_goal():
                try:
                    point = np.array([float(x_value.get()), float(y_value.get())])
                    if not np.isfinite(point).all():
                        raise ValueError('坐标必须为有限数字')
                    events.put(('goal', point))
                except ValueError:
                    panel_message.set('请输入有效的 X、Y 数字。')
            ttk.Button(panel, text='发送目标并开始导航', command=send_goal).pack(pady=10)
            buttons = ttk.Frame(panel); buttons.pack()
            ttk.Button(buttons, text='暂停／继续', command=lambda: events.put(32)).pack(side='left', padx=5)
            ttk.Button(buttons, text='重置小车', command=lambda: events.put(82)).pack(side='left', padx=5)
            ttk.Label(panel, textvariable=panel_message, wraplength=410).pack(pady=10)
        overlay = NavigationOverlay(env.model, env.lidar, runtime.core.path, case['goal'])
        if a.pick_place_obstacles:
            from navrl.pick_place_viewer import PickPlaceViewer
            handle = PickPlaceViewer(env, key_callback=key)
        else:
            from mujoco import viewer as passive_viewer
            handle = passive_viewer.launch_passive(env.model, env.data, key_callback=key)
        with handle as viewer:
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            viewer.cam.lookat[:] = [5, 4, .4]
            viewer.cam.distance = 13
            viewer.cam.azimuth = 90
            viewer.cam.elevation = -65
            paused = a.interactive_goals
            finished = False
            result = None
            state = 'RUNNING'
            steps = 0
            next_step = 0.
            last_placement_message = None
            print('MUJOCO SENSOR PREVIEW READY: Space pause; B switch NEW/OLD and reset; R reset.', flush=True)
            while viewer.is_running():
                start = time.monotonic()
                if a.pick_place_obstacles:
                    viewer.poll_events()
                    if panel_message is not None and env.placement_message != last_placement_message:
                        panel_message.set(env.placement_message)
                        last_placement_message = env.placement_message
                if panel is not None:
                    try:
                        panel.update_idletasks(); panel.update()
                    except tk.TclError:
                        break
                while not events.empty():
                    code = events.get()
                    if isinstance(code, tuple) and code[0] == 'goal':
                        if env.collision_events or env.hard_guard_activations:
                            panel_message.set('发生碰撞后请先重置小车。')
                            continue
                        try:
                            with viewer.lock():
                                runtime.core.set_goal(code[1])
                                env.goal = code[1].copy()
                            overlay.goal = code[1].copy()
                            finished = False; paused = False; steps = 0; next_step = 0.
                            state = 'RUNNING'; result = None
                            panel_message.set(f'目标已发送：X={code[1][0]:.2f}, Y={code[1][1]:.2f}')
                        except ValueError:
                            panel_message.set('目标不可用：请选择地图内远离墙体的空闲位置。')
                    elif code == 32:
                        paused = not paused
                    elif code in (66, 82):
                        if code == 66:
                            index = 1-index
                        with viewer.lock():
                            runtime = reset(index)
                        overlay.history.clear()
                        finished = False
                        result = None
                        state = 'RUNNING'
                        steps = 0
                        next_step = 0.
                native_paused = bool(getattr(viewer, 'simulation_paused', False))
                if not paused and not native_paused and not finished and start >= next_step:
                    next_step = start + env.dt
                    with viewer.lock():
                        result = sensor_control_step(runtime, env)
                        command = np.asarray(result['command'])
                        if result['stop_reason'] == 'estimated_goal_reached':
                            finished = True
                            state = ('ARRIVED' if np.linalg.norm(env.truth[:2]-env.goal) <= .2
                                     else 'FALSE ARRIVAL')
                        else:
                            action = [command[0]/(.2 if command[0] >= 0 else .08), command[1]/.5]
                            env.step(np.asarray(action))
                            steps += 1
                            if env.collision_events or env.hard_guard_activations:
                                finished = True
                                state = 'COLLISION / PHYSICAL GUARD'
                            elif steps >= 700:
                                finished = True
                                state = 'TIMEOUT'
                    status = dict(model='NEW' if index == 0 else 'OLD', bundle=str(bundles[index]),
                                  case_id=case['id'], state=state, simulation_seconds=env.elapsed,
                                  travel_m=env.travel, collisions=env.collision_events,
                                  physical_guards=env.hard_guard_activations,
                                  goal_distance_m=float(np.linalg.norm(env.truth[:2]-env.goal)),
                                  command=result['command'], stop_reason=result['stop_reason'],
                                  efficient_routes=a.efficient_routes,
                                  route_candidate_count=getattr(runtime.core,'route_candidate_count',1),
                                  route_length_m=runtime.core.remaining,
                                  deployment_approved=False)
                    if steps % 10 == 0 or finished:
                        (a.output/'status.json').write_text(json.dumps(status, indent=2))
                    if finished:
                        print('RESULT', json.dumps(status), flush=True)
                with viewer.lock():
                    viewer.user_scn.ngeom = 0
                    draw_checker_floor(viewer.user_scn)
                    overlay.path = runtime.core.path
                    overlay.draw(viewer.user_scn, env.data, env.truth, env.scan)
                cmd = result['command'] if result else [0., 0.]
                viewer.set_texts((None, None,
                    'Sensor PPO comparison\nCase\nState\nTime / travel\nCollisions\nCommand v / w\nGuard reason\nControls',
                    f'{"NEW candidate" if index == 0 else "OLD baseline"}\n{case["id"]}\n'
                    f'{"PAUSED" if paused or native_paused else state}\n{env.elapsed:.1f}s / {env.travel:.2f}m\n'
                    f'{env.collision_events}\n{cmd[0]:.3f} / {cmd[1]:.3f}\n'
                    f'{result["stop_reason"] if result else "initial"}\n'
                    + ('Double-click column: pick up | Right-click ground: place | R reset'
                     if a.pick_place_obstacles else 'Space pause | B switch model + reset | R reset')))
                viewer.sync()
                time.sleep(max(0., 1/30-(time.monotonic()-start)))
    finally:
        if panel is not None:
            try: panel.destroy()
            except Exception: pass
        env.close()


if __name__ == '__main__':
    main()
