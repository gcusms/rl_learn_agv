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
import mujoco.viewer
import numpy as np
import torch

from navrl.geometry import ROOT
from navrl.visualization import NavigationOverlay
from neural_navigation.runtime import SensorNavigationRuntime
from neural_navigation.stress_env import StressNavigationEnv


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
    a = p.parse_args()
    torch.set_num_threads(1)
    case = next(c for c in json.loads(a.manifest.read_text())['cases'] if c['id'] == a.case)
    bundles = [a.bundle.resolve(), a.baseline.resolve()]
    scene = ROOT / case.get('scene', 'assets/room_navigation_scripted')
    env = StressNavigationEnv(case=case, components=bundles[0], scene_dir=scene,
                              route_mode='shortcut', max_steps=700)
    a.output.mkdir(parents=True, exist_ok=True)
    events = SimpleQueue()
    def key(code):
        events.put(code)
    def reset(index):
        env.reset(seed=case['seed'], options={k: case[k] for k in
                  ['start', 'goal', 'initial_prior_offset'] if k in case})
        prior = np.asarray(case['start']) + np.asarray(case.get('initial_prior_offset', [0, 0, 0]))
        return SensorNavigationRuntime(bundles[index], scene/'decision_map.yaml',
                                       prior, case['goal'], 'shortcut')
    index = 0
    try:
        runtime = reset(index)
        overlay = NavigationOverlay(env.model, env.lidar, runtime.core.path, case['goal'])
        with mujoco.viewer.launch_passive(env.model, env.data, key_callback=key) as viewer:
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            viewer.cam.lookat[:] = [5, 4, .4]
            viewer.cam.distance = 13
            viewer.cam.azimuth = 90
            viewer.cam.elevation = -65
            paused = False
            finished = False
            result = None
            state = 'RUNNING'
            steps = 0
            next_step = 0.
            print('MUJOCO SENSOR PREVIEW READY: Space pause; B switch NEW/OLD and reset; R reset.', flush=True)
            while viewer.is_running():
                start = time.monotonic()
                while not events.empty():
                    code = events.get()
                    if code == 32:
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
                if not paused and not finished and start >= next_step:
                    next_step = start + env.dt
                    with viewer.lock():
                        result = runtime.step(env.scan, env.observed_odometry,
                                              env.measured_velocity, env.elapsed)
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
                    f'{"NEW 300k candidate" if index == 0 else "OLD safer parent"}\n{case["id"]}\n'
                    f'{"PAUSED" if paused else state}\n{env.elapsed:.1f}s / {env.travel:.2f}m\n'
                    f'{env.collision_events}\n{cmd[0]:.3f} / {cmd[1]:.3f}\n'
                    f'{result["stop_reason"] if result else "initial"}\n'
                    'Space pause | B switch model + reset | R reset'))
                viewer.sync()
                time.sleep(max(0., 1/30-(time.monotonic()-start)))
    finally:
        env.close()


if __name__ == '__main__':
    main()
