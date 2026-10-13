"""Smoke check: CPU training, MuJoCo stepping/rays, ROS imports and map/model assets.
This verifies tooling, not a trained AGV policy or sim-to-real readiness.
"""
import argparse
import importlib.metadata as metadata
import json
from pathlib import Path
import time
import xml.etree.ElementTree as ET
import numpy as np
import torch
import gymnasium as gym
import mujoco
import yaml
from PIL import Image
from stable_baselines3 import PPO


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path)
    parser.add_argument('--render', action='store_true')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    torch.set_num_threads(4)
    report = {'versions': {name: metadata.version(name) for name in
              ['torch', 'numpy', 'stable-baselines3', 'gymnasium', 'mujoco', 'tensorboard']},
              'cuda_available': torch.cuda.is_available()}
    xml = """<mujoco><option timestep="0.01"/><worldbody>
    <geom name="floor" type="plane" size="5 5 .1"/>
    <body name="ball" pos="0 0 1"><freejoint/><geom type="sphere" size=".1" mass="1"/></body>
    <body name="obstacle" mocap="true" pos="2 0 .5"><geom type="box" size=".2 .2 .5"/></body>
    </worldbody></mujoco>"""
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    t0 = time.perf_counter()
    for _ in range(1000):
        mujoco.mj_step(model, data)
    assert np.isfinite(data.qpos).all()
    assert abs(data.qpos[2] - .1) < .03, data.qpos
    report['mujoco'] = {'steps': 1000, 'seconds': time.perf_counter() - t0,
                        'settled_ball_height': float(data.qpos[2])}
    geom_id = np.zeros(1, dtype=np.int32)
    ray = lambda: mujoco.mj_ray(model, data, np.array([0., 0., .5]),
                               np.array([1., 0., 0.]), None, 1, -1, geom_id)
    first = ray()
    data.mocap_pos[0, 0] = 3.
    mujoco.mj_forward(model, data)
    second = ray()
    assert np.isclose(first, 1.8) and np.isclose(second, 2.8), (first, second)
    report['mujoco']['moving_obstacle_ray_ranges'] = [first, second]
    if args.render:
        with mujoco.Renderer(model, height=120, width=160) as renderer:
            renderer.update_scene(data)
            pixels = renderer.render()
            assert pixels.shape == (120, 160, 3)
            report['render'] = {'shape': list(pixels.shape)}
    env = gym.make('Pendulum-v1')
    try:
        policy = PPO('MlpPolicy', env, device='cpu', n_steps=64, batch_size=32,
                     n_epochs=2, policy_kwargs={'net_arch': [64, 64]}, seed=0, verbose=0)
        policy.learn(total_timesteps=256)
        obs, _ = env.reset(seed=0)
        action, _ = policy.predict(obs, deterministic=True)
        assert env.action_space.contains(action)
        assert all(torch.isfinite(p).all() for p in policy.policy.parameters())
        report['ppo'] = {'steps': policy.num_timesteps, 'device': str(policy.device),
                         'action': action.tolist()}
    finally:
        env.close()
    try:
        import rclpy
        from geometry_msgs.msg import Twist
        from sensor_msgs.msg import LaserScan
        from nav_msgs.msg import OccupancyGrid
        from ament_index_python.packages import get_package_prefix, PackageNotFoundError
        report['ros'] = {'rclpy': rclpy.__file__, 'packages': {}}
        for name in ['nav2_bringup', 'nav2_amcl', 'robot_localization', 'laser_filters', 'agv_description']:
            try:
                report['ros']['packages'][name] = get_package_prefix(name)
            except PackageNotFoundError:
                report['ros']['packages'][name] = None
    except ImportError as exc:
        report['ros'] = {'import_error': str(exc)}
    robot = ET.parse(root/'agv/simulations/agv_description/urdf/agv_chassis.urdf').getroot()
    base = robot.find("link[@name='base_link']")
    box = base.find('collision/geometry/box').attrib['size']
    mesh_paths = [root/'agv/simulations/agv_description'/m.attrib['filename'].split('package://agv_description/')[-1]
                  for m in robot.findall('.//mesh') if m.attrib['filename'].startswith('package://agv_description/')]
    missing = [str(p) for p in mesh_paths if not p.is_file()]
    assert not missing, missing
    report['urdf'] = {'links': len(robot.findall('link')), 'joints': len(robot.findall('joint')),
                       'base_collision_box_m': box, 'missing_meshes': missing,
                       'base_has_inertial': base.find('inertial') is not None}
    report['maps'] = {}
    for path in sorted((root/'agv/slam/maps').glob('*.yaml')):
        config = yaml.safe_load(path.read_text())
        image_path = path.parent/config['image']
        with Image.open(image_path) as image:
            report['maps'][path.stem] = {'image': str(image_path.relative_to(root)),
                'pixels': list(image.size), 'resolution_m': config['resolution'], 'origin': config['origin']}
    result = json.dumps(report, ensure_ascii=False, indent=2)
    print(result)
    if args.output:
        args.output.write_text(result + '\n')


if __name__ == '__main__':
    main()
