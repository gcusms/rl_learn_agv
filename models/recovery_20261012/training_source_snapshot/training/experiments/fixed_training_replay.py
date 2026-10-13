"""Recreate recorded training episodes, retaining the deployed sensor core.

Initial simulator placement is a training reset. Actor inputs are recreated
from scans/odometry, never substituted with the recorded true pose.
"""
import gc
import json
from pathlib import Path
import gymnasium as gym
import numpy as np
from navrl.map import StaticMap
from neural_navigation.training_distance import TrainingDistanceGraph
from neural_navigation.stress_env import StressNavigationEnv
from neural_navigation.core import NavigationCore
from neural_navigation.runtime import EmergencyStop
from neural_navigation.safety_context import SafetyContext


def reset_training_episode(env, case):
    scene = Path(case['scene'])
    if scene not in env.scenes or case.get('stage') != 'robust':
        raise ValueError('Existing robust training layout required')
    options = case['options']
    start, goal = np.asarray(options['start']), np.asarray(options['goal'])
    if start.shape != (3,) or goal.shape != (2,) or not np.isfinite(np.r_[start, goal]).all():
        raise ValueError('Finite original training task required')
    gym.Env.reset(env, seed=int(case['seed']))
    if env.active is not None:
        env.active.close(); env.active = None
    if hasattr(env, 'core'):
        env.core = None
    gc.collect()
    if scene not in env.maps:
        env.maps[scene] = StaticMap(scene/'decision_map.yaml', radius=.3)
        env.graphs[scene] = TrainingDistanceGraph(env.maps[scene])
    env.field = env.graphs[scene].field(goal)
    env.case, env.scene = dict(case['interventions']), scene
    env.active = StressNavigationEnv(case=env.case, components=env.components,
        scene_dir=scene, route_mode=env.route_mode, max_steps=env.max_steps)
    e = env.active
    _, info = e.reset(seed=int(case['seed']), options=options)
    env.training_case_json = json.dumps(case)
    env.previous_distance = env.field.value(e.truth[:2])  # reward only
    env.previous_action = np.zeros(2); env.reverse_seconds = 0.
    prior = start+np.asarray(options.get('initial_prior_offset', [0., 0., 0.]))
    env.core = getattr(env, 'core_factory', NavigationCore)(e.static_map, e.lidar, env.components, env.route_mode)
    env.core.reset(prior, e.goal)
    env.guard = EmergencyStop(e.scene['robot_collision_box_m'][:2],
        memory_config=env.hardware.get('near_field_memory'))
    obs, _ = env.core.update(e.scan, e.observed_odometry, e.measured_velocity, e.elapsed)
    env.guard_overrides = 0; env.guard_steps = 0; env.previous_executed = np.zeros(2)
    env.context = SafetyContext(env.hardware['policy_observation']['schema'])
    obs = env.context.augment(obs, env.core, env.guard, e.elapsed)
    env.rejection_cost.reset(); env.previous_observation = obs.copy()
    return obs, info
