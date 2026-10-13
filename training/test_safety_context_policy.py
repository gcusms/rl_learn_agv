import copy
import json
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
import unittest
import weakref

import gymnasium as gym
import numpy as np
import torch

from navrl.geometry import ROOT, LidarProfile
from navrl.map import StaticMap
from neural_navigation.context_policy import warm_start
from neural_navigation.guarded_curriculum import GuardedNavigationCurriculum
from neural_navigation.context_curriculum import ContextNavigationCurriculum
from neural_navigation.runtime import EmergencyStop, SensorNavigationRuntime
from neural_navigation.sensors import LidarSpec
from neural_navigation.safety_context import (SafetyContext, PROBES, REASONS,
    REASON_START, OBSERVATION_DIM, contract, validate_contract,LEGACY_SCHEMA,
    SCHEMA, RECOVERY_SCHEMA, RECOVERY_PROBES)


class SafetyContextTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def core(self):
        lidar = LidarProfile()
        return SimpleNamespace(scan=np.full(lidar.beams, 2.), lidar=lidar,
            odom=np.zeros(3), elapsed=1., estimate=np.zeros(3),
            localizer=SimpleNamespace(std=np.array([.02, .02, .01])),
            odometry_sanity_failure=False,
            static_map=SimpleNamespace(free=np.ones((1, 1)), lookup=lambda *a, **k: True),
            diagnostics=lambda: {'route_complete': True})

    def test_observes_actual_action_reason_and_wait_without_mutating_guard(self):
        core = self.core()
        guard = EmergencyStop(memory_config={'near_range_m': 1.5})
        guard.memory.entries[(-4, 5)] = (np.array([-.15, .22]), .8)
        guard.reverse_started = .2
        before = copy.deepcopy(guard)
        context = SafetyContext()
        context.record(dict(timestamp=1., proposed_action=[-.8, .4], command=[0., .2],
                            safety_override=True, stop_reason='reverse_without_rear_visibility'))
        out = context.observation(core, guard, 1.)
        np.testing.assert_allclose(out[10:12], [-.8, .4])
        np.testing.assert_allclose(out[12:14], [0., .4])
        self.assertEqual(out[14], 1.)
        self.assertEqual(out[REASON_START+REASONS.index('reverse_without_rear_visibility')], 1.)
        self.assertEqual(out[-1], 0.)  # Executed rotation is not a full stop.
        for i, command in enumerate(PROBES):
            executed, _ = copy.deepcopy(before).check(core, command, 1., 1.)
            self.assertEqual(out[i], float(np.allclose(executed, command, atol=1e-9, rtol=0)))
        self.assertEqual(guard.reverse_started, .2)
        self.assertIsNone(guard.memory.last_time)
        np.testing.assert_array_equal(guard.memory.entries[(-4, 5)][0], before.memory.entries[(-4, 5)][0])
        context.record(dict(timestamp=1., proposed_action=[.5, 0.], command=[0., 0.],
                            safety_override=True, stop_reason='laser_braking_footprint'))
        core.elapsed = 4.
        out = context.observation(core, guard, 4., received_timestamp=5.)
        np.testing.assert_array_equal(out[:9], np.zeros(9))
        self.assertAlmostEqual(out[-1], .6, places=6)
        self.assertEqual(out[REASON_START+REASONS.index('laser_braking_footprint')], 1.)
        with self.assertRaises(ValueError): context.observation(core, guard, 3.)

    def test_unknown_reasons_and_contract_changes_are_explicit(self):
        context = SafetyContext()
        context.record(dict(timestamp=1., proposed_action=[0., 0.], command=[0., 0.],
                            safety_override=True, stop_reason='future_guard_reason'))
        self.assertEqual(context.last_reason, 'other')
        validate_contract(contract())
        bad = contract(); bad['probes_m_s_rad_s'][0][0] = .3
        with self.assertRaises(ValueError): validate_contract(bad)

    def test_versioned_recovery_probes_cover_real_missed_escape_commands(self):
        old=contract(SCHEMA);new=contract(RECOVERY_SCHEMA)
        self.assertEqual(old['dimensions'],new['dimensions'])
        self.assertEqual(old['probes_m_s_rad_s'],PROBES.tolist())
        self.assertEqual(new['probes_m_s_rad_s'],RECOVERY_PROBES.tolist())
        self.assertEqual(old['reasons'],new['reasons'])
        validate_contract(new)
        mislabeled=dict(new);mislabeled['schema']=SCHEMA
        with self.assertRaises(ValueError):validate_contract(mislabeled)
        path=ROOT/'runs/safety_context_20261011/recovery_reward_16000/stall_dense_feasibility_audit.json'
        audit=json.loads(path.read_text())
        for snapshot in audit['reports'][0]['feasibility_snapshots']:
            checks={tuple(c['proposal']):c['fully_accepted'] for c in snapshot['additional_grid']}
            self.assertFalse(any(checks.get(tuple(p),False) for p in PROBES))
            self.assertTrue(any(checks[tuple(p)] for p in RECOVERY_PROBES))
        core=self.core();guard=EmergencyStop()
        context=SafetyContext(RECOVERY_SCHEMA)
        observed=context.observation(core,guard,core.elapsed)
        for i,proposal in enumerate(RECOVERY_PROBES):
            actual,_=copy.deepcopy(guard).check(core,proposal,core.elapsed,core.elapsed)
            self.assertEqual(observed[i],float(np.allclose(actual,proposal,atol=1e-9,rtol=0)))

    def test_v2_wait_includes_tiny_projected_and_voluntary_idle_but_not_arrival(self):
        core=self.core();guard=EmergencyStop()
        modern=SafetyContext();legacy=SafetyContext(LEGACY_SCHEMA)
        result=dict(timestamp=1.,proposed_action=[.5,.006],command=[0.,.003],
                    safety_override=True,stop_reason='learned_route_incomplete')
        modern.record(result);legacy.record(result);core.elapsed=4.
        self.assertAlmostEqual(modern.observation(core,guard,4.)[-1],.6,places=6)
        self.assertEqual(legacy.observation(core,guard,4.)[-1],0.)
        modern.record(dict(timestamp=4.,proposed_action=[0.,0.],command=[0.,0.],safety_override=False,stop_reason=None))
        core.elapsed=5.;self.assertAlmostEqual(modern.observation(core,guard,5.)[-1],.8,places=6)
        modern.record(dict(timestamp=5.,proposed_action=[0.,0.],command=[0.,0.],safety_override=False,stop_reason='estimated_goal_reached'))
        self.assertEqual(modern.observation(core,guard,5.)[-1],0.)
        validate_contract(contract(LEGACY_SCHEMA))

    def test_real_stall_snapshot_exposes_the_verified_forward_turn(self):
        root = ROOT/'runs/optimization_stage3_20261011'
        data = json.loads((root/'stop_go_60k_trace/motion_profile_015.json').read_text())
        hardware = json.loads((root/'portable_clearance_60k/hardware.json').read_text())
        sample = data['samples'][200]
        scene = ROOT/data['case'].get('scene', 'assets/room_navigation_scripted')
        core = SimpleNamespace(elapsed=sample['timestamp'], odom=np.array(sample['odometry']),
            estimate=np.array(sample['estimate']), scan=np.array([np.nan if v is None else v for v in sample['scan']]),
            lidar=LidarSpec(hardware['lidar']), localizer=SimpleNamespace(std=np.array(sample['diagnostics']['localization_std'])),
            odometry_sanity_failure=sample['diagnostics']['odometry_sanity_failure'],
            static_map=StaticMap(scene/'decision_map.yaml', radius=.3), diagnostics=lambda: sample['diagnostics'])
        guard = EmergencyStop(hardware['robot_footprint_m'], memory_config=hardware['near_field_memory'])
        for entry in sample['guard_memory']:
            point = np.array(entry['point'])
            key = tuple(np.floor(point/guard.memory.config.get('resolution_m', .04)).astype(int))
            guard.memory.entries[key] = (point, entry['last_seen'])
        context = SafetyContext().observation(core, guard, core.elapsed)
        self.assertEqual(context[5], 1.)  # v=.1, w=.25 really passes this guard.
        self.assertEqual(context[3], 0.)  # Saturated stationary left turn fails.

    def test_warm_start_preserves_mean_but_restores_exploration_and_context_gradients(self):
        class Dummy(gym.Env):
            observation_space = gym.spaces.Box(-1, 1, (OBSERVATION_DIM,), np.float32)
            action_space = gym.spaces.Box(-1, 1, (2,), np.float32)
            def reset(self, seed=None, options=None): return np.zeros(OBSERVATION_DIM, np.float32), {}
            def step(self, action): return np.zeros(OBSERVATION_DIM, np.float32), 0., False, False, {}
        model, evidence = warm_start(ROOT/'runs/optimization_stage3_20261011/guarded_parent/policy.zip', Dummy(), 93000)
        self.assertLess(evidence['max_initial_mean_action_error'], 2e-6)
        np.testing.assert_allclose(model.policy.log_std.detach().exp().numpy(), [.3, .3], atol=1e-7)
        obs = torch.zeros(1, OBSERVATION_DIM); obs[0, 235:] = 1.
        model.policy.get_distribution(obs).distribution.mean.sum().backward()
        gradient = model.policy.mlp_extractor.policy_net[0].weight.grad[:, 235:]
        self.assertGreater(float(gradient.abs().sum()), 0.)

    def test_training_and_exported_runtime_observations_and_commands_match(self):
        for schema in (SCHEMA, RECOVERY_SCHEMA):
            with self.subTest(schema=schema):
                self.assert_training_and_runtime_observations_match(schema)

    def assert_training_and_runtime_observations_match(self,schema):
        parent = ROOT/'runs/optimization_stage3_20261011/portable_parent_near_memory'
        with tempfile.TemporaryDirectory() as directory:
            bundle = Path(directory)
            for name in ['localizer.pt', 'router.pt', 'hardware.json']:
                shutil.copy2(parent/name, bundle/name)
            hardware = json.loads((bundle/'hardware.json').read_text())
            hardware['policy_observation'] = contract(schema)
            (bundle/'hardware.json').write_text(json.dumps(hardware))
            actor = torch.nn.Linear(OBSERVATION_DIM, 2)
            torch.jit.trace(actor, torch.zeros(1, OBSERVATION_DIM)).save(str(bundle/'actor.pt'))
            env = GuardedNavigationCurriculum(bundle, stage='static')
            env.scenes = [ROOT/'assets/room_navigation_scripted']
            try:
                observation, _ = env.reset(seed=90001)
                source = json.loads(env.training_case_json); e = env.active
                runtime = SensorNavigationRuntime(bundle, e.scene_dir/'decision_map.yaml',
                    source['options']['start'], source['options']['goal'], 'shortcut')
                class Capture:
                    def predict(self, obs):
                        self.observation = obs.copy()
                        return self.proposed
                capture = Capture(); runtime.policy = capture
                for proposed in [np.array([-.8, .4]), np.array([.6, .2]), np.array([0., -.4])]:
                    capture.proposed = proposed
                    result = runtime.step(e.scan, e.observed_odometry, e.measured_velocity, e.elapsed,
                                          capture_observation=True)
                    np.testing.assert_array_equal(capture.observation, observation)
                    np.testing.assert_array_equal(result['policy_observation'],capture.observation)
                    self.assertTrue(env.observation_space.contains(observation))
                    observation, reward, _, _, info = env.step(proposed)
                    np.testing.assert_array_equal(info['commanded_velocity'], result['command'])
                    np.testing.assert_allclose(env.context.last_executed, runtime.context.last_executed, atol=0, rtol=0)
                    self.assertEqual(env.context.last_reason, runtime.context.last_reason)
                    self.assertTrue(np.isfinite(reward))
            finally:
                env.close()

    def test_failure_rehearsal_randomizes_training_cases_and_rejects_external_maps(self):
        parent = ROOT/'runs/optimization_stage3_20261011/guarded_parent'
        env = ContextNavigationCurriculum(parent, stage='robust')
        try:
            _, _ = env.reset(seed=94000)
            template = json.loads(env.training_case_json)
            template['origin'] = 'unit-test training generator'
            _, _ = env.reset(seed=94001, options={'training_template': template})
            generated = json.loads(env.training_case_json)
            self.assertNotEqual(generated['seed'], template['seed'])
            self.assertNotEqual(generated['options']['start'], template['options']['start'])
            self.assertEqual(generated['failure_template_origin'], template['origin'])
            self.assertEqual(generated['scene'], template['scene'])
            bad = copy.deepcopy(template); bad['scene'] = '/not/a/training/map'
            with self.assertRaises(ValueError): env.reset(seed=94002, options={'training_template': bad})
        finally:
            env.close()

    def test_episode_reset_releases_previous_heavy_simulation(self):
        env=ContextNavigationCurriculum(ROOT/'runs/optimization_stage3_20261011/guarded_parent',stage='robust')
        try:
            references=[]
            for seed in [96000,96001,96002,96003]:
                if env.active is not None:references.append(weakref.ref(env.active))
                env.reset(seed=seed)
                self.assertTrue(all(reference() is None for reference in references))
        finally:env.close()


if __name__ == '__main__':
    unittest.main(verbosity=2)
