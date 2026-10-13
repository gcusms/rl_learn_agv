import copy
import json
from pathlib import Path
import shutil
import tempfile
import subprocess
import sys
from types import SimpleNamespace
import unittest
import numpy as np
import torch
from stable_baselines3 import PPO
from navrl.geometry import ROOT, LidarProfile
from neural_navigation.runtime import EmergencyStop, SensorNavigationRuntime
from neural_navigation.safety_context import (
    SafetyContext, contract, validate_contract, RECOVERY_SCHEMA, PREDICTIVE_SCHEMA,
    OBSERVATION_DIM, PREDICTIVE_DIM, CONTEXT_DIM)
from experiments.predictive_policy import migrate_predictive_policy
from experiments.recovery_curriculum import RecoveryNavigationCurriculum
from experiments.prior_anchored_ppo import PriorAnchoredPPO


class PredictivePolicyTests(unittest.TestCase):
    components = ROOT/'runs/safety_context_20261011/recovery_probes_v3_16000/candidate'

    def test_predictive_anchor_exempts_supported_parking_risk(self):
        model = migrate_predictive_policy(self.components/'policy.zip',
            algorithm=PriorAnchoredPPO, anchor_strength=.1)
        with torch.no_grad():
            model.policy.action_net.weight.zero_()
            model.policy.action_net.bias.copy_(torch.tensor([.5, 0.]))
        model.snapshot_reference()
        obs = torch.zeros((3, PREDICTIVE_DIM))
        obs[:, 235:244] = 1.; obs[:, 250] = 1.
        obs[:, 232] = 1.; obs[:, 234] = 1.
        obs[1, 269] = .5; obs[1, 281] = 1.
        obs[2, 269] = .5  # Unknown support must not imply a forecast.
        with torch.no_grad(): model.policy.action_net.bias[0] += .3
        loss, fraction = model.reference_loss(obs)
        self.assertAlmostEqual(fraction.item(), 2/3, places=6)
        self.assertGreater(loss.item(), 0.)

    def test_exported_actor_receives_actual_runtime_v4_observation(self):
        torch.set_num_threads(1)
        model = migrate_predictive_policy(self.components/'policy.zip')
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory)
            for name in ('hardware.json', 'localizer.pt', 'router.pt'):
                shutil.copy2(self.components/name, target/name)
            hardware = json.loads((target/'hardware.json').read_text())
            hardware['policy_observation'] = contract(PREDICTIVE_SCHEMA)
            (target/'hardware.json').write_text(json.dumps(hardware))
            model.save(target/'policy.zip')
            subprocess.run([sys.executable, str(ROOT/'export_navigation_actor.py'),
                '--policy', str(target/'policy.zip'), '--components', str(target),
                '--output', str(target/'portable')], check=True, capture_output=True)
            runtime = SensorNavigationRuntime(target/'portable',
                ROOT/'assets/room_navigation_scripted/decision_map.yaml',
                [1.2, 1.2, 0.], [8.6, 1.2])
            result = runtime.step(np.full(220, 2.), np.zeros(3), np.zeros(2),
                1., capture_observation=True)
            observation = np.asarray(result['policy_observation'], np.float32)
            self.assertEqual(observation.shape, (PREDICTIVE_DIM,))
            np.testing.assert_allclose(result['proposed_action'],
                model.predict(observation, deterministic=True)[0], rtol=0, atol=2e-6)

    def test_context_prefix_wait_and_live_guard_state_preserved(self):
        lidar = LidarProfile()
        core = SimpleNamespace(scan=np.full(lidar.beams, 2.), lidar=lidar,
            odom=np.zeros(3), elapsed=4., estimate=np.zeros(3), odometry_sanity_failure=False,
            localizer=SimpleNamespace(std=np.array([.02, .02, .01]), scan_update_accepted=True, map_agreement=1.),
            static_map=SimpleNamespace(free=np.ones((1, 1)), lookup=lambda *a, **kw: True),
            diagnostics=lambda: {'route_complete': True})
        core.laser_fusion = SimpleNamespace(predictor=SimpleNamespace(tracks=[dict(
            time=4., hits=4, velocity=np.array([-.2, 0.]), points=np.array([[.6, 0.], [.6, .02]]))]))
        guard = EmergencyStop(memory_config={'near_range_m': 1.5})
        old, new = SafetyContext(RECOVERY_SCHEMA), SafetyContext(PREDICTIVE_SCHEMA)
        result = dict(timestamp=1., proposed_action=[-.5, .1], command=[0., 0.],
            safety_override=True, stop_reason='reverse_without_rear_visibility')
        old.record(result); new.record(result)
        before = copy.deepcopy(guard)
        prefix = old.observation(core, guard, 4.); value = new.observation(core, guard, 4.)
        np.testing.assert_array_equal(prefix, value[:CONTEXT_DIM])
        self.assertEqual(len(value)+235, PREDICTIVE_DIM)
        self.assertAlmostEqual(value[CONTEXT_DIM-1], .6, places=6)
        self.assertEqual(value[-1], 1.)
        self.assertEqual(guard.memory.last_time, before.memory.last_time)
        self.assertEqual(guard.memory.entries, before.memory.entries)
        validate_contract(contract(PREDICTIVE_SCHEMA))
        bad = contract(PREDICTIVE_SCHEMA); bad['dimensions'] -= 1
        with self.assertRaises(ValueError): validate_contract(bad)

    def test_warm_start_predictions_and_adam_moments_retained(self):
        torch.set_num_threads(1)
        path = self.components/'policy.zip'
        old = PPO.load(path, device='cpu'); new = migrate_predictive_policy(path)
        rng = np.random.default_rng(181)
        inputs = rng.uniform(-1, 1, (128, OBSERVATION_DIM)).astype(np.float32)
        expanded = np.c_[inputs, rng.uniform(0, 1, (128, 15)).astype(np.float32)]
        np.testing.assert_allclose(old.predict(inputs, deterministic=True)[0],
            new.predict(expanded, deterministic=True)[0], rtol=0, atol=2e-6)
        old_params = dict(old.policy.named_parameters())
        for name, parameter in new.policy.named_parameters():
            source = old_params[name]
            if parameter.shape == source.shape:
                torch.testing.assert_close(parameter, source, rtol=0, atol=0)
            else:
                torch.testing.assert_close(parameter[:, :OBSERVATION_DIM], source, rtol=0, atol=0)
                self.assertEqual(torch.count_nonzero(parameter[:, OBSERVATION_DIM:]).item(), 0)
            for key, value in old.policy.optimizer.state.get(source, {}).items():
                actual = new.policy.optimizer.state[parameter][key]
                if torch.is_tensor(value) and value.shape == source.shape and parameter.shape != source.shape:
                    torch.testing.assert_close(actual[:, :OBSERVATION_DIM], value, rtol=0, atol=0)
                    self.assertEqual(torch.count_nonzero(actual[:, OBSERVATION_DIM:]).item(), 0)
                elif torch.is_tensor(value): torch.testing.assert_close(actual, value, rtol=0, atol=0)
                else: self.assertEqual(actual, value)

    def test_native_v3_v4_sensor_prefix_commands_and_rewards_identical(self):
        torch.set_num_threads(1)
        histories = []
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory)
            for name in ('hardware.json', 'localizer.pt', 'router.pt'):
                shutil.copy2(self.components/name, target/name)
            hw = json.loads((target/'hardware.json').read_text()); hw['policy_observation'] = contract(PREDICTIVE_SCHEMA)
            (target/'hardware.json').write_text(json.dumps(hw))
            for components in (self.components, target):
                env = RecoveryNavigationCurriculum(components, stage='robust', motion_profiles=True,
                    footprint_clearance_cost=1.)
                history = []
                try:
                    obs, _ = env.reset(seed=92115)
                    history.append((obs[:OBSERVATION_DIM].copy(),))
                    for _ in range(9):
                        obs, reward, done, truncated, info = env.step(np.array([.3, -.2]))
                        history.append((obs[:OBSERVATION_DIM].copy(), env.active.truth.copy(),
                            info['commanded_velocity'].copy(), reward, done, truncated))
                    self.assertEqual(len(obs), env.observation_space.shape[0])
                finally: env.close()
                histories.append(history)
        for left, right in zip(*histories):
            for a, b in zip(left, right):
                if isinstance(a, np.ndarray): np.testing.assert_array_equal(a, b)
                else: self.assertEqual(a, b)


if __name__ == '__main__': unittest.main()
