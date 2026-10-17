"""Protect the PPO/execution contract and the frozen navigation branch."""
import unittest
import gymnasium as gym
import numpy as np
import torch
from experiments.docking_navigation import DockingPolicy,PortableDockingActor,demonstration_choice
from neural_navigation.masked_actions import ACTIONS,masks_from_observation


class DockingTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1);torch.manual_seed(9)
        self.policy=DockingPolicy(gym.spaces.Box(-1,1,(303,),np.float32),gym.spaces.Discrete(12),lambda _:1e-3)
        self.obs=torch.randn(24,303).clamp(-1,1)
        self.obs[:12,218:220]=.01;self.obs[:12,302]=1
        self.obs[12:18,218:220]=.2;self.obs[12:18,302]=1
        self.obs[18:,218:220]=.01;self.obs[18:,302]=0
        self.masks=masks_from_observation(self.obs[:,:282].numpy())

    def test_rollout_and_update_use_identical_masked_likelihood(self):
        action,value,old=self.policy(self.obs,action_masks=self.masks)
        new_value,new,entropy=self.policy.evaluate_actions(self.obs,action,action_masks=self.masks)
        torch.testing.assert_close(old,new,rtol=0,atol=0)
        torch.testing.assert_close(value,new_value,rtol=0,atol=0)
        self.assertTrue(np.all(self.masks[np.arange(len(action)),action.numpy()]))
        self.assertTrue(torch.isfinite(entropy).all())

    def test_learning_changes_terminal_branch_and_preserves_navigation(self):
        before=self.policy.logits(self.obs).detach().clone()
        loss=-self.policy.get_distribution(self.obs[:12],action_masks=self.masks[:12]).log_prob(torch.zeros(12,dtype=torch.long)).mean()
        self.policy.optimizer.zero_grad();loss.backward();self.policy.optimizer.step()
        after=self.policy.logits(self.obs).detach()
        torch.testing.assert_close(before[12:],after[12:],rtol=0,atol=0)
        self.assertGreater(float((before[:12]-after[:12]).abs().max()),0)

    def test_portable_has_identical_deterministic_choices(self):
        choice=self.policy.get_distribution(self.obs,action_masks=self.masks).get_actions(deterministic=True)
        with torch.no_grad():actual=PortableDockingActor(self.policy)(self.obs).numpy()
        np.testing.assert_array_equal(actual,ACTIONS[choice.numpy()])

    def test_staged_gate_keeps_approach_with_navigation_and_export_matches(self):
        from experiments.docking_navigation import docking_contract,terminal_gate
        self.policy.terminal_radius=.10
        obs=self.obs.clone();obs[:,218:220]=0.;obs[:,218]=.20/12;obs[:,302]=1
        self.assertFalse(terminal_gate(obs,.10).any())
        expected=self.policy.action_net(self.policy.mlp_extractor.forward_actor(self.policy.extract_features(obs)))
        torch.testing.assert_close(self.policy.logits(obs),expected,rtol=0,atol=0)
        obs[:12,218]=.03/12
        self.assertTrue(terminal_gate(obs,.10)[:12].all())
        masks=masks_from_observation(obs[:,:282].numpy())
        choice=self.policy.get_distribution(obs,action_masks=masks).get_actions(deterministic=True)
        scripted=torch.jit.script(PortableDockingActor(self.policy))
        with torch.no_grad():actual=scripted(obs).numpy()
        np.testing.assert_array_equal(actual,ACTIONS[choice.numpy()])
        self.assertEqual(docking_contract(.10)['activation_distance_m'],.10)

    def test_reference_uses_heading_sign_and_keeps_stop_legal(self):
        obs=np.zeros(303,np.float32);obs[301]=1;mask=np.ones(12,bool)
        self.assertEqual(demonstration_choice(obs,mask),0)
        obs[300]=1;obs[301]=0;self.assertEqual(demonstration_choice(obs,mask),4)
        obs[300]=-1;self.assertEqual(demonstration_choice(obs,mask),5)
        mask[5]=False;self.assertEqual(demonstration_choice(obs,mask),0)

    def test_turn_response_is_opt_in_and_keeps_position_gate(self):
        from neural_navigation.odom_anchored_localizer import contract,turn_response_contract
        original=contract();candidate=turn_response_contract()
        self.assertEqual(original['maximum_yaw_rate_rad_s'],.06)
        self.assertEqual(candidate['maximum_yaw_rate_rad_s'],.15)
        for key in ('maximum_translation_rate_m_s','endpoint_gate_m','minimum_hits','minimum_support_fraction'):
            self.assertEqual(original[key],candidate[key])

    def test_adapter_accepts_terminal_contract_and_rejects_wrong_runtime_or_gate(self):
        import hashlib,importlib.util,json,tempfile
        from pathlib import Path
        from experiments.docking_navigation import docking_contract
        from neural_navigation.goal_pose import contract as goal_contract
        from neural_navigation.quality_prediction_context import contract as quality_contract
        from neural_navigation.masked_actions import action_contract
        adapter=Path(__file__).resolve().parents[2]/'deployment/ros2_recording_adapter/model_bundle.py'
        spec=importlib.util.spec_from_file_location('terminal_bundle_test',adapter)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);bundle=root/'bundle';bundle.mkdir()
            for name in ('actor.pt','localizer.pt','router.pt'):(bundle/name).write_text('fixture')
            (bundle/'hardware.json').write_text(json.dumps(dict(policy_action=action_contract())))
            for name,value in [('quality_observation.json',quality_contract()),('goal_pose_observation.json',goal_contract()),('docking_contract.json',docking_contract(.10))]:
                (bundle/name).write_text(json.dumps(value))
            manifest=dict(sensor_core='learned-terminal-goal-pose',required_runtime='experiments.docking_navigation.DockingRuntime',
                observation_contract=goal_contract(),quality_prefix_contract=quality_contract(),action_contract=action_contract(),
                docking_contract=docking_contract(.10),requires_quality_feature_generator=True,deployment_approved=False,
                files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in bundle.iterdir()})
            (bundle/'manifest.json').write_text(json.dumps(manifest))
            config=dict(bundle=str(bundle),required_runtime=manifest['required_runtime'],observation_dimensions=303,action_choices=12)
            path=root/'model.json';path.write_text(json.dumps(config))
            self.assertFalse(module.inspect_bundle(path)['deployment_approved'])
            config['required_runtime']='experiments.goal_pose_navigation.GoalPoseRuntime';path.write_text(json.dumps(config))
            with self.assertRaisesRegex(ValueError,'runtime'):module.inspect_bundle(path)
            config['required_runtime']=manifest['required_runtime'];path.write_text(json.dumps(config))
            manifest['docking_contract']['activation_distance_m']=.8
            (bundle/'docking_contract.json').write_text(json.dumps(manifest['docking_contract']))
            manifest['files']['docking_contract.json']=hashlib.sha256((bundle/'docking_contract.json').read_bytes()).hexdigest()
            (bundle/'manifest.json').write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError,'Terminal policy'):module.inspect_bundle(path)

    def test_stationary_and_moving_away_cannot_farm_progress(self):
        from experiments.docking_navigation import docking_progress_reward
        self.assertEqual(docking_progress_reward(.03,.03,np.pi,np.pi),0.)
        self.assertLess(docking_progress_reward(.03,.06,np.pi,np.pi),0.)
        self.assertGreater(docking_progress_reward(.03,.03,.5,.3),0.)
        self.assertEqual(docking_progress_reward(.5,.5,.5,.3),0.)

    def test_heading_counterfactuals_preserve_all_sensor_fields_and_groups(self):
        from experiments.train_docking_navigation import augment_goal_conditions
        obs=self.obs.numpy().copy();obs[:,218:220]=.002;obs[:,302]=1
        data=dict(observations=obs,masks=self.masks.copy(),actions=np.zeros(24,int),episode=np.arange(24))
        augmented=augment_goal_conditions(data,2)
        self.assertEqual(len(augmented['observations']),72)
        np.testing.assert_array_equal(augmented['observations'][24:48,:300],obs[:,:300])
        np.testing.assert_array_equal(augmented['masks'][24:48],data['masks'])
        np.testing.assert_array_equal(augmented['episode'][24:48],data['episode'])
        np.testing.assert_array_equal(augmented['observations'][24:48,302],obs[:,302])


if __name__=='__main__':unittest.main()
