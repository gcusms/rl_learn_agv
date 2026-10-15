import json
from pathlib import Path
import tempfile
import unittest
import gymnasium as gym
import numpy as np
import torch
from navrl.geometry import ROOT
from neural_navigation.decision_feedback import TRAIN_DIM, RUNTIME_DIM, WAIT_SCHEMA, LEGACY_SCHEMA, contract
from neural_navigation.masked_actions import masks_from_observation
from experiments.stop_feedback_ppo import warm_start, decision_support, StopFeedbackPPO
from experiments.train_stop_feedback import export_candidate


class Interface(gym.Env):
    reward_scale=1/60
    observation_space=gym.spaces.Box(-1,1,(TRAIN_DIM,),np.float32)
    action_space=gym.spaces.Discrete(12)
    def reset(self,*,seed=None,options=None):raise RuntimeError('Unit interface, not a simulator')
    def step(self,action):raise RuntimeError('Unit interface, not a simulator')


class TrainingInterface(Interface):
    """Checks the custom PPO update only; never scored as navigation."""
    def observation(self):
        x=np.zeros(TRAIN_DIM,np.float32);x[:216]=.7;x[216]=.25;x[218]=.2
        if self.index%3:
            x[235:244]=1.;x[267:269]=1.
        x[251]=1.;x[294]=1.
        return x
    def reset(self,*,seed=None,options=None):
        gym.Env.reset(self,seed=seed)
        self.index=0
        return self.observation(),{}
    def action_masks(self):return masks_from_observation(self.observation()[:282])
    def step(self,action):
        assert self.action_masks()[action]
        self.index+=1
        return self.observation(),float(action!=0)*.01,False,self.index>=20,{}


class StopFeedbackPPOTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.model=warm_start(ROOT.parent/'models/quality_recovery_20261014/policy_2000.zip',Interface())
        rng=np.random.default_rng(22)
        cls.obs=rng.uniform(-1,1,(128,TRAIN_DIM)).astype(np.float32)
        cls.obs[:,235:244]=rng.integers(0,2,(128,9));cls.obs[:,267:269]=rng.integers(0,2,(128,2))

    def test_new_inputs_start_at_exact_parent_logits_and_all_actor_weights_train(self):
        model=self.model
        with torch.no_grad():
            x=torch.from_numpy(self.obs)
            logits=model.policy.action_net(model.policy.mlp_extractor.policy_net(x))
            original=model.reference_net(x[:,:300])
            torch.testing.assert_close(logits,original,rtol=0,atol=0)
        names=[(name,p) for name,p in model.policy.named_parameters() if name.startswith(('action_net.','mlp_extractor.policy_net.'))]
        self.assertTrue(all(parameter.requires_grad for _,parameter in names))

    def test_available_choices_are_trainable_without_the_old_quality_gate(self):
        obs=torch.zeros(3,TRAIN_DIM);obs[0,235]=1;obs[1,267]=1
        self.assertEqual(decision_support(obs).tolist(),[True,True,False])
        self.assertEqual(obs[:,294].sum(),0.)

    def test_appended_explanations_can_receive_a_policy_gradient(self):
        model=self.model;model.policy.optimizer.zero_grad()
        x=torch.from_numpy(self.obs.copy());mask=masks_from_observation(self.obs[:,:282])
        distribution=model.policy.get_distribution(x,action_masks=mask)
        loss=-distribution.log_prob(torch.zeros(len(x),dtype=torch.long)).mean();loss.backward()
        gradient=model.policy.mlp_extractor.policy_net[0].weight.grad
        self.assertGreater(float(gradient[:,300:].abs().sum()),0.)
        model.policy.optimizer.zero_grad()

    def test_custom_ppo_update_runs_with_forced_and_free_choice_samples(self):
        model=warm_start(ROOT.parent/'models/quality_recovery_20261014/policy_2000.zip',TrainingInterface())
        model.learn(total_timesteps=250)
        self.assertEqual(model.num_timesteps,250)
        self.assertTrue(np.isfinite(model.logger.name_to_value['train/loss']))
        fraction=model.logger.name_to_value['train/actor_decision_fraction']
        self.assertGreater(fraction,0);self.assertLess(fraction,1)

    def test_save_reload_and_export_keep_masks_and_directional_runtime(self):
        model=self.model;mask=masks_from_observation(self.obs[:,:282])
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)
            model.save(path/'policy.zip')
            restored=StopFeedbackPPO.load(path/'policy.zip',device='cpu')
            np.testing.assert_array_equal(model.predict(self.obs,action_masks=mask,deterministic=True)[0],
                restored.predict(self.obs,action_masks=mask,deterministic=True)[0])
            bundle=ROOT.parent/'models/localization_mapping_20261015/ray_consistent_candidate/portable'
            export_candidate(restored,bundle,path/'portable')
            from neural_navigation.decision_feedback_runtime import StopFeedbackRuntime
            from neural_navigation.goal_pose_runtime import DockingRuntime,resolve_runtime
            meta=json.loads((ROOT/'assets/localization_replay_20261015/short_00.json').read_text())
            with np.load(ROOT/'assets/localization_replay_20261015/short_00.npz') as data:
                m=ROOT/meta['case']['scene']/'decision_map.yaml';prior=data['prior'];goal=[*prior[:2],prior[2]+.5]
                old=DockingRuntime(bundle,m,prior,goal,'shortcut')
                new=StopFeedbackRuntime(path/'portable',m,prior,goal,'shortcut')
                for i in range(4):
                    args=(data['scan'][i],data['odom'][i],data['velocity'][i],float(data['t'][i]))
                    before=old.step(*args,capture_observation=True);after=new.step(*args,capture_observation=True)
                    self.assertEqual(after['policy_observation_dimensions'],RUNTIME_DIM)
                    np.testing.assert_array_equal(before['policy_observation'],after['policy_observation'][:303])
                    np.testing.assert_array_equal(before['command'],after['command'])
                self.assertIs(resolve_runtime('neural_navigation.decision_feedback_runtime.StopFeedbackRuntime'),StopFeedbackRuntime)

    def test_v2_training_schema_survives_checkpoint_export_and_runtime(self):
        env=Interface();env.feedback_schema=WAIT_SCHEMA
        model=warm_start(ROOT.parent/'models/quality_recovery_20261014/policy_2000.zip',env)
        self.assertEqual(model.feedback_schema,WAIT_SCHEMA)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory);model.save(path/'policy.zip')
            restored=StopFeedbackPPO.load(path/'policy.zip',device='cpu')
            self.assertEqual(restored.feedback_schema,WAIT_SCHEMA)
            bundle=ROOT.parent/'models/localization_mapping_20261015/ray_consistent_candidate/portable'
            export_candidate(restored,bundle,path/'portable')
            manifest=json.loads((path/'portable/manifest.json').read_text())
            self.assertEqual(manifest['observation_contract'],contract(WAIT_SCHEMA))
            from neural_navigation.decision_feedback_runtime import StopFeedbackRuntime
            meta=json.loads((ROOT/'assets/localization_replay_20261015/short_00.json').read_text())
            with np.load(ROOT/'assets/localization_replay_20261015/short_00.npz') as data:
                prior=data['prior'];goal=[*prior[:2],prior[2]+.5]
                runtime=StopFeedbackRuntime(path/'portable',ROOT/meta['case']['scene']/'decision_map.yaml',prior,goal,'shortcut')
                result=runtime.step(data['scan'][0],data['odom'][0],data['velocity'][0],float(data['t'][0]))
                self.assertEqual(runtime.policy.feedback.schema,WAIT_SCHEMA)
                self.assertEqual(result['policy_observation_dimensions'],RUNTIME_DIM)

    def test_export_rejects_relabelling_v1_weights_as_v2(self):
        model=warm_start(ROOT.parent/'models/quality_recovery_20261014/policy_2000.zip',Interface())
        self.assertEqual(model.feedback_schema,LEGACY_SCHEMA)
        model.feedback_schema=WAIT_SCHEMA
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                export_candidate(model,ROOT.parent/'models/localization_mapping_20261015/ray_consistent_candidate/portable',Path(directory)/'invalid')


if __name__=='__main__':unittest.main()
