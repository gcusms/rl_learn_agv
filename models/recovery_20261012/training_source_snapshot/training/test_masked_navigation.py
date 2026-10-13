import json
from pathlib import Path
import shutil
import tempfile
import unittest
import numpy as np
import torch
from stable_baselines3 import PPO
from navrl.geometry import ROOT
from neural_navigation.safety_context import contract,PREDICTIVE_SCHEMA,BOUNDED_PREDICTIVE_SCHEMA
from neural_navigation.runtime import SensorNavigationRuntime
from neural_navigation.masked_actions import action_contract
from experiments.export_masked_navigation import export_actor
from experiments.masked_navigation import (
    ACTIONS,COMMANDS,masks_from_observation,warm_start,
    PortableMaskedActor,MaskedNavigationCurriculum,FocusedMaskedCurriculum)


class MaskedNavigationTests(unittest.TestCase):
    source=ROOT/'runs/safety_context_20261011/recovery_probes_v3_16000/candidate'

    def test_focused_training_reset_and_bounded_export_share_native_masks(self):
        torch.set_num_threads(1)
        records=[json.loads(line) for line in (ROOT/'runs/safety_context_20261011/plain_recovery_control_12000/candidate/episodes.jsonl').read_text().splitlines()]
        case=json.loads(next(r['training_case_json'] for r in records if r['step']==10458))
        model=warm_start(self.source/'policy.zip')
        with tempfile.TemporaryDirectory() as directory:
            components=Path(directory)
            for name in ('hardware.json','localizer.pt','router.pt'):shutil.copy2(self.source/name,components/name)
            hw=json.loads((components/'hardware.json').read_text())
            hw['policy_observation']=contract(BOUNDED_PREDICTIVE_SCHEMA);hw['policy_action']=action_contract()
            (components/'hardware.json').write_text(json.dumps(hw));model.save(components/'policy.zip')
            export_actor(components/'policy.zip',components,components/'portable')
            env=FocusedMaskedCurriculum(components,stage='robust',motion_profiles=True,
                footprint_clearance_cost=1.,recorded_cases=[case],replay_probability=.5)
            try:
                obs,info=env.reset(seed=2)
                self.assertTrue(info['focused_training_replay'])
                self.assertEqual(json.loads(env.training_case_json),case)
                actor=torch.jit.load(str(components/'portable/actor.pt'))
                chosen=int(model.predict(obs,action_masks=env.action_masks(),deterministic=True)[0])
                with torch.no_grad():portable=actor(torch.from_numpy(obs[None])).numpy()[0]
                np.testing.assert_array_equal(portable,ACTIONS[chosen])
                _,_,_,_,step_info=env.step(chosen)
                self.assertFalse(step_info['nonzero_choice_changed_by_guard'])
                again,_=env.reset(seed=2)
                np.testing.assert_array_equal(obs,again)
            finally:env.close()

    def test_unknown_retains_only_wait_and_risk_does_not_forbid_legal_wait(self):
        obs=np.zeros(282,np.float32)
        np.testing.assert_array_equal(masks_from_observation(obs),[True]+[False]*11)
        obs[240]=1.;obs[267]=1.;obs[269]=1.;obs[281]=1.
        mask=masks_from_observation(obs)
        self.assertTrue(mask[0]);self.assertTrue(mask[6]);self.assertTrue(mask[10])
        self.assertEqual(mask.sum(),3)
        with self.assertRaises(ValueError):masks_from_observation(np.zeros(267))

    def test_gaussian_density_migration_and_portable_masked_inference(self):
        torch.set_num_threads(1)
        old=PPO.load(self.source/'policy.zip',device='cpu')
        new=warm_start(self.source/'policy.zip')
        obs=np.random.default_rng(23).uniform(-1,1,(128,282)).astype(np.float32)
        obs[:,235:244]=np.random.default_rng(31).integers(0,2,(128,9))
        obs[:,267:269]=np.random.default_rng(41).integers(0,2,(128,2))
        obs[0,235:244]=0.;obs[0,267:269]=0.
        masks=masks_from_observation(obs)
        with torch.no_grad():
            mean=old.policy.get_distribution(torch.from_numpy(obs[:,:267])).distribution.mean
            sigma=old.policy.log_std.exp()
            expected=-.5*((torch.from_numpy(ACTIONS)[None,:,:]-mean[:,None,:])/sigma).square().sum(dim=2)
            expected=torch.where(torch.from_numpy(masks),expected,torch.full_like(expected,-1e8))
            distribution=new.policy.get_distribution(torch.from_numpy(obs),action_masks=masks)
            actual=distribution.distribution.logits
            torch.testing.assert_close(actual,torch.log_softmax(expected,dim=1),rtol=2e-5,atol=2e-5)
        indices=new.predict(obs,action_masks=masks,deterministic=True)[0]
        np.testing.assert_array_equal(masks[np.arange(len(indices)),indices],True)
        actor=PortableMaskedActor(new.policy).eval()
        exported=torch.jit.freeze(torch.jit.trace(actor,torch.zeros(1,282)))
        with torch.no_grad():result=exported(torch.from_numpy(obs)).numpy()
        np.testing.assert_array_equal(result,ACTIONS[indices])
        np.testing.assert_array_equal(result[0],[0.,0.])

    def test_native_ppo_rollouts_store_valid_executed_choices(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as directory:
            components=Path(directory)
            for name in ('hardware.json','localizer.pt','router.pt'):shutil.copy2(self.source/name,components/name)
            hw=json.loads((components/'hardware.json').read_text());hw['policy_observation']=contract(PREDICTIVE_SCHEMA)
            (components/'hardware.json').write_text(json.dumps(hw))
            env=MaskedNavigationCurriculum(components,stage='robust',motion_profiles=True,footprint_clearance_cost=1.)
            model=warm_start(self.source/'policy.zip',n_steps=8,batch_size=8,n_epochs=1)
            try:
                obs,_=env.reset(seed=92201)
                mask=env.action_masks()
                index=int(np.flatnonzero(mask)[-1])
                _,_,_,_,info=env.step(index)
                np.testing.assert_allclose(info['commanded_velocity'],COMMANDS[index],atol=2e-8,rtol=0)
                self.assertFalse(info['nonzero_choice_changed_by_guard'])
                model.set_env(env)
                model.learn(16)
                stored=model.rollout_buffer
                indices=stored.actions.astype(int).reshape(-1)
                masks=stored.action_masks.reshape(-1,len(ACTIONS))
                self.assertTrue(masks[np.arange(len(indices)),indices].all())
                bad=np.flatnonzero(~env.action_masks())
                if len(bad):
                    with self.assertRaises(ValueError):env.step(int(bad[0]))
            finally:env.close()

    def test_export_runtime_and_stale_scan_use_same_masks(self):
        torch.set_num_threads(1)
        model=warm_start(self.source/'policy.zip')
        with tempfile.TemporaryDirectory() as directory:
            components=Path(directory)
            for name in ('hardware.json','localizer.pt','router.pt'):shutil.copy2(self.source/name,components/name)
            hw=json.loads((components/'hardware.json').read_text())
            hw['policy_observation']=contract(PREDICTIVE_SCHEMA);hw['policy_action']=action_contract()
            (components/'hardware.json').write_text(json.dumps(hw))
            model.save(components/'policy.zip')
            manifest=export_actor(components/'policy.zip',components,components/'portable')
            self.assertEqual(manifest['max_action_error'],0.)
            runtime=SensorNavigationRuntime(components/'portable',
                ROOT/'assets/room_navigation_scripted/decision_map.yaml',[1.2,1.2,0.],[8.6,1.2])
            result=runtime.step(np.full(220,2.),np.zeros(3),np.zeros(2),1.,capture_observation=True)
            obs=np.array(result['policy_observation'],np.float32);mask=masks_from_observation(obs)
            chosen=int(model.predict(obs,action_masks=mask,deterministic=True)[0])
            np.testing.assert_array_equal(result['proposed_action'],ACTIONS[chosen])
            stale=runtime.step(np.full(220,2.),np.zeros(3),np.zeros(2),1.2,
                received_timestamp=2.,capture_observation=True)
            np.testing.assert_array_equal(masks_from_observation(stale['policy_observation']),[True]+[False]*11)
            np.testing.assert_array_equal(stale['proposed_action'],[0.,0.])
            np.testing.assert_array_equal(stale['command'],[0.,0.])


if __name__=='__main__':unittest.main()
