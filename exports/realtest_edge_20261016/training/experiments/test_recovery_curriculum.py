import unittest
import numpy as np
import torch
from navrl.geometry import ROOT
from neural_navigation.context_curriculum import ContextNavigationCurriculum
from experiments.recovery_curriculum import RejectedTranslationCost, RecoveryNavigationCurriculum
from neural_navigation.safety_context import BASE_DIM, OBSERVATION_DIM
from neural_navigation.safety_context import SCHEMA, RECOVERY_SCHEMA
from stable_baselines3 import PPO
from experiments.train_recovery_reward import migrate_probe_columns


class RecoveryCostTests(unittest.TestCase):
    def observation(self, feasible=True):
        result=np.zeros(OBSERVATION_DIM)
        result[BASE_DIM+3]=float(feasible)
        return result

    def info(self, **changes):
        result=dict(proposed_action=np.array([-.05,-1.]), commanded_velocity=np.array([0.,-.5]),
                    execution_guard_reason='reverse_without_rear_visibility',
                    training_actions_overridden=True, fixed_geodesic_progress_m=0.)
        result.update(changes)
        return result

    def test_small_rejected_translation_has_bounded_cost_after_grace(self):
        cost=RejectedTranslationCost()
        for _ in range(5): self.assertAlmostEqual(cost.step(self.observation(),self.info(),.2),0.)
        self.assertGreater(cost.step(self.observation(),self.info(),.2),0.)
        for _ in range(100): value=cost.step(self.observation(),self.info(),.2)
        self.assertAlmostEqual(value,.2)

    def test_waiting_missing_scan_valid_turn_and_no_alternative_are_exempt(self):
        for before, info in [
            (self.observation(), self.info(proposed_action=[0.,0.],commanded_velocity=[0.,0.],training_actions_overridden=False)),
            (self.observation(), self.info(execution_guard_reason='scan_unavailable')),
            (self.observation(), self.info(execution_guard_reason='localization_uncertain')),
            (self.observation(), self.info(proposed_action=[0.,1.],commanded_velocity=[0.,.5],training_actions_overridden=False)),
            (self.observation(False), self.info()),
            (self.observation(), self.info(fixed_geodesic_progress_m=.02))]:
            cost=RejectedTranslationCost()
            for _ in range(20): self.assertEqual(cost.step(before,info,.2),0.)

    def test_direction_or_reason_change_restarts_grace(self):
        cost=RejectedTranslationCost()
        for _ in range(10):cost.step(self.observation(),self.info(),.2)
        self.assertEqual(cost.step(self.observation(),self.info(proposed_action=[.5,-1.],
            execution_guard_reason='unsafe_translation_blocked'),.2),0.)
        self.assertAlmostEqual(cost.duration,.2)
        cost.reset();self.assertEqual(cost.duration,0.)

    def test_reward_variant_keeps_real_sensor_transitions_and_commands_identical(self):
        torch.set_num_threads(1)
        components=ROOT/'runs/safety_context_20261011/pilot_v2/candidate'
        transitions=[]
        for cls in (ContextNavigationCurriculum,RecoveryNavigationCurriculum):
            env=cls(components,stage='static',footprint_clearance_cost=1.)
            env.scenes=[ROOT/'assets/room_navigation_scripted']
            collected=[]
            try:
                obs,_=env.reset(seed=94022)
                for _ in range(12):
                    obs,reward,done,truncated,info=env.step(np.array([-.05,.3]))
                    collected.append((obs.copy(),info['commanded_velocity'].copy(),
                        env.active.truth.copy(),reward,info.get('repeated_rejection_penalty',0.)))
                    self.assertFalse(done or truncated)
            finally:env.close()
            transitions.append(collected)
        for original,variant in zip(*transitions):
            for a,b in zip(original[:3],variant[:3]):np.testing.assert_array_equal(a,b)
            self.assertAlmostEqual(original[3]-variant[3],variant[4])
        self.assertGreater(sum(v[4] for v in transitions[1]),0.)

    def test_probe_migration_erases_only_changed_feature_associations(self):
        torch.set_num_threads(1)
        model=PPO.load(ROOT/'runs/safety_context_20261011/pilot_v2/candidate/policy_24000.zip',device='cpu')
        before={k:v.clone() for k,v in model.policy.state_dict().items()}
        observations=np.random.default_rng(12).uniform(-1,1,(32,OBSERVATION_DIM)).astype(np.float32)
        observations[:,240:244]=0.
        actions=model.predict(observations,deterministic=True)[0]
        migrate_probe_columns(model,SCHEMA,RECOVERY_SCHEMA)
        np.testing.assert_array_equal(actions,model.predict(observations,deterministic=True)[0])
        names={'mlp_extractor.policy_net.0.weight','mlp_extractor.value_net.0.weight'}
        for name,after in model.policy.state_dict().items():
            expected=before[name]
            if name in names:expected[:,240:244]=0.
            torch.testing.assert_close(after,expected,atol=0,rtol=0)
        for network in (model.policy.mlp_extractor.policy_net,model.policy.mlp_extractor.value_net):
            state=model.policy.optimizer.state[network[0].weight]
            for key in ('exp_avg','exp_avg_sq'):
                self.assertEqual(torch.count_nonzero(state[key][:,240:244]).item(),0)


if __name__=='__main__':unittest.main()
