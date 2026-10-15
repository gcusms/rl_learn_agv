import unittest
from unittest.mock import patch
from types import SimpleNamespace
import numpy as np
from prototypes.entrance_ppo_curriculum import (
    sample_task, milestone_reached, fixed_cases, EntrancePPOCurriculum)


class EntranceCurriculumTests(unittest.TestCase):
    def test_milestone_requires_heading_and_centering_not_just_forward_travel(self):
        task=dict(milestone_x=1.72)
        self.assertTrue(milestone_reached([1.73, 2., 0.], task))
        for pose in ([1.71, 2., 0.], [1.73, 2.02, 0.], [1.73, 2., .04], [np.nan, 2., 0.]):
            self.assertFalse(milestone_reached(pose, task))
        self.assertFalse(milestone_reached([5., 2., 0.], dict(milestone_x=None)))

    def test_curriculum_replays_regressions_and_never_places_mission_goals_inside_lane(self):
        rng=np.random.default_rng(74140)
        tasks=[sample_task(rng) for _ in range(1000)]
        for stage in ('entrance_align', 'entrance_cross', 'inside_advance', 'full_passage', 'short_exit', 'open_room'):
            self.assertGreater(sum(t['stage']==stage for t in tasks), 25)
        self.assertTrue(all(t['goal'][0]<2. or t['goal'][0]>4. for t in tasks))

    def test_fixed_holdouts_reproducible_and_disjoint_from_development(self):
        cases=fixed_cases()
        self.assertEqual(cases, fixed_cases()); self.assertEqual(len(cases), 25)
        self.assertEqual(len({c['id'] for c in cases}), 25)
        self.assertEqual(len({c['seed'] for c in cases}), 25)
        self.assertEqual(sum(c['split']=='heldout' for c in cases), 20)
        self.assertEqual(sum(c['category']=='full_passage' for c in cases), 13)

    def synthetic_env(self):
        env=object.__new__(EntrancePPOCurriculum)
        env.task=dict(stage='entrance_align', milestone_x=1.72, horizon=100)
        env.runtime=SimpleNamespace(core=SimpleNamespace(estimate=np.array([1.73, 2., 0.])))
        env.env=SimpleNamespace(truth=np.array([1.73, 2., 0.]))
        env.observation=np.zeros(318, np.float32)
        env.only_stop_streak=0; env.previous_potential=0.; env.training_steps=0; env.count=5
        return env

    def test_stage_terminal_is_not_falsely_a_full_mission_success(self):
        env=self.synthetic_env()
        info=dict(collision_events=0, physical_guards=0, verified_mission_success=False)
        with patch('prototypes.narrow_training_env.NarrowTrainingEnv.step', return_value=(np.zeros(318), 0., False, False, info)):
            _, reward, done, truncated, result=env.step(0)
        self.assertTrue(done); self.assertFalse(truncated); self.assertEqual(reward, .5)
        self.assertTrue(result['verified_training_milestone'])
        self.assertFalse(result['verified_mission_success'])

    def test_collision_cannot_earn_milestone_bonus(self):
        env=self.synthetic_env()
        info=dict(collision_events=1, physical_guards=0, verified_mission_success=False)
        with patch('prototypes.narrow_training_env.NarrowTrainingEnv.step', return_value=(np.zeros(318), -1., True, False, info)):
            _, reward, _, _, result=env.step(0)
        self.assertEqual(reward, -1.); self.assertFalse(result['training_milestone_reached'])

    def test_compulsory_stopping_is_counted_as_deadlock_not_collision(self):
        env=self.synthetic_env(); env.runtime.core.estimate[0]=1.4; env.only_stop_streak=19
        info=dict(collision_events=0, physical_guards=0, verified_mission_success=False)
        with patch('prototypes.narrow_training_env.NarrowTrainingEnv.step', return_value=(np.zeros(318), -.01, False, False, info)):
            _, _, done, truncated, result=env.step(0)
        self.assertFalse(done); self.assertTrue(truncated)
        self.assertTrue(result['training_deadlock_truncation'])
        self.assertEqual(result['collision_events'], 0)


if __name__ == '__main__':
    unittest.main()
