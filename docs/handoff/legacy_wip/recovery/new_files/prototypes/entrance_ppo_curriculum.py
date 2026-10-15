"""Short entrance milestones; these are not full navigation successes.

Control and shaping use the runtime pose/lidar, never the simulator truth.
Truth is used only to score a completed milestone, as in mission evaluation.
The external mission goal stays valid under the existing map contract.
"""
import numpy as np
from prototypes.narrow_training_env import NarrowTrainingEnv


def sample_task(rng):
    choice = rng.random()
    if choice < .35:
        return dict(stage='entrance_align', start=[float(rng.uniform(1.2, 1.45)),
                    float(rng.uniform(1.965, 2.035)), float(rng.uniform(-.10, .10))],
                    goal=[5., 2.], milestone_x=1.72, horizon=100)
    if choice < .65:
        return dict(stage='entrance_cross', start=[float(rng.uniform(1.63, 1.70)),
                    float(rng.uniform(1.994, 2.006)), float(rng.uniform(-.025, .025))],
                    goal=[5., 2.], milestone_x=2.22, horizon=120)
    if choice < .75:
        return dict(stage='inside_advance', start=[float(rng.uniform(2.06, 2.20)),
                    float(rng.uniform(1.997, 2.003)), float(rng.uniform(-.01, .01))],
                    goal=[5., 2.], milestone_x=2.55, horizon=90)
    if choice < .85:
        return dict(stage='full_passage', start=[float(rng.uniform(1.2, 1.6)),
                    float(rng.uniform(1.985, 2.015)), float(rng.uniform(-.06, .06))],
                    goal=[5., 2.], milestone_x=None, horizon=250)
    if choice < .90:
        return dict(stage='short_exit', start=[float(rng.uniform(3.6, 3.8)), 2.,
                    float(rng.uniform(-.015, .015))], goal=[4.55, 2.],
                    milestone_x=None, horizon=250)
    return dict(stage='open_room', start=[float(rng.uniform(.75, 1.05)),
                float(rng.uniform(1.8, 2.2)), float(rng.uniform(-.1, .1))],
                goal=[1.65, 2.], milestone_x=None, horizon=100)


def milestone_reached(pose, task):
    x = task['milestone_x']
    if x is None:
        return False
    pose = np.asarray(pose, float)
    angle = np.arctan2(np.sin(pose[2]), np.cos(pose[2]))
    return bool(np.isfinite(pose).all() and pose[0] >= x and
                abs(pose[1] - 2.) <= .01 and abs(angle) <= .025)


def alignment_potential(pose, task):
    if task['milestone_x'] is None:
        return 0.
    angle = np.arctan2(np.sin(pose[2]), np.cos(pose[2]))
    return -float(min(abs(pose[1]-2.)/.04, 2.) + min(abs(angle)/.10, 2.))


class EntrancePPOCurriculum(NarrowTrainingEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.training_steps = 0

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.task = sample_task(self.rng)
        merged = dict(start=self.task['start'], goal=self.task['goal'])
        merged.update(options or {})
        obs, info = super().reset(options=merged)
        self.previous_potential = alignment_potential(self.runtime.core.estimate, self.task)
        self.only_stop_streak = 0
        return obs, dict(info, training_stage=self.task['stage'])

    def step(self, action):
        legal_motion = bool(self.action_masks()[1:].any())
        self.only_stop_streak = 0 if legal_motion else self.only_stop_streak + 1
        obs, reward, done, truncated, info = super().step(action)
        collision = bool(info['collision_events'] or info['physical_guards'])
        milestone = milestone_reached(self.runtime.core.estimate, self.task) and not collision
        verified = milestone and milestone_reached(self.env.truth, self.task)
        new_potential = alignment_potential(self.runtime.core.estimate, self.task)
        # No unbounded reward for compulsory stopping. Deadlocks are diagnosed
        # separately and truncated after 20 frames, not counted as collisions.
        deadlock = self.only_stop_streak >= 20 and not done and not milestone
        task_limit = self.count >= self.task['horizon'] and not done and not milestone
        terminal = done or truncated or milestone or deadlock or task_limit
        shaping = .4 * ((0. if terminal else .99 * new_potential) - self.previous_potential)
        self.previous_potential = new_potential
        if milestone and not done:
            reward += .5
            done = True
            truncated = False
        elif not done and (deadlock or task_limit):
            truncated = True
        info.update(training_stage=self.task['stage'], training_milestone_reached=bool(milestone),
                    verified_training_milestone=bool(verified), only_stop_before_step=not legal_motion,
                    training_deadlock_truncation=bool(deadlock), alignment_shaping=shaping)
        self.training_steps += 1
        return obs, float(reward + shaping * self.reward_scale), done, truncated, info


def fixed_cases():
    """Five repeated development cases + twenty newly frozen final cases."""
    from prototypes.train_narrow_long import CASES
    cases = [dict(c, split='development', category=('full_passage' if i < 3 else c['id']))
             for i, c in enumerate(CASES)]
    rng = np.random.default_rng(95140)
    for group in ('centered', 'offset', 'short_exit', 'open_room'):
        for i in range(5):
            if group == 'centered':
                start = [float(rng.uniform(1.25, 1.55)), 2., float(rng.uniform(-.02, .02))]
                goal = [5., 2.]
            elif group == 'offset':
                sign = 1 if i % 2 else -1
                start = [float(rng.uniform(1.3, 1.6)), 2.+sign*float(rng.uniform(.008, .02)),
                         sign*float(rng.uniform(.025, .065))]
                goal = [5., 2.]
            elif group == 'short_exit':
                start = [float(rng.uniform(3.6, 3.8)), 2., float(rng.uniform(-.01, .01))]
                goal = [4.55, 2.]
            else:
                start = [float(rng.uniform(.8, 1.05)), float(rng.uniform(1.9, 2.1)),
                         float(rng.uniform(-.08, .08))]
                goal = [1.65, 2.]
            cases.append(dict(id=f'heldout_{group}_{i}', seed=95150+len(cases), start=start,
                              goal=goal, split='heldout', category='full_passage' if group in
                              ('centered', 'offset') else group))
    return cases
