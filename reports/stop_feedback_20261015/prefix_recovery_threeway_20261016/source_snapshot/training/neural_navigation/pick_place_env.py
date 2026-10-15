"""Manual world edits only; no obstacle truth is passed to the policy."""
import mujoco
import numpy as np
from neural_navigation.waypoint_interference import WaypointInterferenceEnv


class PickPlaceEnv(WaypointInterferenceEnv):
    def __init__(self, **kwargs):
        kwargs['dynamic'] = True
        super().__init__(**kwargs)
        self.carried = None
        self.saved_visual = None
        self.placement_message = 'Double-click column to pick up; right-click ground to place'

    def reset(self, **kwargs):
        if getattr(self, 'carried', None) is not None:
            self._restore_visual()
        self.carried = None
        result = super().reset(**kwargs)
        # A custom robot start can overlap the manual scene's parked columns.
        # Move only conflicting initial columns to validated free parking cells.
        for index, position in enumerate(self._humans()):
            if self.robot_gap(position) >= .15:
                continue
            for cell in self.free_cells:
                point = self.human_map.xy(cell)
                if (self.wall_gap(point) > .05 and self.robot_gap(point) > .5
                        and all(j == index or np.linalg.norm(point-other) > .6
                                for j, other in enumerate(self._humans()))):
                    q, d = self.addresses[self.names[index+1]]
                    self.data.qpos[q] = point; self.data.qvel[d] = 0
                    mujoco.mj_forward(self.model, self.data)
                    break
            else:
                raise ValueError('No safe initial obstacle parking position')
        self.scan = self._scan()
        self.estimate = self.localizer.update(self.odom, self.scan)
        observation = self._observation(self.scan, initialize=True)
        self.previous_remaining = self.remaining
        self.minimum_gap = self._minimum_gap()
        return observation, self._info('running')

    def _walker_velocity(self, index, h):
        # No scripted trajectories, yielding, or random walking in this UI.
        self.walkers[index]['state'] = 'manual_static'
        return np.zeros(2)

    def _restore_visual(self):
        ids, rgba, contype, conaffinity = self.saved_visual
        self.model.geom_rgba[ids] = rgba
        self.model.geom_contype[ids] = contype
        self.model.geom_conaffinity[ids] = conaffinity
        self.saved_visual = None

    def pick_up(self, index):
        if self.carried is not None:
            return False
        if index not in range(len(self.walkers)):
            raise ValueError('Only movable obstacle columns can be picked up')
        body = self.human_body_ids[index]
        ids = np.flatnonzero(self.model.geom_bodyid == body)
        self.saved_visual = (ids, self.model.geom_rgba[ids].copy(),
                             self.model.geom_contype[ids].copy(),
                             self.model.geom_conaffinity[ids].copy())
        self.carried = index
        self.model.geom_rgba[ids, 3] = 0
        self.model.geom_contype[ids] = 0
        self.model.geom_conaffinity[ids] = 0
        q, d = self.addresses[self.names[index+1]]
        self.data.qpos[q] = [30.+index, 30.]
        self.data.qvel[d] = 0
        mujoco.mj_forward(self.model, self.data)
        self.scan = self._scan()
        self.placement_message = f'Carrying column {index+1}; right-click free ground'
        return True

    def place(self, point):
        if self.carried is None:
            return False
        point = np.asarray(point, float)
        index = self.carried
        if (point.shape != (2,) or not np.isfinite(point).all()
                or not self.human_map.lookup(point, self.human_map.free, outside=False)
                or self.wall_gap(point) < .02 or self.robot_gap(point) < .10
                or any(j != index and np.linalg.norm(point-other) < .55
                       for j, other in enumerate(self._humans()))):
            self.placement_message = 'Rejected: choose free ground away from robot, walls and columns'
            return False
        q, d = self.addresses[self.names[index+1]]
        self.data.qpos[q] = point
        self.data.qvel[d] = 0
        self._restore_visual()
        self.carried = None
        mujoco.mj_forward(self.model, self.data)
        self.scan = self._scan()
        self.placement_message = f'Column {index+1} placed at {point[0]:.2f}, {point[1]:.2f}'
        return True
