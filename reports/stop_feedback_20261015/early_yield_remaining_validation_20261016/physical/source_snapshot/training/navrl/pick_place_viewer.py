"""Column-only pick/place interaction on the native MuJoCo window."""
import time
import glfw
import mujoco
import numpy as np
from navrl.native_viewer import NativeManualNavigationViewer


class PickPlaceViewer(NativeManualNavigationViewer):
    def _button(self, window, button, action, mods):
        x, y = glfw.get_cursor_pos(self.window)
        if self._inside(x, y):
            if button == glfw.MOUSE_BUTTON_LEFT and action == glfw.PRESS:
                now = time.monotonic(); last, px, py = self.last_click
                if now-last < .4 and np.hypot(x-px, y-py) < 8:
                    self.events.put(('pick', x, y))
                    self.last_click = (-10., x, y)
                    return
                self.last_click = (now, x, y)
            if button == glfw.MOUSE_BUTTON_RIGHT and self.env.carried is not None:
                if action == glfw.PRESS:
                    self.events.put(('place', x, y))
                return
        if self.old_button:
            self.old_button(window, button, action, mods)

    def _move(self, window, x, y):
        if self.old_move:
            self.old_move(window, x, y)

    def poll_events(self):
        with self.lock():
            mujoco.mjv_updateScene(self.env.model, self.env.data, self.opt, None,
                                  self.cam, mujoco.mjtCatBit.mjCAT_ALL, self.scene)
            while not self.events.empty():
                kind, x, y = self.events.get()
                if kind == 'pick' and self.env.carried is None:
                    px, py, h = self._pixels(x, y); v = self.viewport
                    point = np.zeros(3); geom = np.zeros(1, np.int32)
                    body = mujoco.mjv_select(self.env.model, self.env.data, self.opt,
                        v.width/max(v.height, 1), (px-v.left)/v.width,
                        (h-py-v.bottom)/v.height, self.scene, point,
                        geom, geom.copy(), geom.copy())
                    if body in self.env.human_body_ids:
                        index = self.env.human_body_ids.index(body)
                        self.env.pick_up(index)
                        print('OBSTACLE PICKED', index+1, flush=True)
                elif kind == 'place':
                    point = self._point(x, y)
                    if point is not None:
                        success = self.env.place(point)
                        print('OBSTACLE PLACE', success, self.env.placement_message, flush=True)
                self.handle.perturb.active = 0

    def __enter__(self):
        return self

    def __exit__(self, *unused):
        self.close()
