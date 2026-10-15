"""Native MuJoCo side panels with column-only mouse interaction.

A tiny GLFW forwarding library exposes the native window without accessing
private GLFW structures. Native panel/camera callbacks are preserved; column
input is queued for the simulation thread to avoid mutating physics on the UI thread.
"""
import ctypes
from pathlib import Path
from queue import SimpleQueue
import time
import glfw
import mujoco
import numpy as np
from navrl.mouse_drag import ground_point

class NativeManualNavigationViewer:
    def __init__(self,env,key_callback=None):
        self.env=env;self.selected=None;self.dragging=False;self.drag_offset=np.zeros(2)
        self.drag_camera=None;self.events=SimpleQueue();self.last_click=(-10.,-100.,-100.)
        self.shim=ctypes.CDLL(str(Path(__file__).with_name('glfw_capture.so')))
        self.shim.navigation_glfw_window.restype=ctypes.c_void_p
        # viewer.py initializes native GLFW dispatch once, using this handle.
        original_handle=glfw._glfw._handle
        try:
            glfw._glfw._handle=self.shim._handle
            import mujoco.viewer
        finally:glfw._glfw._handle=original_handle
        def native_key(code):
            # Space is handled by the native Run checkbox; other task keys pass through.
            if code!=glfw.KEY_SPACE and key_callback:key_callback(code)
        self.handle=mujoco.viewer.launch_passive(env.model,env.data,key_callback=native_key,show_left_ui=True,show_right_ui=True)
        self.cam=self.handle.cam;self.opt=self.handle.opt;self.user_scn=self.handle.user_scn
        self.scene=mujoco.MjvScene(env.model,maxgeom=7000)
        address=self.shim.navigation_glfw_window()
        if not address:raise RuntimeError('Native MuJoCo GLFW window was not captured')
        self.window=ctypes.cast(address,ctypes.POINTER(glfw._GLFWwindow))
        glfw.set_window_size(self.window,1400,900)
        glfw.set_window_title(self.window,'MuJoCo - Manual navigation with native panels')
        self.button_type=ctypes.CFUNCTYPE(None,ctypes.c_void_p,ctypes.c_int,ctypes.c_int,ctypes.c_int)
        self.move_type=ctypes.CFUNCTYPE(None,ctypes.c_void_p,ctypes.c_double,ctypes.c_double)
        self.button_callback=self.button_type(self._button);self.move_callback=self.move_type(self._move)
        def install(name,callback,prototype):
            fn=getattr(glfw._glfw,name);fn.argtypes=[ctypes.c_void_p,ctypes.c_void_p];fn.restype=ctypes.c_void_p
            old=fn(address,ctypes.cast(callback,ctypes.c_void_p))
            return prototype(old) if old else None
        self.old_button=install('glfwSetMouseButtonCallback',self.button_callback,self.button_type)
        self.old_move=install('glfwSetCursorPosCallback',self.move_callback,self.move_type)
    @property
    def viewport(self):return self.handle.viewport
    @property
    def simulation_paused(self):
        sim=self.handle._get_sim()
        return sim is not None and not bool(sim.run)
    def lock(self):return self.handle.lock()
    def is_running(self):return self.handle.is_running()
    def set_texts(self,texts):self.handle.set_texts(texts)
    def _pixels(self,x,y):
        w,h=glfw.get_window_size(self.window);fw,fh=glfw.get_framebuffer_size(self.window)
        return x*fw/max(w,1),y*fh/max(h,1),fh
    def _inside(self,x,y):
        px,py,h=self._pixels(x,y);v=self.viewport
        return bool(v and v.left<=px<v.left+v.width and v.bottom<=h-py<v.bottom+v.height)
    def _restore_drag_camera(self):
        if self.drag_camera is None:return
        for name,value in self.drag_camera.items():
            if name=='lookat':self.cam.lookat[:]=value
            else:setattr(self.cam,name,value)
    def _point(self,x,y):
        px,py,h=self._pixels(x,y);cam=mujoco.mjv_averageCamera(self.scene.camera[0],self.scene.camera[1])
        return ground_point(px,py,h,self.viewport,cam)
    def _button(self,window,button,action,mods):
        x,y=glfw.get_cursor_pos(self.window)
        if button==glfw.MOUSE_BUTTON_RIGHT and action==glfw.RELEASE and self.dragging:
            self.dragging=False;self.events.put(('release',x,y));return
        if self._inside(x,y):
            if button==glfw.MOUSE_BUTTON_LEFT and action==glfw.PRESS:
                now=time.monotonic();last,px,py=self.last_click
                if now-last<.4 and np.hypot(x-px,y-py)<8:
                    self.events.put(('select',x,y));self.last_click=(-10.,x,y);return
                self.last_click=(now,x,y)
            if button==glfw.MOUSE_BUTTON_RIGHT and action==glfw.PRESS and self.selected is not None:
                self.dragging=True;self.events.put(('press',x,y));return
        if self.old_button:self.old_button(window,button,action,mods)
    def _move(self,window,x,y):
        if self.dragging:self.events.put(('move',x,y));return
        if self.old_move:self.old_move(window,x,y)
    def poll_events(self):
        # Native viewer owns GLFW event polling in its render thread.
        with self.lock():
            mujoco.mjv_updateScene(self.env.model,self.env.data,self.opt,None,self.cam,mujoco.mjtCatBit.mjCAT_ALL,self.scene)
            while not self.events.empty():
                kind,x,y=self.events.get()
                if kind=='select':
                    px,py,h=self._pixels(x,y);v=self.viewport;point=np.zeros(3)
                    geom=np.zeros(1,np.int32);flex=geom.copy();skin=geom.copy()
                    body=mujoco.mjv_select(self.env.model,self.env.data,self.opt,v.width/max(v.height,1),(px-v.left)/v.width,(h-py-v.bottom)/v.height,self.scene,point,geom,flex,skin)
                    self.selected=self.env.human_body_ids.index(body) if body in self.env.human_body_ids else None
                    print('COLUMN SELECTED',None if self.selected is None else self.selected+1,flush=True)
                elif kind=='press' and self.selected is not None:
                    self.drag_camera={name:getattr(self.cam,name) for name in ['type','fixedcamid','trackbodyid','distance','azimuth','elevation']}
                    self.drag_camera['lookat']=self.cam.lookat.copy()
                    point=self._point(x,y)
                    if point is not None:self.drag_offset=self.env._humans()[self.selected]-point
                elif kind=='move' and self.selected is not None:
                    self._restore_drag_camera()
                    point=self._point(x,y)
                    if point is not None:self.env.drag_column(self.selected,point+self.drag_offset)
                elif kind=='release':
                    if self.selected is not None:self.env.drag_column(self.selected,self.env._humans()[self.selected])
                    self._restore_drag_camera();self.drag_camera=None
                    self.selected=None;print('COLUMN RELEASED',flush=True)
                self.handle.perturb.active=0
    def sync(self):
        with self.lock():
            self.handle.perturb.active=0;self.handle.perturb.select=0
            self._restore_drag_camera()
        self.handle.sync()
        # Passive sync transfers state; refresh derived body/geometry positions.
        with self.lock():
            self._restore_drag_camera()
            mujoco.mj_forward(self.env.model,self.env.data)
    def close(self):self.handle.close()
