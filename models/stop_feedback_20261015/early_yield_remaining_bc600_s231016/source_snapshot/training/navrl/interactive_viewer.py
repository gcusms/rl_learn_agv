"""MuJoCo GLFW viewer: double-click column, right-drag, release to deselect."""
from contextlib import nullcontext
import time
import glfw
import mujoco
import numpy as np
from navrl.mouse_drag import ground_point

class ManualNavigationViewer:
    def __init__(self,env,key_callback=None):
        self.env=env;self.key_callback=key_callback;self.selected=None;self.dragging=False
        self.last_click=(-10.,-100.,-100.);self.last_cursor=(0.,0.);self.drag_offset=np.zeros(2);self.texts=None
        if not glfw.init():raise RuntimeError('Cannot initialize MuJoCo desktop viewer')
        self.window=glfw.create_window(1400,900,'MuJoCo - Manual navigation: double-click column, right-drag',None,None)
        if not self.window:glfw.terminate();raise RuntimeError('Cannot open MuJoCo viewer')
        glfw.make_context_current(self.window);glfw.swap_interval(1)
        self.cam=mujoco.MjvCamera();self.opt=mujoco.MjvOption()
        self.scene=mujoco.MjvScene(env.model,maxgeom=7000);self.user_scn=mujoco.MjvScene(env.model,maxgeom=5000)
        self.context=mujoco.MjrContext(env.model,mujoco.mjtFontScale.mjFONTSCALE_150)
        glfw.set_key_callback(self.window,self._key);glfw.set_mouse_button_callback(self.window,self._button)
        glfw.set_cursor_pos_callback(self.window,self._move);glfw.set_scroll_callback(self.window,self._scroll)
    @property
    def viewport(self):
        w,h=glfw.get_framebuffer_size(self.window);return mujoco.MjrRect(0,0,w,h)
    def lock(self):return nullcontext()
    def is_running(self):return not glfw.window_should_close(self.window)
    def set_texts(self,texts):self.texts=texts
    def _pixels(self,x,y):
        w,h=glfw.get_window_size(self.window);fw,fh=glfw.get_framebuffer_size(self.window)
        return x*fw/max(w,1),y*fh/max(h,1),fh
    def _point(self,x,y,z=0.):
        px,py,h=self._pixels(x,y)
        cam=mujoco.mjv_averageCamera(self.scene.camera[0],self.scene.camera[1])
        return ground_point(px,py,h,self.viewport,cam,plane_z=z)
    def _key(self,window,key,scancode,action,mods):
        if action!=glfw.PRESS:return
        if key==glfw.KEY_ESCAPE:glfw.set_window_should_close(window,True)
        elif self.key_callback:self.key_callback(key)
    def _select(self,x,y):
        w,h=glfw.get_window_size(self.window);point=np.zeros(3);geom=np.zeros(1,np.int32);flex=np.zeros(1,np.int32);skin=np.zeros(1,np.int32)
        body=mujoco.mjv_select(self.env.model,self.env.data,self.opt,w/max(h,1),x/max(w,1),1-y/max(h,1),self.scene,point,geom,flex,skin)
        self.selected=self.env.human_body_ids.index(body) if body in self.env.human_body_ids else None
        print('COLUMN SELECTED',None if self.selected is None else self.selected+1,flush=True)
    def _button(self,window,button,action,mods):
        x,y=glfw.get_cursor_pos(window);self.last_cursor=(x,y)
        if button==glfw.MOUSE_BUTTON_LEFT and action==glfw.PRESS:
            now=time.monotonic();last,px,py=self.last_click
            if now-last<.4 and np.hypot(x-px,y-py)<8:self._select(x,y);self.last_click=(-10.,x,y)
            else:self.last_click=(now,x,y)
        if button==glfw.MOUSE_BUTTON_RIGHT:
            if action==glfw.PRESS and self.selected is not None:
                point=self._point(x,y)
                if point is not None:self.dragging=True;self.drag_offset=self.env._humans()[self.selected]-point
            elif action==glfw.RELEASE and self.selected is not None:
                # Stop following the mouse immediately; force servo decelerates physically.
                i=self.selected;self.env.drag_column(i,self.env._humans()[i]);self.selected=None;self.dragging=False
                print('COLUMN RELEASED',i+1,flush=True)
    def _move(self,window,x,y):
        oldx,oldy=self.last_cursor;self.last_cursor=(x,y)
        if self.dragging and self.selected is not None:
            point=self._point(x,y)
            if point is not None:self.env.drag_column(self.selected,point+self.drag_offset)
            return
        left=glfw.get_mouse_button(window,glfw.MOUSE_BUTTON_LEFT)==glfw.PRESS
        right=glfw.get_mouse_button(window,glfw.MOUSE_BUTTON_RIGHT)==glfw.PRESS
        middle=glfw.get_mouse_button(window,glfw.MOUSE_BUTTON_MIDDLE)==glfw.PRESS
        if not (left or right or middle):return
        shifted=glfw.get_key(window,glfw.KEY_LEFT_SHIFT)==glfw.PRESS or glfw.get_key(window,glfw.KEY_RIGHT_SHIFT)==glfw.PRESS
        if right:action=mujoco.mjtMouse.mjMOUSE_MOVE_H if shifted else mujoco.mjtMouse.mjMOUSE_MOVE_V
        elif left:action=mujoco.mjtMouse.mjMOUSE_ROTATE_H if shifted else mujoco.mjtMouse.mjMOUSE_ROTATE_V
        else:action=mujoco.mjtMouse.mjMOUSE_ZOOM
        h=glfw.get_window_size(window)[1]
        mujoco.mjv_moveCamera(self.env.model,action,(x-oldx)/max(h,1),(y-oldy)/max(h,1),self.scene,self.cam)
    def _scroll(self,window,dx,dy):
        if not self.dragging:mujoco.mjv_moveCamera(self.env.model,mujoco.mjtMouse.mjMOUSE_ZOOM,0,-.05*dy,self.scene,self.cam)
    def poll_events(self):glfw.poll_events()
    def sync(self):
        glfw.make_context_current(self.window)
        mujoco.mjv_updateScene(self.env.model,self.env.data,self.opt,None,self.cam,mujoco.mjtCatBit.mjCAT_ALL,self.scene)
        for i in range(self.user_scn.ngeom):
            if self.scene.ngeom>=self.scene.maxgeom:break
            src=self.user_scn.geoms[i];dst=self.scene.geoms[self.scene.ngeom];self.scene.ngeom+=1
            mujoco.mjv_initGeom(dst,src.type,src.size,src.pos,src.mat.reshape(-1),src.rgba)
            dst.category=src.category;dst.label=src.label
        mujoco.mjr_render(self.viewport,self.scene,self.context)
        if self.texts:
            _,_,left,right=self.texts
            mujoco.mjr_overlay(mujoco.mjtFont.mjFONT_NORMAL,mujoco.mjtGridPos.mjGRID_TOPLEFT,self.viewport,left or '',right or '',self.context)
        glfw.swap_buffers(self.window)
    def close(self):
        self.context.free();glfw.destroy_window(self.window);glfw.terminate()
