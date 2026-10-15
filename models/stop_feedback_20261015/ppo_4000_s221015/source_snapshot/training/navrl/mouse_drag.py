"""Modifier-free X11 pointer input and ground-plane projection for MuJoCo."""
import ctypes,ctypes.util,os,subprocess
import numpy as np
import mujoco

class MousePointer:
    def __init__(self,window=None):
        self.lib=ctypes.CDLL(ctypes.util.find_library('X11'))
        self.lib.XOpenDisplay.argtypes=[ctypes.c_char_p];self.lib.XOpenDisplay.restype=ctypes.c_void_p
        self.lib.XCloseDisplay.argtypes=[ctypes.c_void_p]
        self.lib.XQueryPointer.argtypes=[ctypes.c_void_p,ctypes.c_ulong,ctypes.POINTER(ctypes.c_ulong),ctypes.POINTER(ctypes.c_ulong),*[ctypes.POINTER(ctypes.c_int)]*4,ctypes.POINTER(ctypes.c_uint)]
        self.lib.XGetGeometry.argtypes=[ctypes.c_void_p,ctypes.c_ulong,ctypes.POINTER(ctypes.c_ulong),ctypes.POINTER(ctypes.c_int),ctypes.POINTER(ctypes.c_int),*[ctypes.POINTER(ctypes.c_uint)]*4]
        self.display=self.lib.XOpenDisplay(None)
        if not self.display:raise RuntimeError('Cannot open desktop display for mouse dragging')
        if window is None:
            result=subprocess.check_output(['xdotool','search','--onlyvisible','--pid',str(os.getpid()),'--class','MuJoCo'],text=True)
            window=int(result.splitlines()[-1])
        self.window=window
    def read(self):
        root=ctypes.c_ulong();child=ctypes.c_ulong();rx=ctypes.c_int();ry=ctypes.c_int();x=ctypes.c_int();y=ctypes.c_int();mask=ctypes.c_uint()
        ok=self.lib.XQueryPointer(self.display,self.window,ctypes.byref(root),ctypes.byref(child),ctypes.byref(rx),ctypes.byref(ry),ctypes.byref(x),ctypes.byref(y),ctypes.byref(mask))
        gx=ctypes.c_int();gy=ctypes.c_int();w=ctypes.c_uint();h=ctypes.c_uint();border=ctypes.c_uint();depth=ctypes.c_uint()
        self.lib.XGetGeometry(self.display,self.window,ctypes.byref(root),ctypes.byref(gx),ctypes.byref(gy),ctypes.byref(w),ctypes.byref(h),ctypes.byref(border),ctypes.byref(depth))
        return x.value,y.value,h.value,bool(ok and mask.value&256)
    def close(self):
        if self.display:self.lib.XCloseDisplay(self.display);self.display=None

def ground_point(x,y,window_height,viewport,glcamera,plane_z=0.):
    if not viewport or viewport.width<=0 or viewport.height<=0:return None
    u=(x-viewport.left)/viewport.width;v=(window_height-y-viewport.bottom)/viewport.height
    if not (0<=u<=1 and 0<=v<=1):return None
    forward=np.asarray(glcamera.forward);up=np.asarray(glcamera.up);right=np.cross(forward,up)
    height=glcamera.frustum_top-glcamera.frustum_bottom
    horizontal=(u-.5)*height*viewport.width/viewport.height+glcamera.frustum_center
    vertical=glcamera.frustum_bottom+v*height
    origin=np.asarray(glcamera.pos).copy()
    if glcamera.orthographic:
        origin+=right*horizontal+up*vertical;direction=forward
    else:direction=forward*glcamera.frustum_near+right*horizontal+up*vertical
    if abs(direction[2])<1e-8:return None
    distance=(plane_z-origin[2])/direction[2]
    return (origin+distance*direction)[:2] if distance>0 else None
