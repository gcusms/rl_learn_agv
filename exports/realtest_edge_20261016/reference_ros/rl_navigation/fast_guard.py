"""Batch the existing nine-pose guard; retain its geometry and decisions."""
import numpy as np
from neural_navigation.runtime import EmergencyStop
from .fast_footprint import static_footprint_clear


class BatchedEmergencyStop(EmergencyStop):
    def _trajectory_check(self,core,command,points,margins,age,allow_escape=False):
        # Other model contracts retain their reference implementation.
        if self.motion_response_schema is not None:
            return super()._trajectory_check(core,command,points,margins,age,allow_escape)
        initial=np.linalg.norm(points-np.clip(points,-self.half,self.half),axis=1)
        rotation_escape=bool(self.clearance_increasing_rotation and abs(command[0])<1e-8
                             and abs(command[1])>.005 and np.all(initial>0))
        escaping=bool(allow_escape and (command[0]>.005 or rotation_escape) and np.any(initial<margins))
        minimum=np.minimum(initial,margins) if escaping else margins
        horizon=abs(command[0])/self.braking_acceleration+max(age,0)+.2
        times=np.linspace(0,horizon,9)
        v,w=command;pose=core.estimate;yaw=pose[2]+w*times
        local=np.empty((9,3));world=np.empty((9,3))
        if abs(w)<1e-9:
            local[:,0]=v*times;local[:,1]=0.
            world[:,0]=pose[0]+v*times*np.cos(pose[2])
            world[:,1]=pose[1]+v*times*np.sin(pose[2])
        else:
            local[:,0]=v/w*np.sin(w*times);local[:,1]=-v/w*(np.cos(w*times)-1.)
            world[:,0]=pose[0]+v/w*(np.sin(yaw)-np.sin(pose[2]))
            world[:,1]=pose[1]-v/w*(np.cos(yaw)-np.cos(pose[2]))
        local[:,2]=(w*times+np.pi)%(2*np.pi)-np.pi
        world[:,2]=(yaw+np.pi)%(2*np.pi)-np.pi
        c,s=np.cos(local[:,2,None]),np.sin(local[:,2,None])
        dx=points[None,:,0]-local[:,None,0];dy=points[None,:,1]-local[:,None,1]
        relative=np.stack([dx*c+dy*s,-dx*s+dy*c],axis=-1)
        gaps=np.linalg.norm(relative-np.clip(relative,-self.half,self.half),axis=2)
        laser_clear=~np.any(gaps<minimum[None,:]-1e-4,axis=1)
        if self.static_footprint_schema is None:
            static_clear=core.static_map.lookup(world[:,:2],core.static_map.free,outside=False)
        else:
            radius=np.linalg.norm(self.half+self.static_margin)
            sweep_margin=(abs(v)+abs(w)*radius)*horizon/16.
            std=np.asarray(core.localizer.std,float)
            margin=self.static_margin+sweep_margin+self.static_pose_sigma_scale*(max(std[:2])+radius*std[2])
            # Keep the original order of stop reasons at each trajectory pose.
            static_clear=None
        for index in range(9):
            clear=(bool(static_clear[index]) if static_clear is not None else
                   static_footprint_clear(core.static_map,world[index],self.half*2,margin))
            if not clear:return 'static_braking_footprint'
            if not laser_clear[index]:return 'laser_braking_footprint'
        if escaping and np.any(gaps[-1,initial<margins]<initial[initial<margins]+.005):
            return 'escape_not_increasing_clearance'
        return None


def install_acceleration():
    """Apply only inside the inference process; never edit the training tree."""
    from neural_navigation import runtime
    runtime.static_footprint_clear=static_footprint_clear
    runtime.EmergencyStop=BatchedEmergencyStop
