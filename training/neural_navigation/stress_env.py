"""Deterministic WORLD interventions and sensor faults, never policy inputs."""
import numpy as np
import mujoco
import weakref
from neural_navigation.room_env import RoomNavigationEnv


class CorruptedOdometry:
    def __init__(self,base,env):self.base=base;self.env=weakref.proxy(env)
    def __getattr__(self,name):return getattr(self.base,name)
    def reset(self,*args,**kwargs):return self.base.reset(*args,**kwargs)
    def update(self,odom,scan):
        scale=float(self.env.case.get('odom_scale',1.))
        bias=np.asarray(self.env.case.get('odom_bias_m_s',[0.,0.]),float)
        observed=np.asarray(odom,float).copy();observed[:2]=observed[:2]*scale+bias*self.env.elapsed
        observed[2]*=float(self.env.case.get('yaw_odom_scale',1.))
        self.env.observed_odometry=observed.copy()
        # Corrupt the sensor velocity too; no encoder feedback is presumed.
        if scale!=1.:self.env.measured_velocity=self.env.measured_velocity.copy()*[scale,1.]
        return self.base.update(observed,scan)
    def close(self):return self.base.close()


class StressNavigationEnv(RoomNavigationEnv):
    def __init__(self,case,**kwargs):
        self.case=case;self.intervention=None;self.sensor_fault_steps=0
        kwargs['dynamic']=case.get('obstacle') in ('sudden_block','crossing')
        super().__init__(**kwargs)
        self.localizer=CorruptedOdometry(self.localizer,self)

    def _new_route(self,index,away=False):
        point=self._humans()[index]
        self.walkers[index]['path']=np.array([point,point]);self.walkers[index]['goal']=point

    def reset(self,**kwargs):
        self.intervention=None;self.sensor_fault_steps=0;self.observed_odometry=np.zeros(3)
        obs,info=super().reset(**kwargs)
        for index,name in enumerate(self.names[1:]):
            q,d=self.addresses[name];self.data.qpos[q]=[20.+index*2,20.];self.data.qvel[d]=0.
        mujoco.mj_forward(self.model,self.data)
        self.laser_fusion.reset();self.expiry.fill(0);self.routing_ready=False
        self.scan=self._scan();self.estimate=self.localizer.update(self.odom,self.scan)
        obs=self._observation(self.scan,initialize=True);self.previous_remaining=self.remaining
        self.minimum_gap=self._minimum_gap()
        return obs,self._info('running')

    def _scan(self):
        scan=super()._scan()
        windows=list(self.case.get('dropout_windows',[]))
        if self.intervention is not None:
            windows.extend([[self.intervention['time_s']+a,self.intervention['time_s']+b] for a,b in self.case.get('dropout_after_event_s',[])])
        if any(a<=self.elapsed<b for a,b in windows):
            self.sensor_fault_steps+=1;return np.full_like(scan,np.nan)
        dropout=float(self.case.get('extra_scan_dropout',0.))
        if dropout:scan[self.np_random.random(len(scan))<dropout]=np.inf
        return scan

    def _introduce_obstacle(self):
        if self.travel<float(self.case.get('trigger_travel_m',.6)):return
        pose=self.truth;forward=np.array([np.cos(pose[2]),np.sin(pose[2])]);side=np.array([-forward[1],forward[0]])
        kind=self.case['obstacle']
        for distance in [1.2,1.4,1.,1.6]:
            center=pose[:2]+distance*forward
            for sign in [1.,-1.]:
                start=center+sign*.6*side if kind=='crossing' else center
                end=center-sign*.6*side if kind=='crossing' else center+sign*1.2*side
                if self.robot_gap(start)<.35 or self.wall_gap(start)<.05 or self.wall_gap(end)<.05:continue
                if not self.human_map.visible(start,end):continue
                q,d=self.addresses[self.names[1]];self.data.qpos[q]=start;self.data.qvel[d]=0.
                mujoco.mj_forward(self.model,self.data)
                self.intervention=dict(kind=kind,time_s=self.elapsed,start=start.tolist(),end=end.tolist(),
                                       speed_m_s=float(self.case.get('obstacle_speed_m_s',.12)),
                                       hold_s=float(self.case.get('hold_s',3.)) if kind=='sudden_block' else 0.,
                                       initial_robot_gap_m=float(self.robot_gap(start)))
                return

    def _walker_velocity(self,index,h):
        if index!=0:return np.zeros(2)
        if self.intervention is None:self._introduce_obstacle()
        if self.intervention is None:return np.zeros(2)
        event=self.intervention
        if self.elapsed-event['time_s']<event['hold_s']:return np.zeros(2)
        age=self.elapsed-event['time_s']
        for begin,end in self.case.get('actor_pause_after_event_s',[]):
            if begin<=age<end:
                event.setdefault('pause_started_time_s',float(self.elapsed))
                self.walkers[index]['state']='scripted_pause';return np.zeros(2)
            if age>=end and 'pause_started_time_s' in event:
                event.setdefault('pause_completed_time_s',float(self.elapsed))
        target=event['end']
        if 'reverse_after_event_s' in self.case and age>=self.case['reverse_after_event_s']:
            target=event['start'];event.setdefault('reversal_time_s',float(self.elapsed))
        point=self._humans()[index];delta=np.asarray(target)-point;distance=np.linalg.norm(delta)
        self.walkers[index]['state']='scripted_crossing' if distance>.04 else 'scripted_parked'
        if distance<.04:return np.zeros(2)
        return delta/distance*min(event['speed_m_s'],distance*2.)

    def _info(self,status):
        info=super()._info(status)
        info['stress_intervention']=self.intervention
        info['sensor_fault_steps']=self.sensor_fault_steps
        info['observed_odometry']=getattr(self,'observed_odometry',np.zeros(3)).tolist()
        return info
