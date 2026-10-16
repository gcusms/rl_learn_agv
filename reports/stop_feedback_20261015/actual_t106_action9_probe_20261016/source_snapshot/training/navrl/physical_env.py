"""Native MuJoCo solid contacts; constrained planar robot and yielding pedestrians."""
import numpy as np
import mujoco
from .env import NavigationEnv
from .geometry import ROOT, wrap
from .map import StaticMap, path_features


class PhysicalNavigationEnv(NavigationEnv):
    def __init__(self,**kwargs):
        super().__init__(scene_file='physical_scene.xml',**kwargs)
        self.physics_backend='native_mujoco_planar_rigid_bodies'
        self.names=['robot']+[x['name'] for x in self.scene['dynamic_obstacles']]
        self.addresses={}
        for name in self.names:
            axes=['x','y','yaw'] if name=='robot' else ['x','y']
            joints=[mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_JOINT,name+'_'+a) for a in axes]
            self.addresses[name]=(np.array([self.model.jnt_qposadr[j] for j in joints]),np.array([self.model.jnt_dofadr[j] for j in joints]))
        self.human_map=StaticMap(self.scene_dir/'decision_map.yaml',radius=.30)
        self.free_cells=np.argwhere(self.human_map.free)
        self.robot_geoms=set(np.flatnonzero(self.model.body_weldid[self.model.geom_bodyid]==self.robot))
        self.wall_geoms={mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_GEOM,f'wall_{i}') for i in range(len(self.scene['walls']))}
        self.wall_boxes=np.asarray(self.scene['walls'])
        self.human_body_ids=[mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_BODY,n) for n in self.names[1:]]
        self.human_geoms=[mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_GEOM,x['name']+'_geom') for x in self.scene['dynamic_obstacles']]

    def _pose(self):
        return np.r_[self.data.xpos[self.robot,:2],self.data.qpos[self.addresses['robot'][0][2]]]

    def _sync(self):
        # Compatibility for intentional reset/testing placement only. Normal motion uses mj_step.
        q,_=self.addresses['robot']; base=self.model.body_pos[self.robot]
        self.data.qpos[q]=self.truth-[base[0],base[1],0.]
        mujoco.mj_forward(self.model,self.data)

    def robot_gap(self,point):
        delta=np.asarray(point)-self.truth[:2]; yaw=self.truth[2]
        local=np.array([np.cos(yaw)*delta[0]+np.sin(yaw)*delta[1],-np.sin(yaw)*delta[0]+np.cos(yaw)*delta[1]])
        half=np.array(self.scene['robot_collision_box_m'][:2])/2
        return float(np.linalg.norm(local-np.clip(local,-half,half))-.25)

    def wall_gap(self,point):
        if not len(self.wall_boxes):return np.inf
        nearest=np.clip(np.asarray(point),self.wall_boxes[:,:2],self.wall_boxes[:,2:])
        return float(np.min(np.linalg.norm(np.asarray(point)-nearest,axis=1))-.25)

    def _humans(self):
        return [self.data.xpos[bid,:2].copy() for bid in self.human_body_ids]

    def _new_route(self,index,away=False):
        pos=self._humans()[index]
        for _ in range(100):
            goal=self.human_map.xy(self.free_cells[int(self.np_random.integers(len(self.free_cells)))])
            if np.linalg.norm(goal-pos)<1.:continue
            if away and np.linalg.norm(goal-self.truth[:2])<np.linalg.norm(pos-self.truth[:2])+.5:continue
            try:path=self.human_map.plan(pos,goal)
            except ValueError:continue
            self.walkers[index]['path']=path;self.walkers[index]['goal']=goal;return
        self.walkers[index]['path']=np.array([pos,pos]);self.walkers[index]['goal']=pos

    def reset(self,seed=None,options=None):
        # Parent reset calls our placement hook after initializing the nominal prior.
        self.walkers=[];self.hard_guard_activations=0;self.native_contact_steps=0
        self.yield_steps=0;self.minimum_gap=np.inf;self.human_pauses=[]
        mujoco.mj_resetData(self.model,self.data)
        obs,info=super().reset(seed=seed,options=options)
        starts=[np.array([2.5,3.5]),np.array([4.8,2.])]
        for i,name in enumerate(self.names[1:]):
            q,d=self.addresses[name]
            spawn=starts[i] if i<len(starts) else np.asarray(self.scene['dynamic_obstacles'][i]['spawn_m'],float)
            point=spawn+self.np_random.uniform(-.12,.12,2) if self.dynamic else np.array([20.+i,20.])
            if self.dynamic:
                for _ in range(200):
                    prior=self._humans()[:i]
                    if self.human_map.lookup(point,self.human_map.free,outside=False) and self.robot_gap(point)>.35 and self.wall_gap(point)>.08 and all(np.linalg.norm(point-other)>.7 for other in prior):break
                    point=self.human_map.xy(self.free_cells[int(self.np_random.integers(len(self.free_cells)))])
                else:raise RuntimeError('Could not spawn separated physical pedestrians')
            self.data.qpos[q]=point;self.data.qvel[d]=0.
            self.walkers.append(dict(path=None,goal=point,speed=float(self.np_random.uniform(.5,.85)),
                                     state='walking' if self.dynamic else 'disabled',wait=0.,pause=0.,yield_count=0))
        mujoco.mj_forward(self.model,self.data)
        if self.dynamic:
            for i in range(len(self.walkers)):self._new_route(i)
        self.truth=self._pose();self.scan=self._scan();self.estimate=self.localizer.update(self.odom,self.scan)
        self.path=self.map.plan(self.estimate[:2],self.goal)
        obs=self._observation(self.scan,initialize=True);self.previous_remaining=self.remaining
        self.minimum_gap=self._minimum_gap()
        return obs,self._info('running')

    def _minimum_gap(self):
        if not self.dynamic:return np.inf
        positions=self._humans(); gaps=[]
        for i,p in enumerate(positions):
            gaps.extend([self.robot_gap(p),self.wall_gap(p)])
            for other in positions[:i]:gaps.append(float(np.linalg.norm(p-other)-.5))
        return min(gaps)

    def _walker_velocity(self,index,h):
        w=self.walkers[index];pos=self._humans()[index]
        block=w.get('block')
        if block is not None and block.get('mode')=='approach_cross_return':
            from .scripted_columns import velocity
            return velocity(self,index)
        if block is not None:
            group=[(j,other['block']) for j,other in enumerate(self.walkers) if 'block' in other and (j==index or (block.get('group') and other['block'].get('group')==block['group']))]
            positions=self._humans()
            near=any(np.linalg.norm(positions[j]-self.truth[:2])<=other['trigger_distance_m'] for j,other in group)
            if block['phase']=='armed' and (near or self.elapsed>=block['trigger_timeout_s']):
                for _,other in group:
                    if other['phase']=='armed':
                        other['phase']='holding';other['trigger_time_s']=float(self.elapsed)
                        other['release_time_s']=float(self.elapsed+other['hold_seconds'])
            if block['phase']=='armed' or (block['phase']=='holding' and self.elapsed<block['release_time_s']):
                w['state']='staging_block' if block['phase']=='armed' else 'temporary_block'
                return np.zeros(2)
            if block['phase']=='holding':block['phase']='releasing'
            if block['phase']=='releasing' and np.linalg.norm(pos-w['goal'])<.18:
                block['phase']='cleared';block['cleared_time_s']=float(self.elapsed)
        if w['pause']>0:
            w['pause']=max(0.,w['pause']-h);w['state']='pausing';return np.zeros(2)
        if np.linalg.norm(pos-w['goal'])<.18:
            self._new_route(index);w['pause']=float(self.np_random.uniform(.2,1.));w['state']='pausing';return np.zeros(2)
        target,_,_=path_features(w['path'],pos,self.human_map)
        delta=target-pos;norm=np.linalg.norm(delta);desired=delta/max(norm,1e-6)*min(w['speed'],2*norm)
        gap=self.robot_gap(pos);closing=np.dot(desired,self.truth[:2]-pos)>0
        # Stop before touching; hysteresis avoids alternating every substep.
        yielding=(closing and gap<.45) or (w['state']=='yielding' and gap<.55)
        if yielding and w['wait']<1.5:
            if w['state']!='yielding':w['yield_count']+=1
            w['state']='yielding';w['wait']+=h;return np.zeros(2)
        if yielding:
            # Change the pedestrian goal after yielding; otherwise it repeatedly
            # steps back then immediately walks toward its old goal through the robot.
            self._new_route(index,away=True)
            target,_,_=path_features(w['path'],pos,self.human_map)
            away=target-pos;away/=max(np.linalg.norm(away),1e-6)
            desired=away*.3;w['state']='giving_way';w['wait']=0.
        else:w['wait']=0.;w['state']='walking'
        positions=self._humans()
        for degrees in [0,30,-30,60,-60,90,-90,120,-120,180]:
            angle=np.deg2rad(degrees);c,s=np.cos(angle),np.sin(angle)
            vel=np.array([c*desired[0]-s*desired[1],s*desired[0]+c*desired[1]])
            future=pos+vel*.5
            if not self.human_map.visible(pos,future):continue
            if self.robot_gap(future)<.12:continue
            if any(i!=index and np.linalg.norm(future-other)<.64 for i,other in enumerate(positions)):continue
            return vel
        w['state']='blocked';w['wait']+=h;return np.zeros(2)

    def _set_velocity_force(self,name,desired,max_acceleration):
        _,d=self.addresses[name];current=self.data.qvel[d]
        acceleration=np.clip((desired-current)*20,-max_acceleration,max_acceleration)
        # Planar generalized inertia; force-based acceleration, not qpos teleportation.
        diag=self.data.qM[self.model.dof_Madr[d]]
        self.data.qfrc_applied[d]=diag*acceleration

    def _velocity_command(self,action):
        return np.array([max(0.,action[0])*.2,action[1]*.5])

    def step(self,action):
        action=np.asarray(action,float)
        if action.shape!=(2,) or not np.isfinite(action).all():raise ValueError('Action must be two finite normalized values')
        action=np.clip(action,-1,1);command=self._velocity_command(action)
        previous_action=self.last_action.copy();collided=False
        h=self.model.opt.timestep
        walker_velocities=[np.zeros(2) for _ in self.walkers]
        for substep in range(round(self.dt/h)):
            self.data.qfrc_applied[:]=0.;yaw=self.truth[2]
            desired=np.array([np.cos(yaw)*command[0]*self.v_scale,np.sin(yaw)*command[0]*self.v_scale,command[1]*self.w_scale])
            self._set_velocity_force('robot',desired,np.array([.5,.5,1.]))
            if self.dynamic:
                for i,name in enumerate(self.names[1:]):
                    if substep%10==0:walker_velocities[i]=self._walker_velocity(i,h*10)
                    self._set_velocity_force(name,walker_velocities[i],1.8)
                    if self.walkers[i]['state']=='yielding':self.yield_steps+=1
            oldq=self.data.qpos.copy();previous_pose=self.truth.copy()
            mujoco.mj_step(self.model,self.data);mujoco.mj_forward(self.model,self.data)
            self.truth=self._pose()
            native_contacts=[];penetration=False
            for c in self.data.contact:
                pair={int(c.geom1),int(c.geom2)}
                if pair&self.robot_geoms and (pair&self.wall_geoms or pair&set(self.human_geoms)):
                    native_contacts.append(c)
                if c.dist<-.0001:penetration=True
            if native_contacts:collided=True;self.native_contact_steps+=1
            # Extra guard against visible overlap at high relative speeds; previous configuration was valid.
            current_gap=self._minimum_gap()
            if penetration or current_gap<-.0001:
                self.data.qpos[:]=oldq;self.data.qvel[:]=0.;mujoco.mj_forward(self.model,self.data)
                self.truth=self._pose();self.hard_guard_activations+=1;current_gap=self._minimum_gap()
            self.minimum_gap=min(self.minimum_gap,current_gap)
            self.elapsed+=h
            _,d=self.addresses['robot'];vxy=self.data.qvel[d[:2]]
            self.velocity=np.array([np.dot(vxy,[np.cos(self.truth[2]),np.sin(self.truth[2])]),self.data.qvel[d[2]]])
            self.measured_velocity=self.velocity+ self.np_random.normal(0,[.002,.003])
            from .geometry import integrate
            self.odom=integrate(self.odom,*self.measured_velocity,h)
            self.travel+=float(np.linalg.norm(self.truth[:2]-previous_pose[:2]))
        if collided:
            self.collision_steps+=1
            if not self.in_contact:self.collision_events+=1
        self.in_contact=collided;self.steps+=1;self.last_action=action.copy()
        self.scan=self._scan();self.estimate=self.localizer.update(self.odom,self.scan)
        obs=self._observation(self.scan);progress=self.previous_remaining-self.remaining;self.previous_remaining=self.remaining
        clearance=float(np.min(np.nan_to_num(self.scan,nan=12.,posinf=12.)))
        reward=15*progress-.01-.01*np.sum((action-previous_action)**2)-.05*max(0.,(.45-clearance)/.45)
        arrived=np.linalg.norm(self.estimate[:2]-self.goal)<.12
        truncated=self.steps>=self.max_steps and not arrived
        if collided:reward-=1.
        if arrived:reward+=50
        if truncated:reward-=2
        return obs,float(reward),bool(arrived),bool(truncated),self._info('success' if arrived else 'timeout' if truncated else 'running')

    def _collision(self):
        return any(c.dist<=0 and {int(c.geom1),int(c.geom2)}&self.robot_geoms for c in self.data.contact)

    def _info(self,status):
        info=super()._info(status)
        info.update(column_block_events=[dict(w['block']) for w in self.walkers if 'block' in w],simulation_backend=self.physics_backend if hasattr(self,'physics_backend') else 'native_mujoco',
                    pedestrian_states=[w['state'] for w in self.walkers],
                    pedestrian_yield_events=sum(w['yield_count'] for w in self.walkers),
                    pedestrian_positions=[p.tolist() for p in self._humans()],
                    pedestrian_velocities=[self.data.qvel[self.addresses[n][1]].tolist() for n in self.names[1:]],
                    minimum_dynamic_gap_m=float(self.minimum_gap),
                    native_contact_substeps=self.native_contact_steps,
                    penetration_guard_activations=self.hard_guard_activations)
        return info


def make_navigation_env(legacy_planar=False,**kwargs):
    return NavigationEnv(**kwargs) if legacy_planar else PhysicalNavigationEnv(**kwargs)
