"""Learned robot localization/routing/control; no runtime robot AMCL or graph planner."""
from pathlib import Path
import numpy as np
import torch
from scipy.ndimage import binary_dilation
from gymnasium import spaces
from navrl.geometry import ROOT
from navrl.physical_env import PhysicalNavigationEnv
from navrl.multimap import MultiMapNavigationEnv
from navrl.map import continuous_remaining
from neural_navigation.models import LearnedLocalizer,NeuralRouter


class ResetMapView:
    """Legacy reset API needs a provisional path; this supplies no planned route.
    Every observation replaces the straight placeholder with neural routing outputs.
    """
    def __init__(self,m):self.base=m
    def __getattr__(self,key):return getattr(self.base,key)
    def plan(self,start,goal):return np.asarray([start,goal],float)


class LearnedNavigationEnv(PhysicalNavigationEnv):
    def __init__(self,components=None,**kwargs):
        if components is None:
            frozen=ROOT/'runs/full_model_navigation_v2'
            components=frozen if (frozen/'config.json').exists() else ROOT/'runs/neural_components_v2' if (ROOT/'runs/neural_components_v2/config.json').exists() else ROOT/'runs/neural_components'
        components=Path(components)
        kwargs.pop('localizer',None)
        super().__init__(localizer='learned_attention',localizer_factory=lambda m,l:LearnedLocalizer(m,l,components/'localizer.pt'),**kwargs)
        self.static_map=self.map;self.map=ResetMapView(self.static_map)
        d=torch.load(components/'router.pt',map_location='cpu',weights_only=True)
        self.router=NeuralRouter(d['iterations']);self.router.load_state_dict(d['state_dict']);self.router.eval()
        m=self.static_map;h,w=m.free.shape
        self.free=m.free[2:h//4*4:4,2:w//4*4:4].astype(np.float32)
        self.grid_res=m.res*4;self.grid_h,self.grid_w=self.free.shape
        self.expiry=np.zeros_like(self.free);self.last_route_time=-100.;self.plan_revision=0;self.routing_ready=False
        self.observation_space=spaces.Box(-1.,1.,shape=(231,),dtype=np.float32)

    def reset(self,**kwargs):
        self.expiry.fill(0);self.last_route_time=-100.;self.plan_revision=0;self.routing_ready=False
        obs,info=super().reset(**kwargs)
        options=kwargs.get('options') or {}
        prior=options.get('initial_pose')
        if prior is None and 'initial_prior_offset' in options:
            task=self.scene['tasks'][options.get('task',0)]
            prior=np.asarray(options.get('start',task['start']),float)+np.asarray(options['initial_prior_offset'],float)
        if prior is not None:
            prior=np.asarray(prior,float)
            if prior.shape!=(3,) or not np.isfinite(prior).all():raise ValueError('Initial pose must contain finite x, y, yaw')
            self.localizer.reset(prior,self.np_random)
            self.estimate=self.localizer.update(self.odom,self.scan)
            self.routing_ready=False
            obs=self._observation(self.scan,initialize=True);self.previous_remaining=self.remaining
            info=self._info('running');info['provided_initial_pose']=prior.tolist()
        return obs,info

    def set_navigation_goal(self, goal):
        """Switch mission target while preserving physics and learned localization."""
        goal=np.asarray(goal,float)
        if goal.shape!=(2,) or not np.isfinite(goal).all():
            raise ValueError('Goal must contain finite x and y')
        if not self.static_map.lookup(goal,self.static_map.free,outside=False):
            raise ValueError('Goal must lie in navigable known-map space')
        self.goal=goal.copy();self.last_route_time=-100.;self.routing_ready=False
        obs=self._observation(self.scan,initialize=False);self.previous_remaining=self.remaining
        return obs

    def cell(self,xy):
        r,c=self.static_map.cells(xy);return int(r)//4,int(c)//4

    def xy(self,cell):
        r,c=cell;return self.static_map.origin[:2]+np.array([(c+.5)*self.grid_res,(self.grid_h-r-.5)*self.grid_res])

    def _routing_dynamic_layer(self):
        now=self.elapsed;pose=self.estimate;yaw=pose[2];mount=self.lidar.mount
        valid=np.isfinite(self.scan)&(self.scan>=self.lidar.range_min)&(self.scan<self.lidar.range_max)
        origin=pose[:2]+[np.cos(yaw)*mount[0]-np.sin(yaw)*mount[1],np.sin(yaw)*mount[0]+np.cos(yaw)*mount[1]]
        angles=self.lidar.angles[valid]+yaw+mount[3]
        points=origin+self.scan[valid,None]*np.c_[np.cos(angles),np.sin(angles)]
        points=points[self.static_map.lookup(points,self.static_map.distance,outside=0.)>.22]
        for point in points:
            r,c=self.cell(point)
            if 0<=r<self.grid_h and 0<=c<self.grid_w:self.expiry[r,c]=now+getattr(self,'dynamic_memory_seconds',1.)
        self.unmatched_returns=len(points)
        return binary_dilation(self.expiry>now,iterations=2).astype(np.float32)

    def _neural_route(self):
        now=self.elapsed;pose=self.estimate;dyn=self._routing_dynamic_layer()
        gm=np.zeros_like(self.free);gr,gc=self.cell(self.goal)
        if 0<=gr<self.grid_h and 0<=gc<self.grid_w:gm[gr,gc]=1.
        pm=np.zeros_like(self.free);sr,sc=self.cell(pose[:2])
        if 0<=sr<self.grid_h and 0<=sc<self.grid_w:pm[sr,sc]=1.
        current=np.stack([self.free*(1-dyn),dyn,gm,pm]);static=np.stack([self.free,np.zeros_like(dyn),gm,pm])
        with torch.no_grad():value,q,reach=self.router(torch.from_numpy(np.stack([current,static])))
        q=q[0].numpy();reach=reach.numpy()
        self.current_probability=float(reach[0,0,sr,sc]) if 0<=sr<self.grid_h and 0<=sc<self.grid_w else 0.
        self.static_probability=float(reach[1,0,sr,sc]) if 0<=sr<self.grid_h and 0<=sc<self.grid_w else 0.
        path=[pose[:2].copy()];r,c=sr,sc;visited=set()
        for _ in range(160):
            if not (0<=r<self.grid_h and 0<=c<self.grid_w) or (r,c) in visited:break
            visited.add((r,c))
            if (r,c)==(gr,gc):path.append(self.goal.copy());break
            direction=int(np.argmax(q[:,r,c]));dr,dc=self.router.directions[direction];r+=dr;c+=dc
            if not (0<=r<self.grid_h and 0<=c<self.grid_w):break
            path.append(self.xy((r,c)))
        new=np.asarray(path)
        if self.routing_ready and (len(new)!=len(self.predicted_path) or not np.allclose(new,self.predicted_path,atol=.08)):self.plan_revision+=1
        self.predicted_path=new;self.routing_ready=True;self.last_route_time=now

    def _observation(self,scan,initialize=False):
        # Parent reset passes a scan before it sets the public scan attribute.
        self.scan=scan
        old=continuous_remaining(self.path,self.estimate[:2]) if self.routing_ready else None
        if initialize or self.elapsed-self.last_route_time>=.4:self._neural_route()
        self.path=self.predicted_path.copy()
        self.remaining=continuous_remaining(self.path,self.estimate[:2]) if len(self.path)>1 else float(np.linalg.norm(self.goal-self.estimate[:2]))
        if old is not None and hasattr(self,'previous_remaining'):self.previous_remaining+=self.remaining-old
        clean=np.nan_to_num(scan,nan=12.,posinf=12.,neginf=.1)
        binned=np.array([p.min() for p in np.array_split(clean,72)])/self.lidar.range_max
        if initialize:
            self.history.clear()
            for _ in range(3):self.history.append(binned.copy())
        else:self.history.append(binned)
        nearest=int(np.argmin(np.linalg.norm(self.path-self.estimate[:2],axis=1)))
        waypoint=self.path[min(nearest+2,len(self.path)-1)]
        yaw=self.estimate[2];rotation=np.array([[np.cos(yaw),np.sin(yaw)],[-np.sin(yaw),np.cos(yaw)]])
        local_waypoint=rotation@(waypoint-self.estimate[:2]);goal=rotation@(self.goal-self.estimate[:2])
        deviation=float(np.min(np.linalg.norm(self.path-self.estimate[:2],axis=1)))
        features=np.r_[local_waypoint/2,goal/12,self.measured_velocity/[.2,.5],self.last_action,self.localizer.std/[.5,.5,.5],deviation/2,self.static_probability,self.current_probability,min(self.remaining/20,1.)]
        return np.clip(np.r_[np.concatenate(self.history),features],-1,1).astype(np.float32)

    def _info(self,status):
        info=super()._info(status)
        if getattr(self,'routing_ready',False):
            state='model_goal_reached' if status=='success' else 'timeout' if status=='timeout' else 'model_predicts_blockage' if self.current_probability<.5 else 'model_following_route'
            info['decision']=dict(state=state,reason='learned_map_scan_pose_and_neural_route_values',plan_revision=self.plan_revision,
                static_reachability_probability=self.static_probability,current_reachability_probability=self.current_probability,
                localization_std=self.localizer.std.tolist(),unmatched_laser_returns=self.unmatched_returns,
                planner_source='trained_neural_convolutional_router',action_source='PPO',no_robot_amcl_or_astar=True)
        info['supervisor_override']=False;info['map_id']=self.scene.get('map_id','original')
        info['proposed_policy_action']=self.last_action.tolist()
        return info


class LearnedMultiMapEnv(MultiMapNavigationEnv):
    def __init__(self,components=None,**kwargs):
        self.components=components
        super().__init__(**kwargs)
    def _get(self,index):
        if index not in self.pool:self.pool[index]=LearnedNavigationEnv(components=self.components,scene_dir=self.entries[index]['directory'],**self.kwargs)
        return self.pool[index]
