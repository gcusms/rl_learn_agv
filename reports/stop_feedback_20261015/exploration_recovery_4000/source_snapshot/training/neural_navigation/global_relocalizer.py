"""Bounded known-map pose recovery, separate from the tracking/model contract.

This experimental module never commands motion or reads simulator truth. A
complete discrete correlative search proposes a pose; fresh scans must confirm
it while the robot remains stopped. Ambiguous, incomplete and weak geometry
remain a hold, rather than an arbitrary map/odom jump. This is NOT SLAM or a
learned localization model. Thresholds are provisional, not calibrated bounds.
"""
import heapq
import time

import numpy as np
from scipy.ndimage import maximum_filter

from navrl.geometry import wrap
from neural_navigation.fast_static_localizer import FastStaticLocalizer, sparse_contract
from neural_navigation.static_footprint import static_footprint_clear


def contract():
    return dict(schema='bounded_known_map_relocalization_v1', experimental=True,
        beams=60, sensor_max_range_m=10., angular_step_rad=float(np.deg2rad(2.)),
        branch_depth=4, branch_batch=64, score_sigma_m=.12, minimum_score=.60,
        uniqueness_gap=.06, retained_score_window=.10,
        mode_translation_m=.40, mode_rotation_rad=.25, maximum_modes=16,
        minimum_hits=24, minimum_static_fraction=.65, static_gate_m=.18,
        maximum_ray_contradiction_fraction=.10, ray_endpoint_allowance_m=.18,
        maximum_search_cpu_s=4., maximum_nodes=200000,
        footprint_m=[.393924904,.329460022], margin_m=.04,
        maximum_scan_age_s=.25, stopped_linear_m_s=.02, stopped_angular_rad_s=.05,
        maximum_stopped_translation_m=.05, maximum_stopped_rotation_rad=.08,
        confirmation_frames=3, confirmation_span_s=.40,
        confirmation_translation_m=.08, confirmation_rotation_rad=.08,
        thresholds_calibrated=False, simulator_truth_used=False,
        motion_commands_issued=False, unknown_is_free=False)


def propagate_pose(pose, previous_odom, odom):
    """Use relative full planar odometry, including measured lateral slip."""
    pose=np.asarray(pose,float);previous_odom=np.asarray(previous_odom,float)
    odom=np.asarray(odom,float)
    angle=pose[2]-previous_odom[2];c,s=np.cos(angle),np.sin(angle)
    delta=(odom[:2]-previous_odom[:2])@np.array([[c,s],[-s,c]])
    return np.r_[pose[:2]+delta,wrap(pose[2]+wrap(odom[2]-previous_odom[2]))]


class GlobalRelocalizer:
    """Map preprocessing is cached; search jobs advance in small CPU slices.

    Max-filtered fields give upper bounds for every translation in a block,
    at each explicitly enumerated yaw. Search completion only refers to this
    finite lattice/endpoint score. It is not a continuous pose optimality or
    probabilistic confidence guarantee. Large maps may exhaust the budget.
    """
    def __init__(self,static_map,lidar,config=None):
        self.config=contract() if config is None else config
        if self.config!=contract():raise ValueError('Unsupported relocalization contract')
        self.map=static_map;self.lidar=lidar
        self.indices=np.linspace(0,lidar.beams-1,self.config['beams']).astype(int)
        self.maximum_range=min(lidar.range_max,self.config['sensor_max_range_m'])
        self.verifier=FastStaticLocalizer(static_map,lidar,sparse_contract())
        # Unknown/padded cells are not physical walls and never earn hit score.
        base=np.exp(-.5*(static_map.distance/self.config['score_sigma_m'])**2)
        base[static_map.unknown]=0.
        self.padding=2**self.config['branch_depth']
        padded=np.pad(base.astype(np.float32),self.padding)
        self.fields=[padded]
        radius=min(self.config['footprint_m'])/2+self.config['margin_m']
        possible=(static_map.clearance>=radius)&~static_map.blocked
        self.possible=[possible]
        for level in range(1,self.config['branch_depth']+1):
            size=2**level
            self.fields.append(maximum_filter(padded,size=size,origin=-(size//2),mode='constant',cval=0.))
            self.possible.append(maximum_filter(possible,size=size,origin=-(size//2),mode='constant',cval=False))
        count=int(np.ceil(2*np.pi/self.config['angular_step_rad']))
        self.yaws=np.linspace(-np.pi,np.pi,count,endpoint=False)

    def body_points(self,scan):
        scan=np.asarray(scan,float)
        if scan.shape!=(self.lidar.beams,):raise ValueError('Scan contract mismatch')
        selected=scan[self.indices]
        valid=np.isfinite(selected)&(selected>=self.lidar.range_min)&(selected<self.maximum_range)
        angles=self.lidar.angles[self.indices[valid]]+self.lidar.mount[3]
        points=selected[valid,None]*np.c_[np.cos(angles),np.sin(angles)]+self.lidar.mount[:2]
        return points,valid

    def start(self,scan):
        return PoseSearch(self,np.asarray(scan,float).copy())

    def _scores(self,nodes,offsets,level):
        nodes=np.asarray(nodes,int).reshape(-1,3)
        yaw,row,col=nodes.T
        endpoints=offsets[yaw]+np.stack([row,col],axis=1)[:,None,:]
        rr,cc=endpoints[:,:,0],endpoints[:,:,1]
        # A block may start outside the map while its children hit inside it.
        # Padded max fields keep that case an upper bound, not a false zero.
        valid=(rr>=-(2**level-1))&(rr<self.map.height)&(cc>=-(2**level-1))&(cc<self.map.width)
        field=self.fields[level]
        values=field[np.clip(rr+self.padding,0,len(field)-1),np.clip(cc+self.padding,0,field.shape[1]-1)]
        scores=np.mean(np.where(valid,values,0.),axis=1)
        inside=(row>=0)&(row<self.map.height)&(col>=0)&(col<self.map.width)
        allowed=self.possible[level][np.clip(row,0,self.map.height-1),np.clip(col,0,self.map.width-1)]
        return np.where(inside&allowed,scores,-np.inf)

    def _ray_contradictions(self,pose,scan):
        """Finite rays must not pass through mapped occupied cells.

        New unmapped objects may shorten a ray. NaN/Inf are unknown here, not
        proof of free space. Unknown cells cannot themselves earn a match.
        Half-cell sampling is a consistency diagnostic, not a continuous-body
        collision certificate or permission to issue motion.
        """
        values=np.asarray(scan)[self.indices]
        valid=np.isfinite(values)&(values>=self.lidar.range_min)&(values<self.maximum_range)
        values=values[valid];angles=self.lidar.angles[self.indices[valid]]+self.lidar.mount[3]+pose[2]
        c,s=np.cos(pose[2]),np.sin(pose[2])
        origin=pose[:2]+self.lidar.mount[:2]@np.array([[c,s],[-s,c]])
        distance=np.arange(0.,self.maximum_range,self.map.res/2)
        active=distance[None,:]<np.maximum(0.,values[:,None]-self.config['ray_endpoint_allowance_m'])
        points=origin+distance[None,:,None]*np.c_[np.cos(angles),np.sin(angles)][:,None,:]
        occupied=self.map.lookup(points,self.map.occupied,outside=False)
        return float(np.mean(np.any(occupied&active,axis=1))) if len(values) else 1.

    def verify(self,pose,scan,refine=True):
        points,valid=self.body_points(scan);pose=np.asarray(pose,float).copy()
        if len(points)<self.config['minimum_hits']:
            return dict(valid=False,reason='insufficient_scan_hits',pose=pose.tolist())
        self.verifier.reset(pose,None)
        if refine:
            pose=self.verifier.update(np.zeros(3),scan,0.)
        else:
            endpoints=self.verifier._points(pose,points)
            residual=abs(self.verifier._field(endpoints)[0])
            keep=(residual<=self.config['static_gate_m'])&self.map.lookup(endpoints,~self.map.unknown,outside=False)
            if keep.any():
                _,jac,weight=self.verifier._linearize(pose,points[keep])
                self.verifier._basis(jac,weight)
        endpoints=self.verifier._points(pose,points)
        residual=abs(self.verifier._field(endpoints)[0])
        known=self.map.lookup(endpoints,~self.map.unknown,outside=False)
        support=(residual<=self.config['static_gate_m'])&known
        if support.any():
            _,jac,weight=self.verifier._linearize(pose,points[support])
            self.verifier._basis(jac,weight)
        fraction=float(support.mean());rank=self.verifier.observable_rank
        clear=static_footprint_clear(self.map,pose,self.config['footprint_m'],self.config['margin_m'])
        contradiction=self._ray_contradictions(pose,scan)
        reason=('insufficient_static_support' if fraction<self.config['minimum_static_fraction']
            else 'weak_geometry' if rank<3 else 'blocked_footprint' if not clear
            else 'free_space_contradiction' if contradiction>self.config['maximum_ray_contradiction_fraction'] else None)
        return dict(valid=reason is None,reason=reason,pose=pose.tolist(),static_fraction=fraction,
            observable_rank=rank,ray_contradiction_fraction=contradiction,footprint_clear=clear,
            hit_count=len(points))


class PoseSearch:
    def __init__(self,matcher,scan):
        self.matcher=matcher;self.scan=scan;self.result=None;self.done=False
        self.cpu_seconds=0.;self.nodes_scored=0;self.maximum_slice_cpu_s=0.
        self._iterator=self._run()

    def advance(self,cpu_budget_s=.008):
        if not np.isfinite(cpu_budget_s) or not 0<cpu_budget_s<=.05:
            raise ValueError('Search slice must be positive and <=50ms CPU')
        start=time.process_time();self._slice_start=start
        while not self.done:
            try:next(self._iterator)
            except StopIteration:self.done=True
            now=time.process_time()
            if now-start>=cpu_budget_s:break
        duration=time.process_time()-start
        self.cpu_seconds+=duration;self.maximum_slice_cpu_s=max(self.maximum_slice_cpu_s,duration)
        if self.result is not None:
            self.result.update(cpu_seconds=self.cpu_seconds,nodes_scored=self.nodes_scored,
                maximum_slice_cpu_ms=self.maximum_slice_cpu_s*1000)
        return self.done

    def _run(self):
        m=self.matcher;cfg=m.config;points,_=m.body_points(self.scan)
        if len(points)<cfg['minimum_hits']:
            self.result=dict(candidate_valid=False,search_complete=True,reason='insufficient_scan_hits',modes=[])
            return
        rotated=[]
        for yaw in m.yaws:
            c,s=np.cos(yaw),np.sin(yaw);xy=points@np.array([[c,s],[-s,c]])
            rotated.append(np.rint(np.c_[-xy[:,1],xy[:,0]]/m.map.res).astype(int))
        offsets=np.asarray(rotated);level=cfg['branch_depth'];stride=2**level
        rows,cols=np.meshgrid(np.arange(0,m.map.height,stride),np.arange(0,m.map.width,stride),indexing='ij')
        base=np.c_[rows.ravel(),cols.ravel()];heap=[]
        exhausted=False;best=-np.inf;leaves=[]
        # Roots cover every possible translation, independently at every yaw.
        for yi in range(len(m.yaws)):
            for index in range(0,len(base),128):
                if self._exhausted():
                    self.result=dict(candidate_valid=False,search_complete=False,reason='search_budget_exhausted',modes=[])
                    return
                block=base[index:index+128];nodes=np.c_[np.full(len(block),yi),block]
                scores=m._scores(nodes,offsets,level);self.nodes_scored+=len(nodes)
                for node,score in zip(nodes,scores):
                    if score>=cfg['minimum_score']:heapq.heappush(heap,(-float(score),level,*map(int,node)))
                yield
        while heap:
            cutoff=max(cfg['minimum_score'],best-cfg['retained_score_window'])
            if -heap[0][0]<cutoff:break
            if self._exhausted():
                exhausted=True;break
            batch=[]
            for _ in range(min(cfg['branch_batch'],len(heap))):
                if -heap[0][0]<cutoff:break
                bound,level,yi,row,col=heapq.heappop(heap)
                if level==0:
                    # A level-zero bound already IS its exact lattice score.
                    score=-bound;best=max(best,score);leaves.append((score,yi,row,col))
                else:batch.append((level,yi,row,col))
            cutoff=max(cfg['minimum_score'],best-cfg['retained_score_window'])
            for level in range(1,cfg['branch_depth']+1):
                parents=np.array([row[1:] for row in batch if row[0]==level],int).reshape(-1,3)
                if not len(parents):continue
                step=2**(level-1)
                shifts=np.array([[0,dr,dc] for dr in (0,step) for dc in (0,step)])
                nodes=(parents[:,None,:]+shifts).reshape(-1,3)
                scores=m._scores(nodes,offsets,level-1);self.nodes_scored+=len(nodes)
                for node,score in zip(nodes,scores):
                    if score>=cutoff:heapq.heappush(heap,(-float(score),level-1,*map(int,node)))
            yield
        if exhausted:
            # Do not spend another unbounded verification pass on an already
            # rejected partial search. A partial best pose is never usable.
            self.result=dict(candidate_valid=False,search_complete=False,reason='search_budget_exhausted',
                pose=None,modes=[])
            return
        modes=[]
        for score,yi,row,col in sorted(leaves,reverse=True):
            if score<best-cfg['retained_score_window']:break
            pose=np.r_[m.map.xy((row,col)),m.yaws[yi]]
            if any(np.linalg.norm(pose[:2]-np.array(other['lattice_pose'][:2]))<cfg['mode_translation_m']
                and abs(wrap(pose[2]-other['lattice_pose'][2]))<cfg['mode_rotation_rad'] for other in modes):continue
            modes.append(dict(score=score,lattice_pose=pose.tolist()))
            if len(modes)>=cfg['maximum_modes']:break
        for mode in modes:
            mode.update(m.verify(mode['lattice_pose'],self.scan));yield
        supported=[mode for mode in modes if mode['valid']]
        winner=supported[0] if supported else None
        # Compare all spatially distinct hypotheses, including a mode that the
        # secondary verifier rejects. Rejecting ambiguity is intentionally
        # conservative; a near tie is never hidden by retaining just one pose.
        competitors=[] if winner is None else [mode['score'] for mode in modes if mode is not winner]
        gap=(winner['score']-max(competitors)) if competitors else cfg['retained_score_window']
        reason=('search_budget_exhausted' if exhausted else 'no_supported_pose' if winner is None
            else 'ambiguous_map' if gap<cfg['uniqueness_gap'] else None)
        self.result=dict(candidate_valid=reason is None,search_complete=not exhausted,reason=reason,
            pose=winner['pose'] if reason is None else None,score_gap=float(gap),modes=modes)

    def _exhausted(self):
        return (self.cpu_seconds+time.process_time()-self._slice_start>self.matcher.config['maximum_search_cpu_s']
            or self.nodes_scored>=self.matcher.config['maximum_nodes'])


class StoppedPoseRecovery:
    """Explicit recovery session: callers must hold zero velocity throughout.

    This API does not silently attach itself to archived navigation bundles.
    A lost-pose detector/supervisor invokes begin(), keeps issuing zero through
    its existing watchdog, and accepts a confirmed reset only via take_pose().
    Equal timestamps do not count as fresh evidence. Detection of all possible
    kidnappings from a partial, ambiguous scan is not promised.
    """
    def __init__(self,matcher):
        self.matcher=matcher;self.config=matcher.config;self.reset()

    def reset(self):
        self.state='idle';self.reason=None;self.job=None;self.pose=None
        self.confirmations=0;self.confirmed_since=None;self.last_timestamp=None
        self.anchor_odom=None;self.anchor_pose=None;self.latest_odom=None

    def _input_reason(self,scan,odom,velocity,timestamp,received_timestamp):
        if np.asarray(scan).shape!=(self.matcher.lidar.beams,) or np.asarray(odom).shape!=(3,) or np.asarray(velocity).shape!=(2,):
            raise ValueError('Recovery sensor contract mismatch')
        if not np.isfinite(np.r_[odom,velocity,timestamp,received_timestamp]).all():return 'nonfinite_sensor_state'
        age=received_timestamp-timestamp
        if age<-.05 or age>self.config['maximum_scan_age_s']:return 'stale_or_future_scan'
        if abs(velocity[0])>self.config['stopped_linear_m_s'] or abs(velocity[1])>self.config['stopped_angular_rad_s']:return 'robot_not_stopped'
        if self.anchor_odom is not None and (np.linalg.norm(np.asarray(odom)[:2]-self.anchor_odom[:2])>self.config['maximum_stopped_translation_m']
            or abs(wrap(odom[2]-self.anchor_odom[2]))>self.config['maximum_stopped_rotation_rad']):return 'odometry_moved_during_recovery'
        return None

    def begin(self,scan,odom,velocity,timestamp,received_timestamp=None):
        self.reset();received_timestamp=timestamp if received_timestamp is None else received_timestamp
        reason=self._input_reason(scan,odom,velocity,timestamp,received_timestamp)
        if reason:self.state='held';self.reason=reason;return
        self.anchor_odom=np.asarray(odom,float).copy();self.latest_odom=self.anchor_odom.copy()
        self.last_timestamp=float(timestamp);self.job=self.matcher.start(scan)
        self.state='searching'

    def advance(self,cpu_budget_s=.008):
        if self.state=='searching' and self.job.advance(cpu_budget_s):
            if not self.job.result['candidate_valid']:
                self.state='held';self.reason=self.job.result['reason']
            else:
                self.anchor_pose=np.asarray(self.job.result['pose'],float)
                self.state='confirming'
        return self.state

    def observe(self,scan,odom,velocity,timestamp,received_timestamp=None):
        if self.state not in ('searching','confirming','ready'):return self.state
        received_timestamp=timestamp if received_timestamp is None else received_timestamp
        reason=self._input_reason(scan,odom,velocity,timestamp,received_timestamp)
        if self.last_timestamp is not None and timestamp<self.last_timestamp:reason='backward_timestamp'
        if reason:
            self.state='held';self.reason=reason;self.pose=None;return self.state
        if timestamp==self.last_timestamp:return self.state
        self.last_timestamp=float(timestamp);self.latest_odom=np.asarray(odom,float).copy()
        if self.state=='searching':return self.state
        predicted=propagate_pose(self.anchor_pose,self.anchor_odom,odom)
        check=self.matcher.verify(predicted,scan)
        delta=np.asarray(check['pose'])-predicted;delta[2]=wrap(delta[2])
        if not check['valid'] or np.linalg.norm(delta[:2])>self.config['confirmation_translation_m'] or abs(delta[2])>self.config['confirmation_rotation_rad']:
            self.state='held';self.reason=check['reason'] or 'inconsistent_confirmation';self.pose=None;return self.state
        if self.confirmed_since is None:self.confirmed_since=float(timestamp)
        self.confirmations+=1
        if self.confirmations>=self.config['confirmation_frames'] and timestamp-self.confirmed_since>=self.config['confirmation_span_s']-1e-9:
            self.pose=np.asarray(check['pose'],float);self.state='ready'
        return self.state

    def take_pose(self,timestamp):
        """One-shot pose proposal; caller must reset all map-frame histories.

        No command is released here. A ready pose expires unless the caller
        continues to supply fresh stationary observations.
        """
        if self.state!='ready':raise RuntimeError('No confirmed relocalization available')
        if not np.isfinite(timestamp) or timestamp<self.last_timestamp or timestamp-self.last_timestamp>self.config['maximum_scan_age_s']:
            self.state='held';self.reason='expired_confirmation';self.pose=None
            raise RuntimeError('Confirmed pose has expired')
        pose=self.pose.copy();self.state='consumed';self.pose=None
        return pose

    def diagnostics(self):
        return dict(state=self.state,reason=self.reason,required_command=[0.,0.],
            confirmations=self.confirmations,search=None if self.job is None else self.job.result,
            pose=None if self.pose is None else self.pose.tolist(),deployment_approved=False)


def apply_confirmed_pose(runtime,recovery,scan,odom,velocity,timestamp,received_timestamp=None):
    """Explicit experimental reset; return a stopped frame, never an actor move.

    Existing localizer/model schemas are untouched. This supplies a new initial
    map pose through the normal reset API, clearing every map-frame cache and
    policy history. The caller retains the mission deadline and must publish
    the usual map/odom transform consistently before resuming on a later frame.
    """
    from neural_navigation.safety_context import SafetyContext
    core=runtime.core
    if not isinstance(core.localizer,FastStaticLocalizer) or recovery.matcher.map is not core.static_map:
        raise ValueError('Explicit recovery requires the same immutable map and fast tracking runtime')
    if recovery.state!='ready':raise RuntimeError('No confirmed relocalization available')
    received_timestamp=timestamp if received_timestamp is None else received_timestamp
    reason=recovery._input_reason(scan,odom,velocity,timestamp,received_timestamp)
    if reason or timestamp!=recovery.last_timestamp or not np.allclose(odom,recovery.latest_odom,atol=1e-9,rtol=0):
        raise ValueError('Reset must use the latest confirmed fresh stationary sensor frame')
    check=recovery.matcher.verify(recovery.pose,scan,refine=False)
    if not check['valid']:raise ValueError('Latest scan no longer confirms the recovered pose')
    goal=np.r_[core.goal,core.goal_yaw] if core.goal_yaw is not None else core.goal.copy()
    pose=recovery.take_pose(received_timestamp)
    core.reset(pose,goal,timestamp)
    if runtime.guard.memory is not None:runtime.guard.memory.reset()
    if runtime.guard.predictive_constraint is not None:runtime.guard.predictive_constraint.reset()
    runtime.guard.reverse_started=None
    if runtime.context is not None:
        capture=getattr(runtime.context,'capture_decision_feedback',False)
        runtime.context=SafetyContext(runtime.context.schema)
        runtime.context.capture_decision_feedback=capture
    prediction=getattr(runtime.policy,'prediction',None)
    if prediction is not None:prediction.reset();runtime.policy.last_observation=None
    reset_hook=getattr(runtime.policy,'on_runtime_reset',None)
    if reset_hook is not None:reset_hook(runtime.context)
    # The first fresh scan installs current odometry as the new incremental
    # anchor. It is not integrated from zero and cannot invent a displacement.
    core.update(scan,odom,velocity,timestamp);core.record_action(np.zeros(2))
    angle=wrap(core.estimate[2]-odom[2]);c,s=np.cos(angle),np.sin(angle)
    translation=core.estimate[:2]-np.asarray(odom)[:2]@np.array([[c,s],[-s,c]])
    return dict(command=[0.,0.],stop_reason='relocalization_reset',goal=goal.tolist(),
        map_T_odom=[*translation,float(angle)],diagnostics=core.diagnostics(),
        deployment_approved=False)
