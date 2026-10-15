"""Recovery-specific invariants; no training, ROS publishers or robot truth inputs."""
import unittest
import numpy as np

from navrl.map import StaticMap
from navrl.geometry import wrap
from neural_navigation.sensors import LidarSpec
from neural_navigation.global_relocalizer import GlobalRelocalizer,StoppedPoseRecovery,propagate_pose


def toy_map(symmetric=False):
    cells=np.zeros((64,80),np.int8)
    cells[:2,:]=100;cells[-2:,:]=100;cells[:,:2]=100;cells[:,-2:]=100
    if not symmetric:
        cells[12:15,15:38]=100;cells[13:30,35:38]=100
        cells[40:48,48:53]=100;cells[36:39,15:20]=100
    return StaticMap.from_occupancy_grid(cells.ravel(),80,64,.1,[-1.,-2.,0.])


def lidar():
    return LidarSpec(dict(beams=220,angle_min=-1.8915926535897931,angle_max=1.938407346410207,
        range_min=.1,range_max=10.,mount=[.118389,0.,.116682537,0.]))


def grid_scan(m,spec,pose):
    # Independent ray forward model for unit fixtures only. Capability results
    # use frozen actual MuJoCo rays, not endpoints constructed from a matcher.
    c,s=np.cos(pose[2]),np.sin(pose[2])
    origin=pose[:2]+spec.mount[:2]@np.array([[c,s],[-s,c]])
    ranges=np.arange(spec.range_min,10.,.01)
    angle=pose[2]+spec.angles+spec.mount[3]
    points=origin+ranges[None,:,None]*np.c_[np.cos(angle),np.sin(angle)][:,None,:]
    hits=m.lookup(points,m.occupied,outside=False)
    return np.where(hits.any(axis=1),ranges[np.argmax(hits,axis=1)],np.inf)


def finish(job):
    while not job.advance():pass
    return job.result


class RelocalizationTests(unittest.TestCase):
    def setUp(self):
        self.map=toy_map();self.lidar=lidar();self.matcher=GlobalRelocalizer(self.map,self.lidar)
        self.pose=np.array([2.,2.8,-.7]);self.scan=grid_scan(self.map,self.lidar,self.pose)

    def test_branch_bounds_cover_children_and_outside_ray_endpoints(self):
        rng=np.random.default_rng(12071)
        offsets=rng.integers(-90,90,size=(5,60,2))
        for level in range(1,5):
            size=2**level
            for _ in range(12):
                r=int(rng.integers(self.map.height));c=int(rng.integers(self.map.width));y=int(rng.integers(5))
                rr,cc=np.meshgrid(np.arange(r,r+size),np.arange(c,c+size),indexing='ij')
                children=np.c_[np.full(rr.size,y),rr.ravel(),cc.ravel()]
                bound=self.matcher._scores([[y,r,c]],offsets,level)[0]
                scores=self.matcher._scores(children,offsets,0)
                if np.isfinite(scores).any():self.assertGreaterEqual(bound+1e-7,np.max(scores))

    def test_global_search_recovers_without_initial_pose(self):
        result=finish(self.matcher.start(self.scan))
        self.assertTrue(result['search_complete']);self.assertTrue(result['candidate_valid'],result)
        self.assertLess(np.linalg.norm(np.array(result['pose'])[:2]-self.pose[:2]),.12)
        self.assertLess(abs(wrap(result['pose'][2]-self.pose[2])),.04)

    def test_branch_search_matches_exhaustive_lattice_score(self):
        # Smaller yaw lattice only for an exhaustive algorithm check. The
        # production contract always enumerates all 180 angles.
        self.matcher.yaws=np.array([0.,np.pi/2,np.pi,-np.pi/2])
        scan=grid_scan(self.map,self.lidar,[2.,2.8,0.])
        points,_=self.matcher.body_points(scan);offsets=[]
        for yaw in self.matcher.yaws:
            c,s=np.cos(yaw),np.sin(yaw);xy=points@np.array([[c,s],[-s,c]])
            offsets.append(np.rint(np.c_[-xy[:,1],xy[:,0]]/self.map.res).astype(int))
        offsets=np.asarray(offsets);rows,cols=np.indices(self.map.occupied.shape)
        best=-np.inf
        for yi in range(4):
            nodes=np.c_[np.full(rows.size,yi),rows.ravel(),cols.ravel()]
            for start in range(0,len(nodes),128):
                best=max(best,float(self.matcher._scores(nodes[start:start+128],offsets,0).max()))
        result=finish(self.matcher.start(scan))
        self.assertTrue(result['search_complete'])
        self.assertAlmostEqual(result['modes'][0]['score'],best,places=6)

    def test_incomplete_search_never_authorizes_reset(self):
        # Force the budget, rather than allowing a nearly complete job to be
        # mistaken for a unique solution on a map that happens to be easy.
        self.matcher.config=dict(self.matcher.config,maximum_nodes=1)
        result=finish(self.matcher.start(self.scan))
        self.assertFalse(result['candidate_valid']);self.assertFalse(result['search_complete'])
        self.assertEqual(result['reason'],'search_budget_exhausted')

    def test_symmetric_room_does_not_select_arbitrary_pose(self):
        m=toy_map(True);matcher=GlobalRelocalizer(m,self.lidar)
        scan=grid_scan(m,self.lidar,[1.5,1.,.3]);result=finish(matcher.start(scan))
        self.assertFalse(result['candidate_valid']);self.assertEqual(result['reason'],'ambiguous_map')
        self.assertGreaterEqual(len(result['modes']),2)

    def test_missing_and_unknown_are_not_static_evidence(self):
        for scan in [np.full(220,np.nan),np.full(220,np.inf),np.zeros(220)]:
            result=finish(self.matcher.start(scan));self.assertFalse(result['candidate_valid'])
            self.assertEqual(result['reason'],'insufficient_scan_hits')
        self.map.unknown[:]=True
        self.assertFalse(self.matcher.verify(self.pose,self.scan)['valid'])

    def test_unmapped_returns_do_not_mutate_avoidance_scan(self):
        scan=self.scan.copy();scan[80:120]=.45;original=scan.copy()
        check=self.matcher.verify(self.pose,scan)
        self.assertTrue(check['valid'],check);np.testing.assert_array_equal(scan,original)

    def test_finite_rays_cannot_pass_through_mapped_walls(self):
        check=self.matcher.verify(self.pose,np.minimum(self.scan+3.,9.8),refine=False)
        self.assertFalse(check['valid']);self.assertGreater(check['ray_contradiction_fraction'],.1)

    def test_fresh_stationary_confirmations_and_one_shot_result(self):
        recovery=StoppedPoseRecovery(self.matcher);odom=np.array([15.,-6.,1.2])
        recovery.begin(self.scan,odom,[0.,0.],0.)
        while recovery.state=='searching':recovery.advance()
        self.assertEqual(recovery.state,'confirming',recovery.diagnostics())
        for _ in range(10):recovery.observe(self.scan,odom,[0.,0.],0.)
        self.assertEqual(recovery.confirmations,0)
        for t in [.2,.4]:self.assertEqual(recovery.observe(self.scan,odom,[0.,0.],t),'confirming')
        self.assertEqual(recovery.observe(self.scan,odom,[0.,0.],.6),'ready')
        recovered=recovery.take_pose(.6)
        self.assertLess(np.linalg.norm(recovered[:2]-self.pose[:2]),.12)
        with self.assertRaises(RuntimeError):recovery.take_pose(.6)

    def test_motion_stale_frames_and_lost_confirmation_keep_hold(self):
        for speed,received,reason in [([.1,0.],0.,'robot_not_stopped'),([0.,0.],.5,'stale_or_future_scan')]:
            recovery=StoppedPoseRecovery(self.matcher);recovery.begin(self.scan,np.zeros(3),speed,0.,received)
            self.assertEqual(recovery.state,'held');self.assertEqual(recovery.reason,reason)
        recovery=StoppedPoseRecovery(self.matcher);recovery.begin(self.scan,np.zeros(3),[0.,0.],0.)
        while recovery.state=='searching':recovery.advance()
        recovery.observe(np.full(220,np.nan),np.zeros(3),[0.,0.],.2)
        self.assertEqual(recovery.state,'held');self.assertIsNone(recovery.pose)

    def test_confirmed_pose_expires(self):
        recovery=StoppedPoseRecovery(self.matcher);recovery.state='ready'
        recovery.pose=self.pose.copy();recovery.last_timestamp=1.
        with self.assertRaises(RuntimeError):recovery.take_pose(1.3)
        self.assertEqual(recovery.reason,'expired_confirmation')

    def test_relative_odometry_uses_odom_axes_and_lateral_motion(self):
        pose=np.array([1.,2.,np.pi/2]);previous=np.array([100.,-20.,0.])
        odom=previous+[.1,.02,.03]
        result=propagate_pose(pose,previous,odom)
        np.testing.assert_allclose(result,[.98,2.1,pose[2]+.03],atol=1e-12)

    def test_runtime_handoff_clears_histories_and_preserves_directional_goal(self):
        import json,shutil,tempfile
        from pathlib import Path
        from types import SimpleNamespace
        from navrl.geometry import ROOT
        from neural_navigation.core import NavigationCore
        from neural_navigation.fast_static_localizer import temporal_contract
        from neural_navigation.global_relocalizer import apply_confirmed_pose
        from neural_navigation.predictive_constraint import cumulative_contract
        from neural_navigation.quality_prediction_context import QualityPredictionContext
        from neural_navigation.runtime import EmergencyStop
        from neural_navigation.safety_context import SafetyContext,PREDICTIVE_SCHEMA
        with tempfile.TemporaryDirectory() as directory:
            bundle=Path(directory)
            shutil.copy2(ROOT.parent/'models/quality_recovery_20261014/portable/router.pt',bundle/'router.pt')
            (bundle/'localization_contract.json').write_text(json.dumps(temporal_contract()))
            core=NavigationCore(self.map,self.lidar,bundle,route_mode='shortcut')
            goal=np.r_[self.map.xy(np.argwhere(self.map.free)[100]),1.1]
            core.reset([2.,2.,0.],goal)
            guard=EmergencyStop(memory_config={'predictive_action_constraint':cumulative_contract()})
            context=SafetyContext(PREDICTIVE_SCHEMA);context.last_executed[:]=1.;context.blocked_since=1.
            policy=SimpleNamespace(prediction=QualityPredictionContext(guard.half*2),last_observation=np.ones(303))
            runtime=SimpleNamespace(core=core,guard=guard,context=context,policy=policy)
            core.router_cache['old_pose']=None;core.expiry[:]=123.;core.history.append(np.full(72,-123.))
            guard.reverse_started=1.;guard.predictive_constraint.tracker.tracks=[dict(id='old')]
            odom=np.array([15.,-6.,1.2]);recovery=StoppedPoseRecovery(self.matcher)
            recovery.begin(self.scan,odom,[0.,0.],10.)
            while recovery.state=='searching':recovery.advance()
            for t in [10.2,10.4,10.6]:recovery.observe(self.scan,odom,[0.,0.],t)
            result=apply_confirmed_pose(runtime,recovery,self.scan,odom,np.zeros(2),10.6)
            self.assertEqual(result['command'],[0.,0.]);np.testing.assert_allclose(result['goal'],goal,atol=1e-12,rtol=0)
            self.assertEqual(core.elapsed,10.6);self.assertAlmostEqual(core.goal_yaw,1.1)
            np.testing.assert_array_equal(core.localizer.last_odom,odom)
            np.testing.assert_array_equal(core.last_action,np.zeros(2))
            self.assertNotIn('old_pose',core.router_cache);self.assertFalse(np.any(core.expiry==123.))
            self.assertFalse(any(np.any(frame==-123.) for frame in core.history))
            self.assertIsNone(guard.reverse_started);self.assertEqual(guard.predictive_constraint.tracker.tracks,[])
            self.assertIsNone(runtime.context.blocked_since);self.assertIsNone(policy.last_observation)
            self.assertEqual(recovery.state,'consumed')
            # A later frame starts from the measured odom anchor, not zero.
            core.update(self.scan,odom,np.zeros(2),10.8)
            self.assertFalse(core.odometry_sanity_failure)
            self.assertLess(np.linalg.norm(core.estimate[:2]-self.pose[:2]),.12)


if __name__=='__main__':unittest.main()
