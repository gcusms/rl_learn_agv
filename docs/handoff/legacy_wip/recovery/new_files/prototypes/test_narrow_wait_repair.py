import unittest
from unittest.mock import patch
from types import SimpleNamespace
import numpy as np
from navrl.map import StaticMap
from neural_navigation.runtime import EmergencyStop
from neural_navigation.static_footprint import static_footprint_clear
from prototypes.anchored_narrow_runtime import feasible_anchor,AnchoredNarrowCore
from prototypes.narrow_passage import swept_segment_clear
from prototypes.narrow_recovery_guard import NarrowRecoveryGuard
from prototypes.narrow_alignment import AlignmentNarrowCore
from experiments.fresh_perception_runtime import FreshPerceptionCore
from prototypes.narrow_observation import ACTIONS

FOOTPRINT=np.array([.393924904,.329460022])


class WaitRepairTests(unittest.TestCase):
    def test_alignment_hint_never_claims_or_changes_a_motion_route(self):
        c=object.__new__(AlignmentNarrowCore);c.estimate=np.array([0.,0.,np.pi/2])
        c.goal=np.array([1.,0.]);c.predicted_path=np.array([[0.,0.]])
        with patch.object(FreshPerceptionCore,'_observation',return_value=np.zeros(235,np.float32)):
            obs=c._observation(np.ones(20))
        np.testing.assert_allclose(obs[216:218],[0.,-.075],atol=1e-7)
        np.testing.assert_array_equal(c.predicted_path,[[0.,0.]])
        self.assertEqual(c.policy_target_kind,'goal_alignment_hint')

    def test_finer_local_sweep_preserves_four_cm_margin(self):
        m=StaticMap('training/runs/narrow_long_50000_v2_20261013/scenes/gap_150/decision_map.yaml',radius=.3)
        pose=np.array([3.7223844925435463,2.00980786588347,.013073279117523002])
        end=pose[:2]+.08*np.array([np.cos(pose[2]),np.sin(pose[2])])
        self.assertTrue(static_footprint_clear(m,pose,FOOTPRINT,.04))
        self.assertFalse(swept_segment_clear(m,pose[:2],end,FOOTPRINT,.04))
        self.assertTrue(swept_segment_clear(m,pose[:2],end,FOOTPRINT,.04,max_spacing=.002))

    def test_nearly_coincident_backward_anchor_does_not_become_steering_target(self):
        c=self.core();c.grid_h=3;c.grid_w=4;c.goal=np.array([.25,0.])
        c.footprint=FOOTPRINT;c.margin=.04
        c.router=SimpleNamespace(directions=[(-1,0),(1,0),(0,-1),(0,1)])
        c.xy=lambda cell:np.array([[-.1,-.01,.12,.2][cell[1]],(1-cell[0])*.1])
        q=np.zeros((4,3,4));q[3]=1.
        path=AnchoredNarrowCore._decode_motion_route(c,q,np.ones((3,4),bool),(1,0),(1,3))
        self.assertGreaterEqual(np.linalg.norm(path[1]-path[0]),.05)
        self.assertTrue((path[1:,0]>0).all())
        np.testing.assert_array_equal(path[-1],c.goal)

    def test_body_blocked_learned_connection_is_never_decoded(self):
        c=self.core();c.grid_h=3;c.grid_w=4;c.goal=np.array([.25,0.])
        c.footprint=FOOTPRINT;c.margin=.04;c._segment_clear=lambda a,b:False
        c.router=SimpleNamespace(directions=[(-1,0),(1,0),(0,-1),(0,1)])
        c.xy=lambda cell:np.array([cell[1]*.1,(1-cell[0])*.1])
        path=AnchoredNarrowCore._decode_motion_route(c,np.ones((4,3,4)),np.ones((3,4),bool),(1,0),(1,3))
        np.testing.assert_array_equal(path,[[0.,0.]])

    def test_diagonal_only_entrance_anchor_is_replaced_by_connectable_horizontal_anchor(self):
        m=StaticMap('training/runs/narrow_long_50000_v2_20261013/scenes/gap_150/decision_map.yaml',radius=.3)
        anchors=[]
        for x in [1.81,1.89,1.97,2.05]:
            p=np.array([x,2.03])
            anchor=feasible_anchor(m,p,FOOTPRINT,.04,.08,lambda q:np.all(np.abs(q-p)<=.04))
            self.assertIsNotNone(anchor)
            self.assertTrue(static_footprint_clear(m,np.r_[anchor,0.],FOOTPRINT,.04))
            anchors.append(anchor)
        for a,b in zip(anchors[:-1],anchors[1:]):
            self.assertTrue(swept_segment_clear(m,a,b,FOOTPRINT,.04),(a,b))

    def core(self):
        m=SimpleNamespace(res=.02,origin=np.array([-2.,-2.,0.]),width=200,height=200,
                          blocked=np.zeros((200,200),bool),free=True)
        return SimpleNamespace(scan=np.full(30,2.),
            lidar=SimpleNamespace(range_min=.1,range_max=12.,angles=np.linspace(-.5,.5,30),mount=np.zeros(4)),
            localizer=SimpleNamespace(std=np.zeros(3),scan_update_accepted=True,map_agreement=1.),
            odometry_sanity_failure=False,estimate=np.zeros(3),odom=np.zeros(3),elapsed=1.,
            measured_velocity=np.zeros(2),static_map=m,unmatched_returns=0,
            laser_fusion=SimpleNamespace(forecast_cells=0,predictor=SimpleNamespace(tracks=[])),
            predicted_path=np.array([[0.,0.],[.2,0.]]),_segment_clear=lambda a,b:True,
            diagnostics=lambda:dict(route_complete=True))

    def guard(self,kind=NarrowRecoveryGuard):
        return kind(memory_config=dict(static_footprint_schema='oriented_rectangle_v1',
            static_footprint_margin_m=.04,clearance_increasing_rotation=True))

    def test_slow_parallel_translation_preserves_clearance_to_both_sides(self):
        c=self.core();points=np.array([[.06,.193],[-.06,-.197]])
        margin=np.full(2,.04);command=np.array([.05,0.])
        self.assertEqual(self.guard(EmergencyStop)._trajectory_check(c,command,points,margin,0.,True),'escape_not_increasing_clearance')
        self.assertIsNone(self.guard()._trajectory_check(c,command,points,margin,0.,True))

    def test_old_rear_corner_returns_do_not_forbid_safe_forward_progress(self):
        c=self.core();points=np.array([[-.2046,.1939],[-.198,.218]])
        margins=np.full(2,.07);command=np.array([.05,0.])
        self.assertEqual(self.guard(EmergencyStop)._trajectory_check(c,command,points,margins,0.,True),'escape_not_increasing_clearance')
        self.assertIsNone(self.guard()._trajectory_check(c,command,points,margins,0.,True))

    def test_safe_small_turn_does_not_require_arbitrary_five_mm_endpoint_gain(self):
        c=self.core();points=np.array([[.06,.193],[-.06,-.197]])
        for command in [[0.,-.1],[.05,-.1]]:
            self.assertEqual(self.guard(EmergencyStop)._trajectory_check(c,np.array(command),points,np.full(2,.04),0.,True),'escape_not_increasing_clearance')
            self.assertIsNone(self.guard()._trajectory_check(c,np.array(command),points,np.full(2,.04),0.,True))

    def test_float32_actor_commands_have_same_recovery_admission_as_probes(self):
        c=self.core();points=np.array([[.06,.193],[-.06,-.197]])
        for index in [13,17]:
            command=ACTIONS[index].astype(float)*[.2,.5]
            self.assertIsNone(self.guard()._trajectory_check(c,command,points,np.full(2,.04),0.,True))
        c.diagnostics=lambda:dict(route_complete=False)
        command=ACTIONS[17].astype(float)*[.2,.5]
        actual,reason=self.guard().check(c,command,1.,1.)
        np.testing.assert_array_equal(actual,command);self.assertIsNone(reason)

    def test_front_obstacle_and_side_closing_turn_remain_blocked(self):
        c=self.core();g=self.guard()
        for command,points in [([.05,0.],[[.22,.193]]),([.05,.1],[[.06,.193]])]:
            self.assertIsNotNone(g._trajectory_check(c,np.array(command),np.array(points),np.array([.04]),0.,True))

    def test_unknown_motion_stale_scan_and_bad_alignment_disable_preserving_motion(self):
        points=np.array([[.06,.193]]);margins=np.array([.04]);command=np.array([.05,0.])
        for key in ['moving','unconfirmed','nan_velocity','scan','map','nan_map','age','new_obstacle','too_close','fast']:
            c=self.core();age=0.;p=points.copy();v=command.copy()
            if key=='moving':c.laser_fusion.predictor.tracks=[dict(hits=3,velocity=np.array([.1,0.]))]
            if key=='unconfirmed':c.laser_fusion.predictor.tracks=[dict(hits=1,velocity=np.zeros(2))]
            if key=='nan_velocity':c.laser_fusion.predictor.tracks=[dict(hits=3,velocity=np.array([np.nan,0.]))]
            if key=='scan':c.localizer.scan_update_accepted=False
            if key=='map':c.localizer.map_agreement=.7
            if key=='nan_map':c.localizer.map_agreement=float('nan')
            if key=='age':age=.3
            if key=='new_obstacle':c.unmatched_returns=1
            if key=='too_close':p[0,1]=.18
            if key=='fast':v[0]=.1
            self.assertIsNotNone(self.guard()._trajectory_check(c,v,p,margins,age,True),key)

    def test_checked_local_prefix_allows_only_bounded_forward_motion(self):
        c=self.core();c.diagnostics=lambda:dict(route_complete=False)
        command,reason=self.guard().check(c,[.05,0.],1.,1.)
        np.testing.assert_allclose(command,[.05,0.]);self.assertIsNone(reason)
        command,_=self.guard().check(c,[.1,0.],1.,1.)
        np.testing.assert_allclose(command,[0.,0.])
        c._segment_clear=lambda a,b:False
        command,_=self.guard().check(c,[.05,0.],1.,1.)
        np.testing.assert_allclose(command,[0.,0.])

    def test_local_prefix_cannot_bypass_missing_scan_or_pose_safety(self):
        c=self.core();c.diagnostics=lambda:dict(route_complete=False);g=self.guard()
        command,reason=g.check(c,[.05,0.],1.,2.)
        np.testing.assert_allclose(command,0.);self.assertEqual(reason,'stale_or_future_scan')
        command,reason=g.check(c,[.05,0.],float('nan'),1.)
        np.testing.assert_allclose(command,0.);self.assertEqual(reason,'nonfinite_input')
        c.scan[:]=np.nan
        command,reason=g.check(c,[.05,0.],1.,1.)
        np.testing.assert_allclose(command,0.);self.assertEqual(reason,'scan_unavailable')
        c.scan[:]=2.;c.odometry_sanity_failure=True
        command,reason=g.check(c,[.05,0.],1.,1.)
        np.testing.assert_allclose(command,0.);self.assertEqual(reason,'odometry_jump')


if __name__=='__main__':unittest.main(verbosity=2)
