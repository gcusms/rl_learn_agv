import unittest
from types import SimpleNamespace
import numpy as np
from neural_navigation.predictive_constraint import PredictiveConstraint,contract,cumulative_contract
from neural_navigation.sensors import LidarSpec


def core():
    lidar=LidarSpec(dict(beams=220,angle_min=-1.92,angle_max=1.92,range_min=.1,range_max=12.,mount=[0.,0.,0.,0.]))
    return SimpleNamespace(lidar=lidar,elapsed=0.,odom=np.zeros(3),estimate=np.zeros(3),
        scan=np.full(220,np.nan),measured_velocity=np.zeros(2),odometry_sanity_failure=False)


def scan_circle(c,center,radius=.18):
    # Static geometry in the continuous odometry frame, with exact ray returns.
    yaw=c.odom[2];angles=c.lidar.angles+yaw
    rays=np.c_[np.cos(angles),np.sin(angles)];offset=np.asarray(center)-c.odom[:2]
    projection=rays@offset;disc=projection**2-(offset@offset-radius**2)
    c.scan=np.full(c.lidar.beams,np.nan);valid=(disc>=0)&(projection>0)
    c.scan[valid]=projection[valid]-np.sqrt(disc[valid])


class PredictiveConstraintTests(unittest.TestCase):
    def test_probe_masks_are_stricter_and_match_execution_on_the_same_frame(self):
        from neural_navigation.runtime import EmergencyStop,execute_policy_action
        from neural_navigation.safety_context import SafetyContext,PREDICTIVE_SCHEMA
        from neural_navigation.masked_actions import ACTIONS,COMMANDS,masks_from_observation
        c=core()
        c.localizer=SimpleNamespace(std=np.array([.02,.02,.01]),scan_update_accepted=False,map_agreement=0.)
        c.static_map=SimpleNamespace(free=np.ones((1,1)),lookup=lambda *a,**k:True)
        c.diagnostics=lambda:dict(route_complete=True,estimated_goal_pose_reached=False)
        c.velocity_command=lambda action:np.asarray(action)*[.2,.5]
        c.record_action=lambda action:setattr(c,'last_action',np.asarray(action).copy())
        legacy=EmergencyStop(memory_config=dict(near_range_m=1.5))
        guard=EmergencyStop(memory_config=dict(near_range_m=1.5,predictive_action_constraint=contract()))
        context=SafetyContext(PREDICTIVE_SCHEMA)
        for t in (0.,.2,.4,.6):
            c.elapsed=t;scan_circle(c,[.82-.2*t,0.]);guard.predictive_constraint.prepare(c)
        base=np.zeros(235,np.float32)
        old_mask=masks_from_observation(SafetyContext(PREDICTIVE_SCHEMA).augment(base,c,legacy,c.elapsed))
        obs=context.augment(base,c,guard,c.elapsed);new_mask=masks_from_observation(obs)
        self.assertTrue(new_mask[0]);self.assertTrue(np.all(~new_mask|old_mask))
        self.assertTrue(np.any(old_mask&~new_mask))
        hits=[t['hits'] for t in guard.predictive_constraint.tracker.tracks]
        for index in np.flatnonzero(new_mask):
            result=execute_policy_action(c,guard,ACTIONS[index],c.elapsed)
            np.testing.assert_allclose(result['command'],COMMANDS[index],rtol=0,atol=2e-8)
            np.testing.assert_allclose(c.last_action,ACTIONS[index],rtol=0,atol=2e-8)
        self.assertEqual(hits,[t['hits'] for t in guard.predictive_constraint.tracker.tracks])

    def test_hardware_and_action_contract_cannot_mix_forecast_versions(self):
        from neural_navigation.masked_actions import validate_hardware_action,action_contract
        hardware=dict(policy_action=action_contract(True),near_field_memory=dict(predictive_action_constraint=cumulative_contract()))
        with self.assertRaises(ValueError):validate_hardware_action(hardware)
        hardware['policy_action']=action_contract('cumulative');self.assertTrue(validate_hardware_action(hardware))

    def test_cumulative_occlusion_keeps_explicit_aged_reachable_hypothesis(self):
        c=core();p=PredictiveConstraint(cumulative_contract(),[.394,.329],.3)
        for t in (0.,.2,.4,.6):
            c.elapsed=t;scan_circle(c,[.82-.2*t,0.]);p.prepare(c)
        self.assertGreater(p.supported_tracks,0)
        c.elapsed=1.6;c.scan[:]=np.nan;p.prepare(c)
        self.assertEqual(len(p.tracker.tracks),0)
        self.assertTrue(any(t['retained_without_measurement'] and t['source']=='unknown' and t['age']>.8 for t in p.tracks))
        self.assertIsNone(p.rejection(c,[0.,0.]))
        c.elapsed=1.9;p.prepare(c);self.assertEqual(len(p.tracks),0)

    def test_approach_vetoes_fast_action_before_current_contact_and_keeps_stop(self):
        c=core();p=PredictiveConstraint(contract(),[.394,.329],.3)
        for t in (0.,.2,.4,.6):
            c.elapsed=t;scan_circle(c,[.82-.2*t,0.]);p.prepare(c)
        self.assertGreater(p.supported_tracks,0)
        self.assertEqual(p.rejection(c,[.2,0.]),'predicted_dynamic_contact')
        self.assertIsNone(p.rejection(c,[0.,0.]))
        self.assertGreater(float(np.nanmin(c.scan))-.394/2,.04)

    def test_static_geometry_and_mecanum_odometry_do_not_become_motion(self):
        c=core();p=PredictiveConstraint(contract(),[.394,.329],.3)
        for t in np.arange(0.,1.6,.2):
            c.elapsed=float(t);c.odom=np.array([.1*t,.025*t,.05*t]);scan_circle(c,[1.,.15]);p.prepare(c)
        self.assertEqual(p.supported_tracks,0);self.assertTrue(np.isinf(p.contacts).all())

    def test_map_pose_jump_cannot_change_odometry_frame_forecast(self):
        c=core();a=PredictiveConstraint(contract(),[.394,.329],.3);b=PredictiveConstraint(contract(),[.394,.329],.3)
        for t in (0.,.2,.4,.6):
            c.elapsed=t;scan_circle(c,[.82-.2*t,0.]);a.prepare(c)
            c.estimate+=np.array([5.,-2.,.5]);b.prepare(c)
        np.testing.assert_array_equal(a.contacts,b.contacts)

    def test_frame_cache_and_missing_scan_tracks_expire(self):
        c=core();p=PredictiveConstraint(contract(),[.394,.329],.3)
        for t in (0.,.2,.4,.6):
            c.elapsed=t;scan_circle(c,[.82-.2*t,0.]);p.prepare(c)
        contacts=p.contacts.copy();hits=[t['hits'] for t in p.tracker.tracks]
        p.prepare(c);np.testing.assert_array_equal(p.contacts,contacts);self.assertEqual(hits,[t['hits'] for t in p.tracker.tracks])
        c.elapsed=1.5;c.scan[:]=np.nan;p.prepare(c)
        self.assertEqual(p.supported_tracks,0);self.assertEqual(len(p.tracker.tracks),0)


if __name__=='__main__':unittest.main()
