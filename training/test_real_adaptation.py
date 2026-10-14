import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from navrl.geometry import ROOT,wrap
from navrl.map import StaticMap
from neural_navigation.sensors import LidarSpec
from neural_navigation.odom_anchored_localizer import OdometryAnchoredLocalizer,contract as loc_contract
from neural_navigation.goal_pose import append_goal_features,goal_status,update_goal_settling
from neural_navigation.core import NavigationCore
from neural_navigation.runtime import SensorNavigationRuntime,execute_policy_action
from neural_navigation.masked_actions import masks_from_observation,ACTIONS
from experiments.goal_pose_navigation import GoalPoseWrapper,migrate_goal_policy,export_goal_policy,GoalPoseRuntime
from experiments.masked_navigation import MaskedNavigationCurriculum
from experiments.quality_masked_navigation import QualityMaskedWrapper
from sb3_contrib import MaskablePPO
from neural_navigation.timed_scan_history import TimedScanHistory

PROJECT=Path(__file__).resolve().parents[1]
BUNDLE=PROJECT/'models/quality_recovery_20261014'


class RealAdaptationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.lidar=LidarSpec(json.loads((BUNDLE/'hardware.json').read_text())['lidar'])
        cls.map=StaticMap(ROOT/'assets/room_navigation_scripted/decision_map.yaml',radius=.3)

    def localizer(self):
        loc=OdometryAnchoredLocalizer(self.map,self.lidar,BUNDLE/'localizer.pt',loc_contract())
        loc.reset([1.2,1.2,0.],None);return loc

    def test_unmapped_foreground_cannot_pull_pose_and_scan_is_preserved(self):
        loc=self.localizer();scan=np.full(self.lidar.beams,.7);original=scan.copy()
        for t in np.arange(0,3,.1):loc.update(np.zeros(3),scan,float(t))
        np.testing.assert_array_equal(scan,original)
        np.testing.assert_allclose(loc.pose,[1.2,1.2,0.],atol=1e-12)
        self.assertFalse(loc.scan_update_accepted)
        self.assertEqual(loc.correction_reason,'insufficient_static_support')

    def test_static_support_can_still_correct_a_small_initial_offset(self):
        truth=np.array([1.2,1.2,.1]);yaw=truth[2];m=self.lidar.mount
        origin=truth[:2]+[np.cos(yaw)*m[0]-np.sin(yaw)*m[1],np.sin(yaw)*m[0]+np.cos(yaw)*m[1]]
        ranges=np.arange(self.lidar.range_min,self.lidar.range_max,.005)
        directions=np.c_[np.cos(yaw+m[3]+self.lidar.angles),np.sin(yaw+m[3]+self.lidar.angles)]
        ends=origin+ranges[:,None,None]*directions[None,:,:]
        hits=self.map.lookup(ends,self.map.occupied,outside=False)
        scan=np.where(hits.any(axis=0),ranges[np.argmax(hits,axis=0)],np.inf)
        loc=self.localizer();loc.reset(truth+[.04,.04,.02],None);accepted=0
        for t in np.arange(0,5,.1):
            loc.update(np.zeros(3),scan,float(t));accepted+=loc.scan_update_accepted
        self.assertGreater(accepted,0)
        self.assertLess(np.linalg.norm(loc.pose[:2]-truth[:2]),np.hypot(.04,.04))

    def test_expired_candidate_does_not_become_executed_action_history(self):
        import importlib.util
        from neural_navigation.safety_context import SafetyContext
        spec=importlib.util.spec_from_file_location('execution_feedback',PROJECT/'deployment/ros2_recording_adapter/execution_feedback.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        context=SafetyContext(json.loads((BUNDLE/'hardware.json').read_text())['policy_observation']['schema'])
        context.record(dict(command=[.2,.5],proposed_action=[1.,1.],timestamp=.2,safety_override=False,stop_reason=None))
        core=SimpleNamespace(record_action=lambda a:setattr(core,'last_action',np.asarray(a)))
        runtime=SimpleNamespace(core=core,context=context)
        module.restore_published_history(runtime,[0.,0.],'external_stop',.4)
        np.testing.assert_array_equal(core.last_action,[0.,0.])
        np.testing.assert_array_equal(context.last_executed,[0.,0.])
        np.testing.assert_array_equal(context.last_proposal,[1.,1.])
        self.assertEqual(context.last_reason,'other')

    def test_turning_odometry_rotates_local_translation_into_map(self):
        loc=self.localizer();loc.reset([2,2,np.pi/2],None)
        scan=np.full(self.lidar.beams,np.nan)
        loc.update([0,0,0],scan,0);loc.update([.1,0,.2],scan,.2)
        np.testing.assert_allclose(loc.pose,[2,2.1,np.pi/2+.2],atol=1e-12)

    def test_single_wall_validates_pose_without_inventing_along_wall_motion(self):
        distance=object();nx=object();ny=object()
        def lookup(points,field,outside=0.):
            points=np.asarray(points)
            if field is distance:return abs(points[...,0]-3.)
            if field is nx:return np.ones(points.shape[:-1])
            if field is ny:return np.zeros(points.shape[:-1])
            return np.ones(points.shape[:-1],bool)
        m=SimpleNamespace(distance=distance,lookup=lookup,blocked=np.zeros((4,4),bool),_alignment_normals=(nx,ny))
        loc=OdometryAnchoredLocalizer(m,self.lidar,BUNDLE/'localizer.pt',loc_contract())
        loc.reset([0.,0.,0.],None)
        with np.errstate(divide='ignore'):
            scan=(3.-self.lidar.mount[0])/np.cos(self.lidar.mount[3]+self.lidar.angles)
        scan[(scan<self.lidar.range_min)|(scan>=self.lidar.range_max)]=np.inf
        for t in np.arange(0,2,.1):loc.update(np.zeros(3),scan,float(t))
        self.assertTrue(loc.scan_update_accepted)
        self.assertEqual(loc.correction_reason,'validated_static_prior')
        np.testing.assert_allclose(loc.pose,[0.,0.,0.],atol=1e-12)
        self.assertGreater(loc.std[1],.12)  # wall tangent remains unobserved

    def test_missing_uncertainty_depends_on_time_not_frame_count(self):
        scans=np.full(self.lidar.beams,np.nan);std=[]
        for n in (10,20):
            loc=self.localizer()
            for t in np.linspace(0,2,n+1):loc.update(np.zeros(3),scans,float(t))
            std.append(loc.std)
        np.testing.assert_allclose(*std,atol=1e-12)

    def test_duplicate_timestamps_do_not_accumulate_correction(self):
        loc=self.localizer();scan=np.full(self.lidar.beams,np.nan)
        for _ in range(20):loc.update(np.zeros(3),scan,1.)
        np.testing.assert_array_equal(loc.std,[.12,.12,.08])
        with self.assertRaises(ValueError):loc.update(np.zeros(3),scan,.9)

    def test_history_physical_offsets_match_at_5_and_10_hz(self):
        outputs=[]
        for period in (.2,.1):
            history=TimedScanHistory()
            for t in np.arange(0,1.01,period):output=history.update(float(t),np.array([t]))
            outputs.append(output)
        np.testing.assert_allclose(outputs[0],outputs[1],atol=1e-9)
        np.testing.assert_allclose(outputs[0],[[.6],[.8],[1.]])
        history.reset();np.testing.assert_array_equal(history.update(2.,[3]),[[3],[3],[3]])

    def test_goal_yaw_wrap_settling_and_motion(self):
        c=SimpleNamespace(estimate=np.array([1.,1.,np.pi-.02]),goal=np.array([1.,1.]),
            goal_yaw=-np.pi+.02,measured_velocity=np.zeros(2),elapsed=0.,_goal_settled_since=None)
        np.testing.assert_allclose(append_goal_features(np.zeros(300),c)[300:],[np.sin(.04),np.cos(.04),1],atol=1e-7)
        update_goal_settling(c);self.assertFalse(goal_status(c)['estimated_goal_pose_reached'])
        c.elapsed=.4;update_goal_settling(c);self.assertTrue(goal_status(c)['estimated_goal_pose_reached'])
        c.measured_velocity[1]=.2;update_goal_settling(c)
        self.assertFalse(goal_status(c)['estimated_goal_pose_reached'])

    def test_wrong_heading_is_not_arrival_and_reset_clears_settling(self):
        core=NavigationCore(self.map,self.lidar,BUNDLE)
        core.reset([1.2,1.2,0],[1.2,1.2,np.pi/2]);core.measured_velocity[:]=0
        update_goal_settling(core)
        self.assertFalse(goal_status(core)['estimated_goal_pose_reached'])
        core.goal_yaw=0.;update_goal_settling(core);core.elapsed=1
        self.assertTrue(goal_status(core)['estimated_goal_pose_reached'])
        core.set_goal([1.2,1.2,0.]);self.assertIsNone(core._goal_settled_since)
        core.set_goal([1.2,1.2]);self.assertTrue(goal_status(core)['estimated_goal_pose_reached'])

    def test_runtime_does_not_override_actor_rotation_at_wrong_heading(self):
        c=SimpleNamespace(elapsed=0.,velocity_command=lambda a:np.array([0.,.5]),record_action=lambda a:None,
            diagnostics=lambda:dict(estimated_goal_distance_m=0.,estimated_goal_pose_reached=False))
        guard=SimpleNamespace(check=lambda *a:(np.array([0.,.5]),None),memory=None)
        result=execute_policy_action(c,guard,[0.,1.],0.)
        self.assertEqual(result['command'],[0.,.5]);self.assertIsNone(result['stop_reason'])

    def test_legacy_actor_rejects_heading_task(self):
        with self.assertRaisesRegex(ValueError,'Directional goals'):
            SensorNavigationRuntime(BUNDLE/'portable',ROOT/'assets/room_navigation_scripted/decision_map.yaml',[1.2,1.2,0],[1.2,1.2,1.])

    def test_migration_masks_export_runtime_and_optimizer(self):
        old=MaskablePPO.load(BUNDLE/'policy_2000.zip',device='cpu')
        env=GoalPoseWrapper(QualityMaskedWrapper(MaskedNavigationCurriculum(BUNDLE,stage='static',max_steps=10),
            reward_scale=old.training_reward_scale),gamma=old.gamma)
        try:
            model=migrate_goal_policy(BUNDLE/'policy_2000.zip',env)
            protected=torch.zeros(2,303);protected[:,250]=1;protected[:,234]=1;protected[:,232]=1
            protected[:,294]=1;protected[:,302]=1;protected[0,218]=.1
            with torch.no_grad():model.reference_policy.action_net.bias[1]=100.
            np.testing.assert_array_equal(model.eligible(protected,torch.ones(2,12,dtype=torch.bool)).numpy(),[True,False])
            rng=np.random.default_rng(12);obs=rng.uniform(-1,1,(100,300)).astype(np.float32)
            obs[:,235:244]=rng.integers(0,2,(100,9));obs[:,267:269]=rng.integers(0,2,(100,2))
            expanded=np.c_[obs,rng.uniform(-1,1,(100,3))].astype(np.float32);masks=masks_from_observation(obs[:,:282])
            np.testing.assert_array_equal(old.predict(obs,action_masks=masks,deterministic=True)[0],model.predict(expanded,action_masks=masks,deterministic=True)[0])
            name='mlp_extractor.policy_net.0.weight';p=dict(model.policy.named_parameters())[name]
            oldp=dict(old.policy.named_parameters())[name]
            np.testing.assert_array_equal(model.policy.optimizer.state[p]['exp_avg'][:,:300].numpy(),old.policy.optimizer.state[oldp]['exp_avg'].numpy())
            self.assertEqual(model.policy.optimizer.state[p]['exp_avg'][:,300:].abs().sum(),0)
            with tempfile.TemporaryDirectory() as d:
                export_goal_policy(model,BUNDLE,Path(d)/'portable')
                rt=GoalPoseRuntime(Path(d)/'portable',ROOT/'assets/room_navigation_scripted/decision_map.yaml',[1.2,1.2,0],[1.2,1.2,1.])
                r=rt.step(np.full(self.lidar.beams,np.nan),np.zeros(3),np.zeros(2),.2,capture_observation=True)
                self.assertEqual(len(r['policy_observation']),303)
                self.assertNotEqual(r['stop_reason'],'estimated_goal_reached')
                # The same exported actor can be independently evaluated with
                # the odometry gate; neither default bundle is changed in place.
                components=Path(d)/'components';components.mkdir()
                import shutil
                for name in ('hardware.json','localizer.pt','router.pt'):shutil.copy2(BUNDLE/name,components/name)
                (components/'localization_contract.json').write_text(json.dumps(loc_contract()))
                export_goal_policy(model,components,Path(d)/'combined')
                combined=GoalPoseRuntime(Path(d)/'combined',ROOT/'assets/room_navigation_scripted/decision_map.yaml',[1.2,1.2,0],[1.2,1.2,1.])
                self.assertIsInstance(combined.core.localizer,OdometryAnchoredLocalizer)
                result=combined.step(np.full(self.lidar.beams,.7),np.zeros(3),np.zeros(2),.2,capture_observation=True)
                np.testing.assert_allclose(result['diagnostics']['estimated_pose'],[1.2,1.2,0.])
                self.assertEqual(len(result['policy_observation']),303)
                data=json.loads((Path(d)/'portable/goal_pose_observation.json').read_text());data['dimensions']=304
                (Path(d)/'portable/goal_pose_observation.json').write_text(json.dumps(data))
                with self.assertRaises(ValueError):GoalPoseRuntime(Path(d)/'portable',ROOT/'assets/room_navigation_scripted/decision_map.yaml',[1.2,1.2,0],[1.2,1.2,1.])
        finally:env.close()


if __name__=='__main__':unittest.main(verbosity=2)
