import json
from pathlib import Path
import tempfile
import unittest
import shutil
from navrl.geometry import ROOT
from experiments.compare_context_trials import manifests,compare,read_trial


class ContextComparisonTests(unittest.TestCase):
    def test_shared_worlds_are_deduplicated_but_changed_seed_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            a=Path(directory)/'a.json';b=Path(directory)/'b.json'
            case=dict(id='same',scene='assets/room_navigation_scripted',seed=1)
            a.write_text(json.dumps(dict(kind='development',cases=[case])))
            b.write_text(json.dumps(dict(kind='training_failure_replay',cases=[case])))
            _,unique=manifests([a,b]);self.assertEqual(len(unique),1)
            b.write_text(json.dumps(dict(kind='development',cases=[dict(case,seed=2)])))
            with self.assertRaises(ValueError):manifests([a,b])

    def test_final_acceptance_is_not_consumed_by_development_comparison(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'acceptance.json'
            path.write_text(json.dumps(dict(kind='acceptance',cases=[])))
            with self.assertRaises(ValueError):manifests([path])

    def trial(self,seconds,second_safe):
        return dict(physical_hardware={'forward_limit_m_s':.2},learned_component_hashes={'localizer':'same'},
            summary={},trace_analysis={},cases={
                'shared':dict(safe_success=True,seconds=seconds,travel_m=2.,emergency_override_steps=0),
                'other':dict(safe_success=second_safe,seconds=.1,travel_m=.1,emergency_override_steps=0)})

    def test_failure_time_is_not_counted_as_faster_navigation(self):
        result=compare(dict(old=self.trial(10.,True),new=self.trial(20.,False)))
        self.assertEqual(result['paired_common_safe_efficiency']['old']['mean_seconds'],10.)
        self.assertEqual(result['paired_common_safe_efficiency']['new']['mean_seconds'],20.)
        self.assertFalse(result['deployment_approved'])

    def test_different_physical_limit_prevents_comparison(self):
        old=self.trial(10.,True);new=self.trial(8.,True)
        new['physical_hardware']['forward_limit_m_s']=.3
        with self.assertRaises(ValueError):compare(dict(old=old,new=new))

    def test_real_completed_checkpoint_reports_verify_and_actor_tampering_is_rejected(self):
        source=ROOT/'runs/safety_context_20261011/checkpoint_v2_24000'
        known,expected=manifests([ROOT/'config/navigation_safety_context_smoke.json'])
        with tempfile.TemporaryDirectory() as directory:
            run=Path(directory);(run/'candidate').mkdir()
            shutil.copytree(source/'portable',run/'portable')
            shutil.copy2(ROOT/'runs/safety_context_20261011/pilot_v2/candidate/policy_24000.zip',run/'candidate/policy.zip')
            shutil.copy2(source/'runtime.json',run/'runtime_0.json')
            shutil.copytree(source/'traces',run/'traces_0')
            shutil.copy2(source/'status.json',run/'status.json')
            report=read_trial(run,known,expected)
            self.assertEqual(report['summary']['unique_tasks'],3)
            self.assertEqual(report['summary']['safe_success'],2)
            self.assertFalse(report['summary']['development_gate_passed'])
            with (run/'portable/actor.pt').open('ab') as f:f.write(b'changed')
            with self.assertRaises(ValueError):read_trial(run,known,expected)


if __name__=='__main__':unittest.main()
